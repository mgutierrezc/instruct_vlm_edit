import re
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage

from .utils import parent_module, brackets_to_periods


class IKE_CLIP(nn.Module):
    """CLIP-style rationale retriever for `revlm`.

    High-level behavior:
    - At edit time, build a small corpus of (image, question, rationale_sentence) pairs:
      - Use each example's own COT/rationale, split into sentences.
      - For each sentence, optionally generate a counterfactual sentence and pair it with
        a black image.
    - Compute `<image, question>` features from a chosen VLM layer (same "average key"
      construction used in BalancEdit / GRACE).
    - Compute sentence embeddings with Sentence-BERT.
    - Train two small projection heads with a CLIP-style loss to align these spaces.
    - Store the projected rationale embeddings as a retrieval index.
    - For each edit example, retrieve top‑k rationale sentences and prepend them as
      "New Facts" to the prompt.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        editor_cfg = getattr(config, "editor", config)

        # Model setup
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))

        # CLIP configuration
        self.seed = int(getattr(config, "seed", 333))
        self.image_only = bool(getattr(editor_cfg, "image_only", False))
        self.k = int(getattr(editor_cfg, "k", 3))
        self.clip_dim = int(getattr(editor_cfg, "clip_dim", 256))
        self.num_epochs = int(getattr(editor_cfg, "clip_epochs", 50))
        self.batch_size = int(getattr(editor_cfg, "clip_batch_size", 10))
        self.lr = float(getattr(editor_cfg, "clip_lr", 1e-4))
        self.temperature = float(getattr(editor_cfg, "clip_temperature", 1))

        # Counterfactual configuration
        self.include_counterfactuals_in_index = bool(getattr(editor_cfg, "include_counterfactuals_in_index", True))
        self.num_counterfacts_per_sentence = int(getattr(editor_cfg, "num_counterfacts_per_sentence", 1))
        self.max_pairs = int(getattr(editor_cfg, "max_pairs", 512))

        # Sentence model
        self.sentence_model_name = getattr(editor_cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.sentence_model = SentenceTransformer(self.sentence_model_name).to(self.device).eval()

        # Prompt prefix
        self.prefix = getattr(editor_cfg, "cot_prefix", "")

        # Inner layer hook setup
        inner_params = getattr(getattr(config, "model", config), "inner_params", None) or []
        if not inner_params:
            raise ValueError("IKE_CLIP requires config.model.inner_params to contain at least one layer name.")
        raw_name = inner_params[0]
        self.inner_param_name = raw_name.rsplit(".", 1)[0] if any(raw_name.endswith(s) for s in [".weight", ".bias"]) else raw_name
        edit_module = parent_module(self.model, brackets_to_periods(self.inner_param_name))
        self.target_layer = getattr(edit_module, self.inner_param_name.rsplit(".", 1)[-1])
        self._last_activations: Optional[torch.Tensor] = None
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

        # CLIP projection heads (lazy initialization)
        self.image_proj: Optional[nn.Linear] = None
        self.text_proj: Optional[nn.Linear] = None

        # Training data storage
        self.all_pairs: List[Dict[str, Any]] = []

        # Retrieval index
        self.rationale_texts: List[str] = []
        self.rationale_embeddings: Optional[torch.Tensor] = None
        self.counterfactual_embeddings: Optional[torch.Tensor] = None

        # Router configuration
        self.use_router = bool(getattr(editor_cfg, "use_router", True))
        self.router: Optional[nn.Sequential] = None
        self.router_lr = float(getattr(editor_cfg, "router_lr", 1e-3))
        self.router_epochs = int(getattr(editor_cfg, "router_epochs", 20))
        self.router_data: List[Tuple[float, float, int]] = []

        # Logging
        self.last_retrieval_log: Optional[List[Dict[str, Any]]] = None

    # -------------------------------------------------------------------------
    # Pass-through model interfaces
    # -------------------------------------------------------------------------
    def generate(self, *args, **kwargs):
        """Delegate to underlying model.generate (no automatic injection)."""
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        if self.wrapper is not None and hasattr(self.wrapper, "generate"):
            return self.wrapper.generate(*args, **kwargs)
        raise NotImplementedError("Model does not have generate method")

    def forward(self, *inputs, **kwargs):
        """Pass-through forward; IKE_CLIP does not alter model internals."""
        return self.model(*inputs, **kwargs)

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------
    def _forward_hook(self, module, inputs, output):
        """Capture activations at the configured inner layer."""
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            self._last_activations = x.detach()
        else:
            self._last_activations = None

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Very simple sentence splitter for rationale / COT text."""
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_pairs_from_dataset(self, dataset) -> List[Dict[str, Any]]:
        """Create (image, question, rationale_sentence) pairs (incl. counterfactuals)."""
        data = getattr(dataset, "data", None)
        if data is None:
            return []

        pairs: List[Dict[str, Any]] = []
        for ex in data:
            rationale = ex.get("cot") or ex.get("rationale") or ""
            question = ex.get("question", "")
            image = ex.get("image", None)

            if not rationale or image is None or not question:
                continue

            base_sents = self._split_sentences(str(rationale))
            # Original rationale sentences paired with real image
            for s in base_sents:
                pairs.append(
                    {
                        "image": image,
                        "question": question,
                        "rationale": s,
                        "is_counterfactual": False,
                    }
                )

            # Counterfactual sentences paired with a black image
            blank = PILImage.new("RGB", (364, 364), color="black")
            if (
                self.wrapper is not None
                and base_sents
                and self.num_counterfacts_per_sentence > 0
            ):
                for s in base_sents:
                    for _ in range(self.num_counterfacts_per_sentence):
                        inst = (
                            "Rewrite the sentence to state a different plausible fact "
                            "about the same object, using common knowledge. \n\n"
                            f"Original sentence: {s}\n\n"
                            "Rewritten sentence:"
                        )
                        try:
                            cf = self.wrapper.generate(
                                [blank], [inst], max_new_tokens=64, temperature=0.0
                            )[0]
                            cf = str(cf).strip()
                        except Exception:
                            cf = ""

                        if cf and cf.lower() != s.lower():
                            pairs.append(
                                {
                                    "image": blank,
                                    "question": question,
                                    "rationale": cf,
                                    "is_counterfactual": True,
                                }
                            )

            if len(pairs) >= self.max_pairs:
                break

        return pairs[: self.max_pairs]

    def _encode_vlm_features(self, images: List[Any], questions: List[str]) -> torch.Tensor:
        """Encode <image, question> pairs using the chosen inner layer (avg over tokens)."""
        if self.wrapper is None or not hasattr(self.wrapper, "encode"):
            raise RuntimeError("IKE_CLIP requires a VQAModel wrapper with an `.encode` method.")

        self.model.eval()
        self._last_activations = None

        # For image-only mode, ignore question content when building VLM features.
        if self.image_only:
            prompts = ["" for _ in images]
        else:
            prompts = [str(q) for q in questions]
        imgs = [im if isinstance(im, PILImage.Image) else im for im in images]

        inputs = self.wrapper.encode(imgs, prompts, tokenize=False)
        with torch.no_grad():
            _ = self.model(**inputs)

        acts = self._last_activations
        if acts is None:
            raise RuntimeError(
                "Forward hook did not capture activations for the configured inner layer."
            )

        if acts.dim() == 2:
            acts = acts.unsqueeze(0)

        # Average over sequence dimension → "average embedding key"
        feats = acts.mean(dim=1)  # [B, H]
        return feats.to(self.device, dtype=torch.float32)

    def _ensure_heads(self, img_dim: int, txt_dim: int) -> None:
        """Initialize CLIP projection heads if missing."""
        if self.image_proj is None:
            self.image_proj = nn.Linear(img_dim, self.clip_dim, bias=True).to(self.device)
        if self.text_proj is None:
            self.text_proj = nn.Linear(txt_dim, self.clip_dim, bias=True).to(self.device)

    def _train_clip(self, pairs: List[Dict[str, Any]]) -> None:
        """Train CLIP-style projection heads on (image, question, rationale) pairs."""
        if not pairs:
            return

        optimizer: Optional[torch.optim.Optimizer] = None
        n = len(pairs)

        # Deterministic shuffling per run given self.seed (and fixed n).
        # Use a CPU generator because torch.randperm expects a CPU generator.
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)

        for epoch in range(self.num_epochs):
            perm = torch.randperm(n, generator=g)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = [pairs[i.item()] for i in idx]

                images = [b["image"] for b in batch]
                questions = [b["question"] for b in batch]
                texts = [b["rationale"] for b in batch]

                img_feats = self._encode_vlm_features(images, questions)  # [B, H_v]
                # SentenceTransformer may return inference-mode tensors; clone to allow grad.
                txt_feats_base = self.sentence_model.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                ).to(self.device, dtype=torch.float32).clone()  # [B, H_t]

                self._ensure_heads(img_feats.shape[-1], txt_feats_base.shape[-1])

                img_emb = self.image_proj(img_feats)
                txt_emb = self.text_proj(txt_feats_base)

                img_emb = F.normalize(img_emb, dim=-1)
                txt_emb = F.normalize(txt_emb, dim=-1)

                logits = img_emb @ txt_emb.t() / self.temperature  # [B, B]
                targets = torch.arange(logits.size(0), device=self.device)

                loss_i2t = F.cross_entropy(logits, targets)
                loss_t2i = F.cross_entropy(logits.t(), targets)
                loss = (loss_i2t + loss_t2i) / 2.0

                if optimizer is None:
                    params = list(self.image_proj.parameters()) + list(self.text_proj.parameters())
                    optimizer = torch.optim.Adam(params, lr=self.lr)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _build_rationale_index(self, pairs: List[Dict[str, Any]]) -> None:
        """Embed all rationale sentences and store them as a retrieval index."""
        if not pairs or self.text_proj is None:
            return

        if self.include_counterfactuals_in_index:
            texts = [p["rationale"] for p in pairs]
        else:
            texts = [
                p["rationale"]
                for p in pairs
                if not p.get("is_counterfactual", False)
            ]
        if not texts:
            return
        txt_feats_base = self.sentence_model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device, dtype=torch.float32).clone()

        txt_emb = self.text_proj(txt_feats_base)
        txt_emb = F.normalize(txt_emb, dim=-1)

        self.rationale_texts = texts
        self.rationale_embeddings = txt_emb

        # Build counterfactual embeddings for router entropy calculation
        cf_pairs = [p for p in pairs if p.get("is_counterfactual", False)]
        if cf_pairs:
            cf_texts = [p["rationale"] for p in cf_pairs]
            cf_feats = self.sentence_model.encode(
                cf_texts, convert_to_tensor=True, show_progress_bar=False
            ).to(self.device, dtype=torch.float32).clone()
            self.counterfactual_embeddings = F.normalize(self.text_proj(cf_feats), dim=-1)
        else:
            self.counterfactual_embeddings = None

    @torch.no_grad()
    def _compute_entropy(self, embeddings: Optional[torch.Tensor], img_emb: torch.Tensor) -> Optional[float]:
        """Compute entropy of similarity distribution over embeddings."""
        if embeddings is None or embeddings.size(0) == 0:
            return None
        sims = torch.matmul(embeddings, img_emb.t()).squeeze(-1)
        probs = F.softmax(sims / self.temperature, dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        return float(-(probs * log_probs).sum().item())

    @torch.no_grad()
    def _retrieve_facts(
        self, image: Any, question: str, top_k: int
    ) -> Tuple[List[str], Optional[float], Optional[float]]:
        """Retrieve top-k rationale sentences and compute entropies.

        Returns:
            (facts, entropy_all, entropy_counterfactual)
        """
        if self.rationale_embeddings is None or not self.rationale_texts:
            return [], None, None

        img_feats = self._encode_vlm_features([image], [question])
        self._ensure_heads(img_feats.shape[-1], self.rationale_embeddings.shape[-1])

        img_emb = self.image_proj(img_feats)
        img_emb = F.normalize(img_emb, dim=-1)

        entropy_all = self._compute_entropy(self.rationale_embeddings, img_emb)
        entropy_cf = self._compute_entropy(self.counterfactual_embeddings, img_emb) if self.counterfactual_embeddings is not None else None

        sims = torch.matmul(self.rationale_embeddings, img_emb.t()).squeeze(-1)
        k = min(top_k, sims.size(0))
        if k <= 0:
            return [], entropy_all, entropy_cf

        topk = torch.topk(sims, k=k, largest=True)
        facts = [self.rationale_texts[i] for i in topk.indices.tolist()]
        return facts, entropy_all, entropy_cf

    def _collect_router_data(self, dataset) -> List[Tuple[float, float, int]]:
        """Collect router training data: (entropy_all, entropy_cf, label).
        
        Returns training pairs:
        - Label 1: (entropy_all, entropy_cf) for edit examples
        - Label 0: (entropy_cf, entropy_all) as counterfactual negatives
        """
        router_data = []
        data = getattr(dataset, "data", None)
        if not data:
            return router_data

        for ex in data:
            question = ex.get("question", "")
            image = ex.get("image", None)
            if not question or image is None:
                continue

            _, entropy_all, entropy_cf = self._retrieve_facts(image, question, self.k)
            if entropy_all is not None and entropy_cf is not None:
                router_data.append((entropy_all, entropy_cf, 1))  # Edit example
                router_data.append((entropy_cf, entropy_all, 0))  # Counterfactual negative

        return router_data

    def _train_router(self) -> None:
        """Train logistic regression router on entropy features."""
        if len(self.router_data) < 2:
            return

        # Initialize router if needed (simple logistic regression)
        if self.router is None:
            self.router = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()).to(self.device)

        # Prepare training data
        X = torch.tensor([[e0, e1] for e0, e1, _ in self.router_data], dtype=torch.float32, device=self.device)
        y = torch.tensor([[l] for _, _, l in self.router_data], dtype=torch.float32, device=self.device)

        # Train router
        optimizer = torch.optim.Adam(self.router.parameters(), lr=self.router_lr)
        for _ in range(self.router_epochs):
            pred = self.router(X)
            loss = F.binary_cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def apply_to_dataset(
        self, dataset, inplace: bool = True
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Augment each example in a dataset with retrieved rationale sentences."""
        if not inplace:
            raise NotImplementedError(
                "Non-inplace dataset augmentation is not supported for IKE_CLIP."
            )

        data = getattr(dataset, "data", None)
        if data is None:
            raise ValueError("Dataset must expose a .data attribute for IKE_CLIP usage.")

        log: List[Dict[str, Any]] = []
        for ex in data:
            prompt = ex.get("prompt", "")
            question = ex.get("question", "")
            image = ex.get("image", None)

            if not prompt or not question or image is None:
                continue

            if "prompt_orig" not in ex:
                ex["prompt_orig"] = prompt

            facts, entropy_all, entropy_cf = self._retrieve_facts(image, question, self.k)
            if not facts:
                continue

            # Use router to decide whether to apply facts
            use_facts = True
            if self.use_router and self.router is not None and entropy_all is not None and entropy_cf is not None:
                with torch.no_grad():
                    router_input = torch.tensor([[entropy_all, entropy_cf]], dtype=torch.float32, device=self.device)
                    use_facts = self.router(router_input).item() > 0.5

            if use_facts:
                facts_str = " ".join(facts)
                ex["prompt"] = f"{self.prefix}{facts_str}\n\n{prompt}"

            # Store entropy values and router decision for logging
            ex["retrieval_entropy"] = entropy_all
            ex["retrieval_entropy_cf"] = entropy_cf
            ex["router_use"] = use_facts

            log.append(
                {
                    "uid": ex.get("uid"),
                    "retrieved": len(facts),
                    "entropy": entropy_all,
                    "entropy_cf": entropy_cf,
                    "router_use": use_facts,
                }
            )

        return log, dataset

    # ---------------------------------------------------------------------
    # revlm editor interface (API-compatible with other editors)
    # ---------------------------------------------------------------------
    def edit(
        self,
        config,
        tokens=None,
        batch_history=None,
        edit_ds=None,
        train_ds=None,
    ):
        """Entry point used by `run/edit.py` when editor_name == 'ike_clip'.

        CLIP is trained only on *edit* examples:
        - For each edit (error case) and its rationale sentences, we update the
          projection heads so that the <image, question> representation is close
          to its sentences.
        - Across multiple calls to `edit`, the same projection heads are further
          refined, effectively doing incremental training as more edits arrive.
        """
        # If there is no dataset to edit, do nothing.
        if edit_ds is None:
            return self.model

        # Build and accumulate edit pairs
        new_pairs = self._build_pairs_from_dataset(edit_ds)
        if new_pairs:
            self.all_pairs.extend(new_pairs)
            if len(self.all_pairs) > self.max_pairs:
                self.all_pairs = self.all_pairs[-self.max_pairs:]

        # Train CLIP and build retrieval index
        self._train_clip(self.all_pairs)
        self._build_rationale_index(self.all_pairs)

        # Collect router training data and train router (if enabled)
        if self.use_router:
            new_router_data = self._collect_router_data(edit_ds)
            if new_router_data:
                self.router_data.extend(new_router_data)
                max_router_data = self.max_pairs * 2  # Each example adds 2 entries
                if len(self.router_data) > max_router_data:
                    self.router_data = self.router_data[-max_router_data:]
            self._train_router()

        # Augment all prompts in-place on `edit_ds` and cache a retrieval log.
        self.last_retrieval_log, _ = self.apply_to_dataset(edit_ds, inplace=True)
        return self.model

