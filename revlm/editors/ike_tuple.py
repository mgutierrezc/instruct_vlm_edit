import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .utils import brackets_to_periods, parent_module

class IKE_TUPLE(nn.Module):
    """One-direction tuple-contrastive retriever for `revlm`.

    High-level behavior:
    - Build a corpus of (image, question, rationale_sentence) positives from COT/rationales.
    - For each positive sentence s+, generate a counterfactual sentence (text-only) by prompting
      the VLM to rewrite it into a different plausible fact (conditioned on the same image).
    - Optionally augment questions by paraphrasing them (same meaning) via the VLM.
    - Train a dual-encoder retriever with a tuple loss on (image, question, s+, s-):
        score(image,q,s+) > score(image,q,s-)
      using ONLY the <image,question> → <sentence> direction (no bidirectional CLIP loss).
    - Index projected sentence embeddings; at inference retrieve top-k by cosine similarity
      and prepend as "New Facts" to each prompt.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        editor_cfg = getattr(config, "editor", config)
        # Model setup
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        # Retriever configuration (reuse ike_clip-style keys for simplicity)
        self.seed = int(getattr(config, "seed", 333))
        self.image_only = bool(getattr(editor_cfg, "image_only", False))
        self.k = int(getattr(editor_cfg, "k", 3))
        self.clip_dim = int(getattr(editor_cfg, "clip_dim", 512))
        self.num_epochs = int(getattr(editor_cfg, "clip_epochs", 500))
        self.batch_size = int(getattr(editor_cfg, "clip_batch_size", 10))
        self.lr = float(getattr(editor_cfg, "clip_lr", 1e-3))
        self.temperature = float(getattr(editor_cfg, "clip_temperature", 0.5))
        # Optional early-stopping based on retrieval accuracy (if <= 0 → disabled).
        self.retrieval_acc_threshold = float(getattr(editor_cfg, "retrieval_acc_threshold", 0.8))
        self.max_pairs = int(getattr(editor_cfg, "max_pairs", 512))
        # Alias to match ike_clip config naming while keeping backward compat.
        self.num_counterfacts_per_sentence = int(
            getattr(
                editor_cfg,
                "num_counterfacts_per_sentence",
                getattr(editor_cfg, "num_negatives_per_sentence", 1),
            )
        )
        self.neg_max_new_tokens = int(getattr(editor_cfg, "neg_max_new_tokens", 64))
        self.neg_temperature = float(getattr(editor_cfg, "neg_temperature", 0.0))
        # Question paraphrase augmentation (0 disables)
        self.num_question_aug = int(getattr(editor_cfg, "num_question_aug", 0))
        self.question_aug_max_new_tokens = int(getattr(editor_cfg, "question_aug_max_new_tokens", 32))
        self.question_aug_temperature = float(getattr(editor_cfg, "question_aug_temperature", 0.0))
        # Optional entropy-based router (same spirit as ike_clip)
        self.use_router = bool(getattr(editor_cfg, "use_router", True))
        self.router_hidden = int(getattr(editor_cfg, "router_hidden", 10))
        self.router: Optional[nn.Sequential] = None
        self.router_lr = float(getattr(editor_cfg, "router_lr", 1e-3))
        self.router_epochs = int(getattr(editor_cfg, "router_epochs", 1000))
        self.router_data: List[Tuple[float, float, int]] = []
        # Sentence model
        self.sentence_model_name = getattr(
            editor_cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.sentence_model = SentenceTransformer(self.sentence_model_name).to(self.device).eval()
        self.txt_dim = int(getattr(self.sentence_model, "get_sentence_embedding_dimension")())
        # Prompt prefix
        self.prefix = getattr(editor_cfg, "cot_prefix", "New Fact: ")
        # Inner layer hook setup (same pattern as other editors)
        inner_params = getattr(getattr(config, "model", config), "inner_params", None) or []
        if not inner_params:
            raise ValueError(
                "IKE_TUPLE requires config.model.inner_params to contain at least one layer name."
            )
        raw_name = inner_params[0]
        self.inner_param_name = (
            raw_name.rsplit(".", 1)[0]
            if any(raw_name.endswith(s) for s in [".weight", ".bias"])
            else raw_name
        )
        edit_module = parent_module(self.model, brackets_to_periods(self.inner_param_name))
        self.target_layer = getattr(edit_module, self.inner_param_name.rsplit(".", 1)[-1])
        self._last_activations: Optional[torch.Tensor] = None
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        # Projection heads (lazy init)
        self.image_proj: Optional[nn.Module] = None
        self.text_proj: Optional[nn.Module] = None
        # Training data storage
        self.all_pairs: List[Dict[str, Any]] = []
        # Retrieval index
        self.rationale_texts: List[str] = []
        self.rationale_embeddings: Optional[torch.Tensor] = None
        # Counterfactual (negative) embeddings for router entropy calculation.
        self.counterfactual_embeddings: Optional[torch.Tensor] = None
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
        """Pass-through forward; IKE_TUPLE does not alter model internals."""
        return self.model(*inputs, **kwargs)

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------
    def _forward_hook(self, module, inputs, output):
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            self._last_activations = x.detach()
        else:
            self._last_activations = None

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

    @staticmethod
    def _clean_generated_sentence(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        # Keep first line only; many wrappers return multi-line generations.
        t = t.splitlines()[0].strip()
        # Remove common leading labels.
        t = re.sub(r"^(rewritten sentence|rewrite|answer)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
        return t

    @staticmethod
    def _clean_generated_question(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        t = t.splitlines()[0].strip()
        t = re.sub(r"^(paraphrase|rewritten question|rewrite|answer)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
        # Drop trailing quotes/punctuation artifacts.
        t = t.strip(" \"'")
        return t

    def _generate_text_negative(self, image: Any, question: str, sentence: str) -> str:
        """Generate a text-only counterfactual sentence by prompting the VLM.

        This does NOT use black images; generation is conditioned on the real image.
        """
        s = (sentence or "").strip()
        if not s:
            return ""
        if self.wrapper is None or not hasattr(self.wrapper, "generate"):
            return ""

        # Keep prompt format close to ike_clip, but condition on the real image (no blank).
        inst = ("Rewrite the sentence to state a different plausible fact about the same object, using common knowledge. "
                "Keep it to one sentence.\n\n"
                f"Question: {question}\n\nOriginal sentence: {s}\n\nRewritten sentence:")
        try:
            out = self.wrapper.generate(
                [image],
                [inst],
                max_new_tokens=self.neg_max_new_tokens,
                temperature=self.neg_temperature,
            )[0]
        except Exception:
            return ""

        neg = self._clean_generated_sentence(str(out))
        if not neg or neg.lower() == s.lower():
            return ""
        return neg

    def _paraphrase_question(self, image: Any, question: str) -> str:
        """Paraphrase question while keeping meaning (VLM-prompted)."""
        q = (question or "").strip()
        if not q:
            return ""
        if self.wrapper is None or not hasattr(self.wrapper, "generate"):
            return ""

        inst = ("Paraphrase the question while keeping the exact same meaning. Return only the rewritten question.\n\n"
                f"Original question: {q}\n\nRewritten question:")
        try:
            out = self.wrapper.generate(
                [image],
                [inst],
                max_new_tokens=self.question_aug_max_new_tokens,
                temperature=self.question_aug_temperature,
            )[0]
        except Exception:
            return ""

        q2 = self._clean_generated_question(str(out))
        if not q2 or q2.lower() == q.lower():
            return ""
        return q2

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        # SentenceTransformer may return inference-mode tensors; clone to allow grad.
        return (
            self.sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            .to(self.device, dtype=torch.float32)
            .clone()
        )

    def _build_pairs_from_dataset(self, dataset) -> List[Dict[str, Any]]:
        data = getattr(dataset, "data", None)
        if not data:
            return []

        pairs: List[Dict[str, Any]] = []
        for ex in data:
            rationale = ex.get("cot") or ex.get("rationale") or ""
            question = ex.get("question", "")
            image = ex.get("image", None)

            if not rationale or image is None or not question:
                continue

            # Build question variants (same meaning) once per example.
            questions = [str(question)]
            if self.num_question_aug > 0:
                for _ in range(self.num_question_aug):
                    q2 = self._paraphrase_question(image, str(question))
                    if q2:
                        questions.append(q2)
                # cheap dedup, preserve order
                questions = list(dict.fromkeys(questions))

            base_sents = self._split_sentences(str(rationale))
            for s_pos in base_sents:
                for _ in range(max(1, self.num_counterfacts_per_sentence)):
                    # Build base (s+, s-) once, then reuse across paraphrased questions.
                    s_neg = self._generate_text_negative(image, str(question), s_pos)
                    if not s_neg:
                        continue
                    for qv in questions:
                        pairs.append({"image": image, "question": qv, "pos": s_pos, "neg": s_neg})

                if len(pairs) >= self.max_pairs:
                    break
            if len(pairs) >= self.max_pairs:
                break

        return pairs[: self.max_pairs]

    def _encode_vlm_features(self, images: List[Any], questions: List[str]) -> torch.Tensor:
        """Encode <image, question> using a chosen inner layer (avg over tokens)."""
        if self.wrapper is None or not hasattr(self.wrapper, "encode"):
            raise RuntimeError(
                "IKE_TUPLE requires a VQAModel wrapper with an `.encode` method."
            )

        self.model.eval()
        self._last_activations = None

        prompts = ["" for _ in images] if self.image_only else [str(q) for q in questions]
        inputs = self.wrapper.encode(images, prompts, tokenize=False)
        with torch.no_grad():
            _ = self.model(**inputs)

        acts = self._last_activations
        if acts is None:
            raise RuntimeError(
                "Forward hook did not capture activations for the configured inner layer."
            )
        if acts.dim() == 2:
            acts = acts.unsqueeze(0)

        feats = acts.mean(dim=1)  # [B, H]
        return feats.to(self.device, dtype=torch.float32)

    def _ensure_heads(self, img_dim: int, txt_dim: int) -> None:
        if self.image_proj is None:
            self.image_proj = nn.Sequential(
                nn.Linear(img_dim, self.clip_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.clip_dim, self.clip_dim, bias=True),
                nn.LayerNorm(self.clip_dim),
            ).to(self.device)
        if self.text_proj is None:
            self.text_proj = nn.Sequential(
                nn.Linear(txt_dim, self.clip_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.clip_dim, self.clip_dim, bias=True),
                nn.LayerNorm(self.clip_dim),
            ).to(self.device)

    @torch.no_grad()
    def _compute_entropy(
        self, embeddings: Optional[torch.Tensor], img_emb: torch.Tensor
    ) -> Optional[float]:
        """Entropy of the similarity distribution over `embeddings` for this query."""
        if embeddings is None or embeddings.size(0) == 0:
            return None
        sims = torch.matmul(embeddings, img_emb.t()).squeeze(-1)
        probs = F.softmax(sims / self.temperature, dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        return float(-(probs * log_probs).sum().item())

    @torch.no_grad()
    def _encode_query(self, image: Any, question: str) -> torch.Tensor:
        img_feats = self._encode_vlm_features([image], [question])
        self._ensure_heads(int(img_feats.shape[-1]), self.txt_dim)
        return F.normalize(self.image_proj(img_feats), dim=-1)

    @torch.no_grad()
    def _compute_retrieval_accuracy(self, dataset) -> Optional[float]:
        """Compute proportion of retrieved facts that are present in example rationales.

        For each example with (image, question, rationale/cot), we:
          - Split the rationale into sentences (ground‑truth sentences).
          - Retrieve top‑k facts from the current index (k = self.k).
          - Count how many retrieved facts exactly match any rationale sentence.

        Returns:
            A float in [0, 1] or None if accuracy cannot be computed.
        """
        data = getattr(dataset, "data", None)
        if not data or self.k <= 0:
            return None

        total_hits = 0
        total_retrieved = 0

        for ex in data:
            rationale = ex.get("cot") or ex.get("rationale") or ""
            question = ex.get("question", "")
            image = ex.get("image", None)

            if not rationale or image is None or not question:
                continue

            base_sents = self._split_sentences(str(rationale))
            if not base_sents:
                continue

            facts, _, _ = self._retrieve_facts(image, question, self.k)
            if not facts:
                continue

            base_set = set(base_sents)
            for f in facts:
                if f in base_set:
                    total_hits += 1
            total_retrieved += len(facts)

        if total_retrieved == 0:
            return None
        return float(total_hits) / float(total_retrieved)

    def _train_tuple(self, pairs: List[Dict[str, Any]], eval_ds=None) -> None:
        """Train the one-direction tuple contrastive retriever on (img,q,pos,neg).

        If `self.retrieval_acc_threshold` > 0 and `eval_ds` is provided, we compute a
        simple retrieval accuracy after each epoch (using top‑k = self.k) and stop
        early once the threshold is reached.
        """
        if not pairs:
            return

        optimizer: Optional[torch.optim.Optimizer] = None
        scheduler = None
        n = len(pairs)

        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)

        for epoch in range(self.num_epochs):
            perm = torch.randperm(n, generator=g)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = [pairs[i.item()] for i in idx]

                images = [b["image"] for b in batch]
                questions = [b["question"] for b in batch]
                pos_texts = [b["pos"] for b in batch]
                neg_texts = [b["neg"] for b in batch]

                img_feats = self._encode_vlm_features(images, questions)  # [B, H_v]
                pos_feats = self._encode_texts(pos_texts)
                neg_feats = self._encode_texts(neg_texts)
                self._ensure_heads(int(img_feats.shape[-1]), self.txt_dim)

                img_emb = F.normalize(self.image_proj(img_feats), dim=-1)
                pos_emb = F.normalize(self.text_proj(pos_feats), dim=-1)
                neg_emb = F.normalize(self.text_proj(neg_feats), dim=-1)

                score_pos = (img_emb * pos_emb).sum(dim=-1) / self.temperature
                score_neg = (img_emb * neg_emb).sum(dim=-1) / self.temperature

                logits = torch.stack([score_pos, score_neg], dim=1)  # [B, 2]
                targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, targets)

                if optimizer is None:
                    params = list(self.image_proj.parameters()) + list(self.text_proj.parameters())
                    optimizer = torch.optim.Adam(params, lr=self.lr)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, self.num_epochs)
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(1, num_batches)
            acc = None

            if eval_ds is not None:
                # Rebuild index with current text head so retrieval uses latest params.
                self._build_rationale_index(pairs)
                acc = self._compute_retrieval_accuracy(eval_ds)
                if (
                    self.retrieval_acc_threshold > 0
                    and acc is not None
                    and acc >= self.retrieval_acc_threshold
                ):
                    print(
                        f"[IKE_TUPLE] epoch {epoch+1}/{self.num_epochs} "
                        f"- loss: {avg_loss:.4f}, retr_acc@{self.k}: {acc:.3f} (early stop)"
                    )
                    break

            msg = f"[IKE_TUPLE] epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}"
            if acc is not None:
                msg += f", retr_acc@{self.k}: {acc:.3f}"
            print(msg)

        # Final index build with latest parameters (no-op if already done above).
        self._build_rationale_index(pairs)

    @torch.no_grad()
    def _build_rationale_index(self, pairs: List[Dict[str, Any]]) -> None:
        if not pairs or self.text_proj is None:
            return
        pos_texts = [p["pos"] for p in pairs if p.get("pos")]
        neg_texts = [p["neg"] for p in pairs if p.get("neg")]
        if not pos_texts:
            return

        pos_texts = list(dict.fromkeys(pos_texts))
        neg_texts = list(dict.fromkeys(neg_texts)) if neg_texts else []

        pos_emb = F.normalize(self.text_proj(self._encode_texts(pos_texts)), dim=-1)

        self.rationale_texts = pos_texts
        self.rationale_embeddings = pos_emb

        if neg_texts:
            self.counterfactual_embeddings = F.normalize(
                self.text_proj(self._encode_texts(neg_texts)), dim=-1
            )
        else:
            self.counterfactual_embeddings = None

    @torch.no_grad()
    def _retrieve_facts(
        self, image: Any, question: str, top_k: int
    ) -> Tuple[List[str], Optional[float], Optional[float]]:
        if self.rationale_embeddings is None or not self.rationale_texts:
            return [], None, None

        img_emb = self._encode_query(image, question)

        entropy_all = self._compute_entropy(self.rationale_embeddings, img_emb)
        entropy_cf = (
            self._compute_entropy(self.counterfactual_embeddings, img_emb)
            if self.counterfactual_embeddings is not None
            else None
        )

        sims = torch.matmul(self.rationale_embeddings, img_emb.t()).squeeze(-1)
        k = min(int(top_k), int(sims.size(0)))
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
        router_data: List[Tuple[float, float, int]] = []
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
                router_data.append((entropy_all, entropy_cf, 1))
                router_data.append((entropy_cf, entropy_all, 0))

        return router_data

    def _train_router(self) -> None:
        """Train entropy-feature router.

        If router_hidden == 0 -> logistic regression.
        If router_hidden  > 0 -> 1-hidden-layer MLP.
        """
        if len(self.router_data) < 2:
            return

        if self.router is None:
            if self.router_hidden > 0:
                self.router = nn.Sequential(
                    nn.Linear(2, self.router_hidden),
                    nn.ReLU(),
                    nn.Linear(self.router_hidden, 1),
                    nn.Sigmoid(),
                ).to(self.device)
            else:
                self.router = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()).to(self.device)

        X = torch.tensor(
            [[e0, e1] for e0, e1, _ in self.router_data],
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.tensor(
            [[l] for _, _, l in self.router_data],
            dtype=torch.float32,
            device=self.device,
        )

        optimizer = torch.optim.Adam(self.router.parameters(), lr=self.router_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.router_epochs))
        for epoch in range(self.router_epochs):
            pred = self.router(X)
            loss = F.binary_cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(
                f"[IKE_TUPLE][router] epoch {epoch+1}/{self.router_epochs} - loss: {float(loss.item()):.4f}"
            )

    def apply_to_dataset(self, dataset, inplace: bool = True) -> Tuple[List[Dict[str, Any]], Any]:
        if not inplace:
            raise NotImplementedError("Non-inplace dataset augmentation is not supported for IKE_TUPLE.")

        data = getattr(dataset, "data", None)
        if data is None:
            raise ValueError("Dataset must expose a .data attribute for IKE_TUPLE usage.")

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

            use_facts = True
            if self.use_router and self.router is not None and entropy_all is not None and entropy_cf is not None:
                with torch.no_grad():
                    router_input = torch.tensor([[entropy_all, entropy_cf]], dtype=torch.float32, device=self.device)
                    use_facts = self.router(router_input).item() > 0.5

            if use_facts:
                ex["prompt"] = f"{self.prefix}{' '.join(facts)}\n\n{prompt}"

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
        """Entry point used by `run/edit.py` when editor_name == 'ike_tuple'.

        For sequential mode: pass train_ds=prior examples, edit_ds=current example.
        Retriever is trained on train_ds (or edit_ds if train_ds is None), then edit_ds
        is augmented using that index.
        """
        if edit_ds is None:
            return self.model

        pair_source = train_ds if train_ds is not None else edit_ds
        new_pairs = self._build_pairs_from_dataset(pair_source)

        if train_ds is not None:
            self.all_pairs = new_pairs[: self.max_pairs]
        elif new_pairs:
            self.all_pairs.extend(new_pairs)
            if len(self.all_pairs) > self.max_pairs:
                self.all_pairs = self.all_pairs[-self.max_pairs :]

        if self.all_pairs:
            self._train_tuple(self.all_pairs, eval_ds=edit_ds)

            if self.use_router:
                new_router_data = self._collect_router_data(pair_source)
                if new_router_data:
                    if train_ds is not None:
                        self.router_data = new_router_data
                    else:
                        self.router_data.extend(new_router_data)
                    max_router_data = self.max_pairs * 2
                    if len(self.router_data) > max_router_data:
                        self.router_data = self.router_data[-max_router_data:]
                self._train_router()

        if self.rationale_embeddings is not None:
            self.last_retrieval_log, _ = self.apply_to_dataset(edit_ds, inplace=True)

        return self.model


