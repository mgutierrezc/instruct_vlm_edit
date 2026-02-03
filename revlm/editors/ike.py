import torch
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Any, Optional, Tuple


class IKE(torch.nn.Module):
    """Simple in-context knowledge editor (IKE) for `revlm`.

    - Build a text corpus from (prompt, gold label) pairs in a training split.
    - For each edit example, retrieve top‑k similar corpus entries using only
      its prompt as the query.
    - Prepend retrieved \"New Fact\" strings to the original prompt.
    - Model weights are never changed; only prompts are augmented.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        # Keep both wrapper (VQAModel) and inner HF model, like other editors
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device

        editor_cfg = getattr(config, "editor", config)
        self.k: int = int(getattr(editor_cfg, "k", 3))
        self.sentence_model_name: str = getattr(
            editor_cfg,
            "sentence_model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.sentence_model = SentenceTransformer(self.sentence_model_name).to(
            self.device
        )

        self.corpus_sentences: Optional[List[str]] = None
        self.corpus_embeddings: Optional[torch.Tensor] = None


    def generate(self, *args, **kwargs):
        """Delegate to underlying model.generate (no automatic ICL injection)."""
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        elif self.wrapper is not None and hasattr(self.wrapper, "generate"):
            return self.wrapper.generate(*args, **kwargs)
        raise NotImplementedError("Model does not have generate method")

    def forward(self, *inputs, **kwargs):
        """Pass-through forward; IKE does not alter model internals."""
        return self.model(*inputs, **kwargs)

    # -------------------------------------------------------------------------
    # IKE core utilities
    # -------------------------------------------------------------------------
    def _normalize_dataset_entries(
        self, dataset_or_entries: Any
    ) -> List[Dict[str, str]]:
        """Convert a dataset or list of examples to simple (prompt, target) dicts."""
        if dataset_or_entries is None:
            return []

        data = (
            dataset_or_entries
            if isinstance(dataset_or_entries, list)
            else getattr(dataset_or_entries, "data", dataset_or_entries)
        )

        entries: List[Dict[str, str]] = []
        for ex in data:
            entries.append(
                {
                    "prompt": ex["prompt"],
                    "target": ex["gold"]["label"],
                }
            )

        return entries

    def _ensure_corpus(self, dataset_or_entries: Any = None) -> None:
        """Ensure corpus embeddings exist, optionally building them from a dataset."""
        if self.corpus_embeddings is not None and self.corpus_sentences is not None:
            return

        normalized = self._normalize_dataset_entries(dataset_or_entries)
        if not normalized:
            raise ValueError(
                "IKE requires a training dataset/list of entries to build the corpus."
            )
        self.build_corpus_from_dataset(normalized)

    @torch.no_grad()
    def build_corpus_from_dataset(self, train_ds) -> None:
        """Build the retrieval corpus from (prompt, target) pairs."""
        normalized_entries = self._normalize_dataset_entries(train_ds)

        sentences: List[str] = []

        for ex in normalized_entries:
            prompt = ex["prompt"]
            target = ex["target"]
            new_fact = f"{prompt} {target}"

            sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")

        if not sentences:
            raise ValueError(
                "IKE.build_corpus_from_dataset: no valid (prompt, target) pairs in train_ds."
            )

        embeddings = self.sentence_model.encode(
            sentences, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)
        embeddings = util.normalize_embeddings(embeddings)

        self.corpus_sentences = sentences
        self.corpus_embeddings = embeddings
        print(f"[IKE] corpus built: {len(sentences)} entries", flush=True)

    @torch.no_grad()
    def retrieve_icl_examples(self, prompt: str, target: str = None) -> List[str]:
        """Retrieve up to k ICL examples for a new query prompt.

        - Query embedding is built from the prompt only (no gold label).
        - Any retrieved corpus entry containing the same prompt is filtered out.
        """
        if self.corpus_embeddings is None or self.corpus_sentences is None:
            raise RuntimeError(
                "IKE.retrieve_icl_examples called before corpus was built. "
                "Call build_corpus_from_dataset(train_ds) first."
            )

        query_sentence = f"Prompt: {prompt}\n\n"

        query_embedding = self.sentence_model.encode(
            query_sentence, convert_to_tensor=True, show_progress_bar=False
        ).unsqueeze(0)
        query_embedding = util.normalize_embeddings(query_embedding.to(self.device))

        search_k = min(len(self.corpus_sentences), max(self.k * 4, self.k))
        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            score_function=util.dot_score,
            top_k=search_k,
        )
        assert len(hits) == 1
        hit = hits[0]

        filtered_hits: List[Dict[str, Any]] = []
        for h in hit:
            s = self.corpus_sentences[h["corpus_id"]]
            # Filter exact prompt match to avoid self-retrieval
            if prompt and prompt in s:
                continue
            filtered_hits.append(h)
            if len(filtered_hits) >= self.k:
                break

        final_hits = filtered_hits if filtered_hits else hit[: self.k]

        icl_examples = [self.corpus_sentences[h["corpus_id"]] for h in final_hits]
        return icl_examples

    def augment_prompt(
        self, prompt: str, target: str = None, train_ds: Any = None
    ) -> Tuple[str, List[str]]:
        """Return prompt prefixed with retrieved demonstrations."""
        self._ensure_corpus(train_ds)
        icl_examples = self.retrieve_icl_examples(prompt, target)
        augmented_prompt = "".join(icl_examples) + prompt
        return augmented_prompt, icl_examples

    def apply_to_dataset(
        self, dataset, train_ds: Any = None, inplace: bool = True
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Augment each example in a dataset with IKE demonstrations (in-place)."""
        if not inplace:
            raise NotImplementedError("Non-inplace dataset augmentation is not supported.")

        self._ensure_corpus(train_ds)
        data = getattr(dataset, "data", None)
        if data is None:
            raise ValueError("Dataset must expose a .data attribute for IKE usage.")

        retrieval_log: List[Dict[str, Any]] = []
        for ex in data:
            prompt = ex.get("prompt", "")
            if not prompt:
                continue

            augmented_prompt, icl_examples = self.augment_prompt(prompt)
            ex.setdefault("prompt_orig", prompt)
            ex["prompt"] = augmented_prompt
            ex["icl_examples"] = icl_examples
            retrieval_log.append(
                {"uid": ex.get("uid"), "icl_count": len(icl_examples)}
            )

        # Log summary and examples
        print(f"[IKE] applied to {len(retrieval_log)}/{len(data)} examples (k={self.k})", flush=True)
        # for i, ex in enumerate(data[:3]):
        #     print(f"  [{i}] q='{ex.get('question', '')[:50]}' target='{ex.get('gold', {}).get('label', '')}' icl_count={len(ex.get('icl_examples', []))}")
        #     if ex.get('icl_examples'):
        #         print(f"       icl[0]: {ex['icl_examples'][0][:100]}...")

        return retrieval_log, dataset

    # -------------------------------------------------------------------------
    # revlm editor interface (API-compatible with other editors)
    # -------------------------------------------------------------------------
    def edit(
        self,
        config,
        tokens=None,
        batch_history=None,
        edit_ds=None,
        train_ds=None,
    ):
        """
        editor.edit(config, edit_ds=edit_ds)
        """
        # If there is no dataset to edit, do nothing.
        if edit_ds is None:
            return self.model

        # Corpus of "new facts" is always built from the edit dataset:
        # - In the common case, callers can pass only `edit_ds` (train_ds=None).
        # - If `train_ds` is provided, we treat it as the corpus source explicitly.
        corpus_source = train_ds if train_ds is not None else edit_ds

        # Always rebuild corpus from corpus_source to capture all edits
        self.build_corpus_from_dataset(corpus_source)

        # Augment all prompts in-place on `edit_ds` and cache a retrieval log.
        self.last_retrieval_log, _ = self.apply_to_dataset(edit_ds, corpus_source)
        return self.model



# """
# usage:
# """
# from revlm.editors import get_editor
# from revlm.metrics.editeval import reliability

# # 1) Run baseline predictions to populate gold/pred fields
# ds.set_dataloader(
#     with_rationale=config.rationale,
#     rationale_in_prompt=False,
#     shuffle_choices=False,
#     unpaired=True,
# )
# ds.task_generate(model, use_cache=False)

# # 2) Build the edit set (examples where model is wrong)
# edit_ds = ds.get_edits()

# # 3) Initialize IKE editor
# editor = get_editor(config, model)
# editor.generate = model.model.generate if hasattr(model, "model") else model.generate

# # 4) Run IKE: build corpus + augment prompts on edit_ds
# #    (internally uses full ds minus edit_ds as retrieval pool)
# editor.edit(config, edit_ds=edit_ds)

# # 5) Evaluate the edited (IKE-augmented) model on the edit set
# ike_reliability = reliability(model, edit_ds)
# print(f"Reliability after IKE: {ike_reliability:.4f}")
