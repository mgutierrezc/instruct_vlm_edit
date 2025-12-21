import torch
from typing import Any, Dict, List, Tuple, Optional


class IKE_COT(torch.nn.Module):
    """In-context knowledge editor that uses each example's COT as the 'new fact'.

    Compared to `IKE`, this variant:
    - **Does not build or query a retrieval corpus**.
    - Uses `ex["cot"]` (or falls back to `ex["rationale"]`) directly as the
      "New Fact" text that is prepended to the original prompt.
    - Leaves model weights untouched; only modifies prompts in-place.
    """

    def __init__(self, config, model):
        super().__init__()
        self.config = config

        # Keep both wrapper (VQAModel) and inner HF model, like other editors
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = getattr(model, "tokenizer", None)
        self.device = getattr(config, "device", None)

        # Optional: how to label the injected COT section
        editor_cfg = getattr(config, "editor", config)
        self.prefix: str = getattr(
            editor_cfg,
            "cot_prefix",
            "New Facts: ",
        )

        # API compatibility with IKE_TUPLE (for ReasonEdit)
        self.k = 1
        self._cot_index = {}

        # For logging / inspection after editing
        self.last_retrieval_log: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------------------
    # Pass-through model interfaces
    # ---------------------------------------------------------------------
    def generate(self, *args, **kwargs):
        """Delegate to underlying model.generate (no automatic injection)."""
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        if self.wrapper is not None and hasattr(self.wrapper, "generate"):
            return self.wrapper.generate(*args, **kwargs)
        raise NotImplementedError("Model does not have generate method")

    def forward(self, *inputs, **kwargs):
        """Pass-through forward; IKE_COT does not alter model internals."""
        return self.model(*inputs, **kwargs)

    # API compat with IKE_TUPLE for ReasonEdit
    def _retrieve(self, image, question, k):
        """API compat with IKE_TUPLE: return COT for question if exists."""
        cot = self._cot_index.get(question)
        return [cot] if cot else []

    def _build_cot_index(self, dataset):
        """Build question -> COT lookup."""
        self._cot_index = {
            ex.get("question"): (ex.get("cot") or ex.get("rationale", "")).strip()
            for ex in getattr(dataset, "data", [])
            if ex.get("question") and (ex.get("cot") or ex.get("rationale"))
        }

    # ---------------------------------------------------------------------
    # Core COT-based prompt augmentation
    # ---------------------------------------------------------------------
    def _augment_example(self, ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Augment a single example in-place, returning a minimal log dict.

        Expected fields on `ex`:
        - `prompt`: original prompt.
        - `cot` (preferred) or `rationale`: COT text to prepend as 'New Fact'.
        """
        # Always augment from the original prompt if available to avoid stacking
        prompt = ex.get("prompt_orig") or ex.get("prompt", "")
        cot = ex.get("cot") or ex.get("rationale") or ""

        if not prompt or not cot:
            # Nothing to do if we lack either prompt or COT text
            return None

        cot_str = str(cot).strip()
        if not cot_str:
            return None

        # Preserve the original prompt once
        if "prompt_orig" not in ex:
            ex["prompt_orig"] = prompt

        # Insert COT near the end of the prompt (right before options) when possible.
        # This is more robust to left-truncation than prepending at the very beginning.
        insert_key = " Options:"
        if insert_key in prompt:
            idx = prompt.index(insert_key)
            augmented_prompt = (
                prompt[:idx]
                + f"\n\n{self.prefix}{cot_str}\n\n"
                + prompt[idx + 1 :]  # drop the leading space before 'Options:'
            )
        else:
            augmented_prompt = f"{self.prefix}{cot_str}\n\n{prompt}"
        ex["prompt"] = augmented_prompt

        log_entry = {
            "uid": ex.get("uid"),
            "cot_chars": len(cot_str),
            "cot_tokens_est": len(cot_str.split()),
        }
        return log_entry

    def apply_to_dataset(
        self, dataset, inplace: bool = True
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Augment each example in a dataset with its own COT as context."""
        if not inplace:
            raise NotImplementedError(
                "Non-inplace dataset augmentation is not supported for IKE_COT."
            )

        data = getattr(dataset, "data", None)
        if data is None:
            raise ValueError("Dataset must expose a .data attribute for IKE_COT usage.")

        log: List[Dict[str, Any]] = []
        for ex in data:
            entry = self._augment_example(ex)
            if entry is not None:
                log.append(entry)

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
        """Entry point used by `run/edit.py` when editor_name == 'ike_cot'.

        Usage:
            editor = get_editor(config, model)
            editor.generate = model.model.generate if hasattr(model, "model") else model.generate
            editor.edit(config, edit_ds=edit_ds)
        """
        # If there is no dataset to edit, do nothing.
        if edit_ds is None:
            return self.model

        # Build COT index for _retrieve() API compatibility with IKE_TUPLE
        self._build_cot_index(edit_ds)

        # No retrieval or external corpus: just use each example's own COT.
        self.last_retrieval_log, _ = self.apply_to_dataset(edit_ds, inplace=True)
        return self.model


