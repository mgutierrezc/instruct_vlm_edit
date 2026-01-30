"""LiveEdit_COT: LiveEdit with rationale sentence keys.

Similar to GRACE_COT, we train additional LoRA experts for each 
factual sentence in the rationale, not just the question→answer pair.

This populates the expert pool with:
  - Expert for question → answer
  - Expert for sentence1 → sentence1
  - Expert for sentence2 → sentence2
  - ...

During inference, if input resembles any rationale sentence,
the relevant LoRA expert gets retrieved and applied.
"""

import re
from typing import Dict, Optional
import torch
from PIL import Image

from .liveedit import LiveEdit


class LiveEdit_COT(LiveEdit):
    """LiveEdit extended with sentence-level experts from rationale."""
    
    def __init__(self, config, model):
        super().__init__(config, model)
        self.wrapper = model
    
    def edit(self, config, tokens, batch_history, 
             image=None, cot=None, **kwargs):
        """Edit with question key + sentence keys.
        
        Args:
            config: Editor config
            tokens: Tokenized question→answer batch
            batch_history: Previous batches
            image: PIL Image (required for sentence edits)
            cot: Chain-of-thought rationale string
        """
        # 1. Original edit: question → answer
        super().edit(config=config, tokens=tokens, batch_history=batch_history, **kwargs)
        
        # 2. If no COT provided, we're done
        if not cot or not image:
            return
        
        # 3. Each sentence = new expert
        sentences = self._split_sentences(cot)
        for sent in sentences:
            if len(sent.split()) < 3:  # Skip very short sentences
                continue
            
            # Create training batch: sentence → sentence (autoencoding)
            sent_tokens = self.wrapper.prepare_training_batch({
                "images": [image] if isinstance(image, Image.Image) else [Image.open(image).convert("RGB")],
                "prompts": [sent],
                "golds": [{"label": sent, "label_train": sent}],
                "idxs": [0],
            })
            
            # Train a new LoRA expert for this sentence
            super().edit(config=config, tokens=sent_tokens, batch_history=batch_history, **kwargs)
            print(f"[liveedit_cot] +sentence expert: '{sent[:50]}...'")
    
    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
