# Concrete Example: Autoregressive Finetuning Walkthrough

## Example Data
From AOKVQA dataset (task="mc", with_rationale=False):

**Raw Example:**
- Image: `data/images/aokvqa/train2017/000000299207.jpg`
- Question: "What is the man by the bags awaiting?"
- Answer (gold label): "cab"
- Choices: "skateboarder; train; delivery; cab"

**After TaskEngineer (`MCTaskEngineer`):**
- `batch['prompts'][0]`: 
  ```
  "Choose the correct answer from the options. What is the man by the bags awaiting? Options: cab; skateboarder; train; delivery"
  ```
- `batch['golds'][0]`: `{"label": "cab", "choices": {...}}`

---

## Step 1: Batch Creation
```python
batch = {
    "images": [PIL.Image(...)],  # 1 image
    "prompts": ["Choose the correct answer from the options. What is the man by the bags awaiting? Options: cab; skateboarder; train; delivery"],
    "golds": [{"label": "cab"}],
    "idxs": [0]
}
```

---

## Step 2: `prepare_training_batch()` Processing

### 2a. Encode Prompt + Image
```python
prompt_inputs = vlm.encode(images, prompts, tokenize=False)
# Returns:
prompt_inputs = {
    "input_ids": tensor([[1, 518, 4555, 297, 3419, ..., 297, 8906, 322]]),  # Shape: [1, 42]
    "attention_mask": tensor([[1, 1, 1, ..., 1, 1, 1]]),  # Shape: [1, 42]
    "pixel_values": tensor([...]),  # Image features, Shape: [1, 3, 336, 336]
    ...
}
# Example tokenization (simplified):
# [BOS] "Choose" "the" "correct" "answer" ... "delivery" [EOS]
# token_ids: [1, 518, 4555, 297, 3419, ..., 8906, 322]
# prompt_len = 42 tokens
```

### 2b. Tokenize Gold Answer
```python
gold_texts = ["cab"]
gold_tok = tokenizer(gold_texts, return_tensors="pt", add_special_tokens=False, padding=True)
labels_ids = gold_tok.input_ids  # Shape: [1, 1]
# labels_ids = [[1234]]  # "cab" tokenized to token_id 1234
```

### 2c. Concatenate for Decoder-Only Model (LLaVA/Qwen3)
```python
# Full sequence: [prompt_tokens] + [answer_tokens]
full_ids = torch.cat([input_ids, labels_ids], dim=1)
# Shape: [1, 43]  (42 prompt + 1 answer)
# full_ids = [[1, 518, 4555, ..., 8906, 322, 1234]]
#             └──────── prompt ────────┘ └answer┘

# Create labels mask
labels = torch.full_like(full_ids, -100)  # Start with all -100
labels[:, prompt_len:] = full_ids[:, prompt_len:]  # Copy answer tokens
# labels = [[-100, -100, -100, ..., -100, -100, 1234]]
#           └──────── prompt (masked) ────────┘ └answer┘
```

### 2d. Final Model Inputs
```python
model_inputs = {
    "input_ids": full_ids,           # [1, 43] - Full sequence: prompt + answer
    "attention_mask": full_attn,      # [1, 43] - All ones
    "pixel_values": pixel_values,    # [1, 3, 336, 336] - Image
    "labels": labels,                 # [1, 43] - -100 for prompt, token_ids for answer
}
```

---

## Step 3: Model Forward Pass (Autoregressive)

### 3a. Model Processing
```python
outputs = model(**model_inputs)
# Model internally does:

# Position 0: Predicts token at position 1 using [BOS]
# Position 1: Predicts token at position 2 using [BOS, "Choose"]
# Position 2: Predicts token at position 3 using [BOS, "Choose", "the"]
# ...
# Position 41: Predicts token at position 42 using [BOS, ..., "delivery"]
# Position 42: Predicts token at position 43 using [BOS, ..., "delivery", ???]
#              └─ This is where we want to predict "cab" (token 1234)
```

### 3b. Logits Output
```python
logits = outputs.logits  # Shape: [1, 43, vocab_size]
# logits[0, 42, :] = probability distribution over vocabulary
#                   for predicting next token after "delivery"
```

### 3c. Loss Computation
```python
# Cross-entropy loss with ignore_index=-100
loss = CrossEntropyLoss(logits, labels, ignore_index=-100)

# Only position 42 contributes to loss (where labels[0, 42] = 1234)
# Loss = -log(softmax(logits[0, 42, 1234]))

# Example:
# logits[0, 42, 1234] = 2.3  (logit for "cab")
# softmax(...) = 0.10  (10% probability)
# loss = -log(0.10) = 2.30
```

---

## Step 4: Editor Finetuning

### 4a. Editor Receives Tokens
```python
editor.edit(config, tokens=model_inputs, batch_history=[])
```

### 4b. Optimizer Setup (Only Selected Layer)
```python
# Only "language_model.lm_head.weight" is trainable
# All other parameters: requires_grad = False
params = [model.language_model.lm_head.weight]  # Only this layer
opt = torch.optim.Adam(params, lr=1e-4)
```

### 4c. Training Step
```python
for iteration in range(n_iter):
    model.zero_grad()
    outputs = model(**tokens)  # Forward pass
    loss = outputs.loss  # Loss = 2.30 (from position 42)
    
    loss.backward()  # Compute gradients for lm_head.weight only
    opt.step()       # Update lm_head.weight
    opt.zero_grad()
```

### 4d. After Optimization
```python
# After updating lm_head.weight, the model should predict "cab" 
# with higher probability given the prompt context.
# Next forward pass might give:
# logits[0, 42, 1234] = 3.5  (higher logit)
# softmax(...) = 0.30  (30% probability)
# loss = -log(0.30) = 1.20  (lower loss!)
```

---

## Key Points

1. **Autoregressive**: Model predicts next token at each position using all previous tokens
2. **Masked Loss**: Only answer tokens (position 42) contribute to loss; prompt tokens (-100) are ignored
3. **Teacher Forcing**: During training, model sees the full sequence `[prompt, answer]` but only computes loss on answer tokens
4. **Layer-wise**: Only selected layer (e.g., `lm_head.weight`) is updated; all others frozen

---

## Visualization

```
Input Sequence (what model sees):
┌─────────────────────────────────────────────┬──────┐
│ PROMPT (42 tokens)                          │ANSWER│
├─────────────────────────────────────────────┼──────┤
│[BOS] Choose the correct ... delivery [EOS] │ cab  │
└─────────────────────────────────────────────┴──────┘
        ↓
    Model processes sequentially:
    
Position: 0   1    2    3  ... 41   42
Context:  [] [BOS][BOS,Choose] ... [full prompt] [full prompt + ???]
Predict:  518 4555 297  3419 ... 8906 1234 ← Only this loss counts!
                              ↑
                        Answer position (labels != -100)
```

---
