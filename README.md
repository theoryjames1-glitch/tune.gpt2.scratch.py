# tune.gpt2.scratch.py

```python
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments
)
from tokenizers import ByteLevelBPETokenizer
from accelerate import dispatch_model

# -----------------------------
# 1. Load dataset (example: WikiText-2)
# -----------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Save dataset text for tokenizer training
with open("train_text.txt", "w", encoding="utf-8") as f:
    for ex in dataset["train"]["text"]:
        f.write(ex + "\n")

# -----------------------------
# 2. Train GPT-2 style tokenizer
# -----------------------------
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files="train_text.txt",
    vocab_size=50257,             # same size as GPT-2
    min_frequency=2,
    special_tokens=["<|endoftext|>"]
)
tokenizer.save_model("./gpt2-scratch-tokenizer")

# Load as HF tokenizer
hf_tokenizer = GPT2TokenizerFast.from_pretrained("./gpt2-scratch-tokenizer")
hf_tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
hf_tokenizer.pad_token = hf_tokenizer.eos_token  # pad = eos like original GPT-2

# -----------------------------
# 3. Tokenize dataset
# -----------------------------
def tokenize_function(examples):
    return hf_tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Group into contiguous blocks
block_size = 512
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(group_texts, batched=True)

# -----------------------------
# 4. Create GPT-2 model config + model
# -----------------------------
config = GPT2Config(
    vocab_size=len(hf_tokenizer),
    n_layer=12,      # GPT-2 small
    n_head=12,
    n_embd=768,
    n_positions=1024,
    eos_token_id=hf_tokenizer.eos_token_id,
    pad_token_id=hf_tokenizer.eos_token_id
)

model = GPT2LMHeadModel(config)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()


# -----------------------------
# 6. Training config
# -----------------------------
training_args = TrainingArguments(
    output_dir="./gpt2-scratch-offload",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,     # effective batch size = 8
    learning_rate=5e-4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,                         # mixed precision
    optim="paged_adamw_8bit"           # memory-efficient optimizer
)

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    tokenizer=hf_tokenizer,
)

# -----------------------------
# 8. Train & Save
# -----------------------------
trainer.train()
trainer.save_model("./gpt2-scratch-offload-final")
hf_tokenizer.save_pretrained("./gpt2-scratch-offload-final")

```
