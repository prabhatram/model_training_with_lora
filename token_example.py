import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

text = """
A temporarily locked account should be
left for 15 minutes.
"""


tokens = tokenizer.tokenize(text)

ids = tokenizer.encode(text)


print(tokens)
print(ids)