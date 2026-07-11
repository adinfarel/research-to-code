'''
src/decoding/model_loader.py

Load pretrained model + tokenizer, shared across deterministic dan
stochastic decoding classes.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache


@lru_cache(maxsize=1)
def load_model(model_name: str = "meta-llama/Llama-3.2-1B", device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device) # type: ignore
    model.eval()
    return model, tokenizer