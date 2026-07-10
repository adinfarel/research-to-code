'''
Utilies for loaded model
'''

from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DRAFT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TARGET_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

@lru_cache(maxsize=1)
def load_draft_model(model_name: str = DRAFT_MODEL, device: str = "cpu"):
    model     = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device) #type: ignore
    model.eval()
    
    return model, tokenizer

@lru_cache(maxsize=1)
def load_target_model(model_name: str = TARGET_MODEL, device: str = "cpu"):
    model     = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device) #type: ignore
    model.eval()
    
    return model, tokenizer