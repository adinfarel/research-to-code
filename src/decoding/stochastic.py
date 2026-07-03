'''
Build StochasticDecoder 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class StochasticDecoder:
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 256,
                 device: str = "cuda", seed: int = 42):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.seed = seed
    
    def _get_input_and_mask(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt") #type: ignore
        
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        
        return input_ids, attn_mask
        
    def temperature_sampling(self, prompt: str, temperature: float = 1.0):
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        for _ in range(self.max_new_tokens):
            logits = self.model( #type: ignore
                input_ids=input_ids,
                attention_mask=attn_mask,
            ).logits
            
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            if next_token_id == self.tokenizer.eos_token_id: #type: ignore
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attn_mask = torch.ones_like(input_ids)
        
        output = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True) #type: ignore
        return output
    
    def top_k_sampling(self, prompt: str, temperature: float = 1.0, k: int = 50):
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        for _ in range(self.max_new_tokens):
            logits = self.model( #type: ignore
                input_ids=input_ids,
                attention_mask=attn_mask,
            ).logits
            
            logits_temp = logits[:, -1, :] / temperature
            
            top_values, top_indices = torch.topk(logits_temp, k=k, dim=-1)
            
            masked = torch.full_like(logits_temp, float("-inf"))
            masked.scatter_(dim=-1, index=top_indices, src=top_values)
            
            probs = torch.softmax(masked, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            if next_token_id == self.tokenizer.eos_token_id: #type: ignore
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attn_mask = torch.ones_like(input_ids)
        
        output = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True) #type: ignore
        return output
    
    def top_p_sampling(self, prompt: str, p: float = 0.9, temperature: float = 1.0):
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        for _ in range(self.max_new_tokens):
            logits = self.model( #type: ignore
                input_ids=input_ids,
                attention_mask=attn_mask,
            ).logits
            
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            cummulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            cutoff_mask = cummulative_probs > p
            cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
            cutoff_mask[:, 0] = False
            
            sorted_probs[cutoff_mask] = 0.0
            
            nucleus_probs = torch.zeros_like(probs)
            nucleus_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
            nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)
            
            # total = 0.0
            # for i in range(len(probs)):
            #     if total > 0.9:
            #         last_idx = i
            #         break
                
            #     total = total + probs_sort[i]
            
            # probs_sort[torch.arange(probs.shape[0]) > last_idx] = float('-inf') #type: ignore
            
            # top_nucleus = F.softmax(probs_sort, dim=-1) #type: ignore
            next_token_id = torch.multinomial(nucleus_probs, num_samples=1)
            
            if next_token_id == self.tokenizer.eos_token_id: #type: ignore
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attn_mask = torch.ones_like(input_ids)
        
        output = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True) #type: ignore
        return output