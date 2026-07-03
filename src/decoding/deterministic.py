'''
Build DeterministicDecoder
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class DeterministicDecoder:
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 256, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def _get_input_and_mask(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt") #type: ignore
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attn_mask
    
    def greedy_search(self, prompt: str) -> str:
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        for _ in range(self.max_new_tokens):
            
            logits = self.model( #type: ignore
                input_ids=input_ids,
                attention_mask=attn_mask,
            ).logits
            
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            if next_token_id == self.tokenizer.eos_token_id: #type: ignore
                break
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attn_mask = torch.ones_like(input_ids)
        
        output = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True) #type: ignore
        return output

    def beam_search(self, prompt: str, num_beams: int = 3) -> str:
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        beams = [(input_ids, attn_mask, 0.0, False)] # [input, mask, score]
        
        for _ in range(self.max_new_tokens):
            new_beams = []
            
            for input, mask, score, finished in beams:
                if finished:
                    new_beams.append((input, mask, score, True))
                    continue
                
                logits = self.model( #type: ignore
                    input_ids=input,
                    attention_mask=mask,   
                ).logits
                
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                log_score, token_id = torch.topk(log_probs, k=num_beams, dim=-1)
                
                for i in range(num_beams):
                    next_token = token_id[:, i:i + 1]
                    new_input = torch.cat([input, ], dim=-1)
                    new_mask = torch.ones_like(new_input)
                    new_score = score + log_score[:, i].item()
                    is_finished = next_token.item() == self.tokenizer.eos_token_id #type: ignore
                    
                    new_beams.append(
                        (new_input, new_mask, new_score, is_finished)
                    )
            
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]
            
            if all(b[3] for b in beams):
                break
        
        output_ids = beams[0][0]
        outputs = self.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True) #type: ignore
        
        return outputs