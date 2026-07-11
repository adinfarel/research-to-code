'''
Build SpeculativeDecoding implementation
'''

import torch
import torch.nn.functional as F

from src.serving.decoding_accelerate.model_loader import load_draft_model, load_target_model

class SpeculativeDecoding:
    
    def __init__(self, n_tokens: int = 4, device="cpu"):
        self.draft_model, self.tokenizer = load_draft_model()
        self.target_model, _ = load_target_model()
        self.device = device
        
        self.n_tokens = n_tokens
    
    def _get_input_and_mask(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        
        return input_ids, attn_mask
    
    @torch.no_grad()
    def _generate_draft(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, do_sample: bool=False):
        draft_tokens = []
        draft_probs_list = []
        
        curr_input_ids = input_ids.clone()
        curr_attn_mask = attn_mask.clone()
        
        for _ in range(self.n_tokens):
            logits = self.draft_model(
                input_ids=curr_input_ids,
                attention_mask=curr_attn_mask
            ).logits
            
            probs = F.softmax(logits[:, -1, :], dim=-1)
            
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
            draft_tokens.append(next_token)
            draft_probs_list.append(probs)
            
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            curr_attn_mask = torch.ones_like(curr_input_ids)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        draft_tokens = torch.cat(draft_tokens, dim=-1) # Shape: [N, actual_n_tokens]
        draft_probs = torch.cat(draft_probs_list, dim=0).unsqueeze(0) # Shape: [N, actual_n_tokens, vocab]
        
        return draft_tokens, draft_probs
    
    def _verify_and_sample(self, draft_tokens: torch.Tensor, draft_probs: torch.Tensor,
                           target_probs: torch.Tensor, do_sample: bool = False):
        '''
        if do_sample=False
        Case 1: draft_token == argmax(target), then accept
        Case 2: rejected, take target token and reject all the next token in draft
        
        if do_sample=True
        Case 1: q(x) >= p(x), then accept
        Case 2: q(x) < p(x), test sampling based on ratio = q(x)/p(x)
        Case 3: rejected, max(0, q(x) - p(x))
        '''
        n_actual = draft_tokens.shape[-1]
        accepted_tokens = []
        is_all_correct = True
        
        for i in range(n_actual):
            tokens = draft_tokens[:, i].item()    

            p_draft = draft_probs[:, i, tokens].item() #type: ignore
            p_target = target_probs[:, i, tokens].item() #type: ignore
            
            if not do_sample:
                target_choice = torch.argmax(target_probs[:, i, :], dim=-1).item()
                if target_choice == tokens:
                    accepted_tokens.append(tokens)
                else:
                    accepted_tokens.append(target_choice)
                    is_all_correct = False
                    break
            else:
                if p_target >= p_draft:
                    accepted_tokens.append(tokens)
                else:
                    u = torch.rand(1).item()
                    if u < (p_target / p_draft):
                        accepted_tokens.append(tokens)
                    else:
                        residual_dist = torch.clamp(target_probs[:, i, :] - draft_probs[:, i, :], min=0)
                        residual_dist = residual_dist / residual_dist.sum()
                        correction = torch.multinomial(residual_dist, num_samples=1).item()
                        
                        accepted_tokens.append(correction)
                        is_all_correct = False
                        break
        
        if is_all_correct:
            if not do_sample:
                bonus_token = torch.argmax(target_probs[:, -1, :], dim=-1)
            else:
                bonus_token = torch.multinomial(target_probs[:, -1, :], num_samples=1).item()
            accepted_tokens.append(bonus_token)
        
        return torch.tensor([accepted_tokens], dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, do_sample: bool = False):
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        generated_tokens = 0
        while generated_tokens < max_new_tokens:
            draft_tokens, draft_probs = self._generate_draft(input_ids, attn_mask, do_sample=do_sample)
            n_actual = draft_tokens.shape[-1]
            
            combined_input_ids = torch.cat([input_ids, draft_tokens], dim=-1)
            combined_attn_mask = torch.ones_like(combined_input_ids)
            
            logits = self.target_model(
                input_ids=combined_input_ids,
                attention_mask=combined_attn_mask,
            ).logits
            
            target_probs = F.softmax(logits[:, -(n_actual+1):, :], dim=-1)
            
            verified_tokens = self._verify_and_sample(draft_tokens, draft_probs, target_probs, do_sample)
            
            input_ids = torch.cat([input_ids, verified_tokens], dim=-1)
            attn_mask = torch.ones_like(input_ids)
            
            generated_tokens += verified_tokens.shape[-1]
            
            if (verified_tokens == self.tokenizer.eos_token_id).any():
                break
        
        output_text = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
        return output_text