'''
Build EAGLE implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class EAGLEDraftHead(nn.Module):
    def __init__(self, hidden_size=1024, vocab_size=49152, version="eagle-2"):
        super().__init__()
        self.version = version
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.draft_embedding = nn.Embedding(vocab_size, hidden_size)
        
        if self.version == "eagle-3":
            self.fusion_projector = nn.Linear(hidden_size * 3, hidden_size)
            self.direct_token_predictor = nn.Linear(hidden_size, vocab_size)
        
        else:
            self.draft_transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*2, batch_first=True
            )
            self.draft_lm_head = nn.Linear(hidden_size, vocab_size)
    
    def shallow_forward(self, f_input, next_token_id):
        e_token = self.draft_embedding(next_token_id)
        
        f_combined = f_input + e_token
        
        if self.version == "eagle-3":
            logits = self.direct_token_predictor(f_combined)
            h_laten = f_combined
        else:
            f_new = self.draft_transformer_layer(f_combined)
            logits = self.draft_lm_head(f_new)
            h_laten = f_new
        
        return logits, h_laten
    
    def advanced_forward(self, h_laten, token_id):
        e_token = self.draft_embedding(token_id)
        f_combined = h_laten + e_token
        
        if self.version == "eagle-3":
            logits = self.direct_token_predictor(f_combined)
            h_laten = f_combined
        else:
            f_new = self.draft_transformer_layer(f_combined)
            logits = self.draft_lm_head(f_new)
            h_laten = f_new
        
        return logits, h_laten

class EAGLEEngine:
    
    def __init__(self, base_model, hidden_size=1024, vocab_size=49152, version="eagle-2"):
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.version = version

        self.draft_head = EAGLEDraftHead(hidden_size=hidden_size, vocab_size=vocab_size, version=version)
    
    def _create_hf_tree_attn_mask(self, history_len, tree_structure, batch_size=1, num_heads=4):
        total_tree_tokens = sum([len(path) for path in tree_structure])
        total_seq_len = history_len + total_tree_tokens
        
        mask = torch.zeros((total_seq_len, total_seq_len))
        
        for i in range(total_seq_len):
            mask[i, i+1:] = float("-inf")
        
        start_idx = history_len
        for path in tree_structure:
            end_idx = start_idx + len(path)
            
            mask[start_idx:end_idx, :history_len] = 0.0
            
            mask[start_idx:end_idx, history_len:start_idx] = float("-inf")
            mask[start_idx:end_idx, end_idx:] = float("-inf")
            
            start_idx = end_idx
        
        mask4d = mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_heads, 1, 1)
        return mask4d
    
    def generative_speculative_tree(self, last_hidden_states, multi_layer_features, base_next_token_id):
        batch_size = last_hidden_states.shape[0]
        
        if self.version == "eagle-3":
            f_low, f_mid, f_high = multi_layer_features
            f_combined = torch.cat([f_low, f_mid, f_high], dim=-1)
            f_fused = self.draft_head.fusion_projector(f_combined[:, -1:, :])
            f_input = f_fused
        
        else:
            f_input = last_hidden_states[:, -1:, :]
        
        logits_t1, h_laten_t1 = self.draft_head.shallow_forward(f_input, base_next_token_id)
        
        top_k_t1 = torch.topk(logits_t1, k=2, dim=-1)
        tokens_path_a = top_k_t1.indices[:, :, 0]
        tokens_path_b = top_k_t1.indices[:, :, 1]
        
        logits_t2_a, _ = self.draft_head.advanced_forward(h_laten_t1, tokens_path_a)
        last_token_a = torch.topk(logits_t2_a, k=1, dim=-1).indices[:, :, 0]
        
        logits_t2_b, _ = self.draft_head.advanced_forward(h_laten_t1, tokens_path_b)
        last_token_b = torch.topk(logits_t2_b, k=1, dim=-1).indices[:, :, 0]
        
        tree_candidates = [
            [base_next_token_id.item(), tokens_path_a.item(), last_token_a.item()],
            [base_next_token_id.item(), tokens_path_b.item(), last_token_b.item()],
        ]
        return tree_candidates
    
    def verify_and_select_winner(self, input_ids, tree_candidates):
        history_len = input_ids.shape[1]
        
        flat_tokens = []
        for path in tree_candidates:
            flat_tokens.extend(path)
        flat_tensor = torch.tensor([flat_tokens], device=input_ids.device)
        
        combined_inputs = torch.cat([input_ids, flat_tensor], dim=-1)
        
        attn_mask = self._create_hf_tree_attn_mask(
            history_len, tree_candidates, input_ids.shape[0]
        )
        
        base_logits = self.base_model(
            input_ids=combined_inputs,
            attention_mask=attn_mask,
        )
        probs = F.softmax(base_logits, dim=-1)
        
        score_path = []
        
        pointer_token = history_len
        
        for path in tree_candidates:
            log_likelihood = 0.0
            eval_pointer = history_len - 1
            
            for token_id in path:
                prob_word = probs[0, eval_pointer, token_id].item()
                log_likelihood += torch.log(torch.tensor(prob_word + 1e-9)).item()
                eval_pointer = pointer_token
                pointer_token += 1
            
            score_path.append(log_likelihood)
        
        score_tensor = torch.tensor(score_path)
        idx_winner = torch.argmax(score_tensor).item()
        
        path_winner = tree_candidates[idx_winner]
        
        return path_winner