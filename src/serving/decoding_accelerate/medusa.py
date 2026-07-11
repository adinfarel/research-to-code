'''
Build Medusa implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from src.serving.decoding_accelerate.model_loader import load_target_model

class MedusaHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, version: str = "medusa1"):
        super().__init__()
        self.version = version
        
        if version == "medusa1":
            self.block = nn.Linear(embed_dim, vocab_size)
        elif version == "medusa2":
            self.linear1 = nn.Linear(embed_dim, vocab_size)
            self.act = nn.SiLU()
            self.linear2 = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor):
        if self.version == "medusa1":
            return self.block(x)
        
        res = x
        x = self.linear1(x)
        x = self.act(x)
        out = x + res
        return self.linear2(out)
    
class MedusaEngine:
    
    def __init__(self, n_heads: int = 3, top_k: int = 2, version: str = "medusa1", device="cpu"):
        self.device = device
        self.n_heads = n_heads  
        self.top_k = top_k       
        self.version = version
        
        self.model, self.tokenizer = load_target_model()
        
        self.embed_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        self.medusa_heads = nn.ModuleList([
            MedusaHead(self.embed_dim, self.vocab_size, version=self.version)
            for _ in range(self.n_heads)
        ]).to(self.device)
    
    def _get_input_and_mask(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)
        
        return input_ids, attn_mask
    
    def train_medusa(self, train_dataloader, epochs: int = 1, lr: float = 1e-4):
        if self.version == "medusa1":
            self.model.requires_grad_(False)
            self.medusa_heads.requires_grad_(True)
            optimizer = torch.optim.AdamW(self.medusa_heads.parameters(), lr=lr)
        else:
            self.model.requires_grad_(True)
            self.medusa_heads.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                list(self.model.parameters()) + list(self.medusa_heads.parameters()), lr=lr
            )
        
        self.model.train()
        self.medusa_heads.train()
        
        for epoch in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                
                total_loss = 0
                criterion = nn.CrossEntropyLoss()
                
                for i, head in enumerate(self.medusa_heads):
                    head_logits = head(last_hidden_states)
                    shift_steps = i + 1
                    logits_slice = head_logits[:, :-(shift_steps), :].contiguous()
                    labels_slice = input_ids[:, shift_steps:].contiguous()
                    
                    loss = criterion(logits_slice.view(-1, self.vocab_size), labels_slice.view(-1))
                    total_loss += loss
                
                total_loss.backward() #type:ignore --> calculate grad
                optimizer.step() # update parameters
        
    @torch.no_grad()
    def _generate_tree_candidates(self, last_hidden_states: torch.Tensor, base_next_token: torch.Tensor):
        '''Build medusa proposal tree.'''
        target_hidden = last_hidden_states[:, -1:, :]
        root_token = base_next_token[0, 0].item()
        
        branch_per_heads = []
        for head in self.medusa_heads:
            head_logits = head(target_hidden)
            probs = F.softmax(head_logits[:, 0, :], dim=-1)
            top_k_tokens = torch.topk(probs, self.top_k, dim=-1).indices.squeeze(0)
            branch_per_heads.append(top_k_tokens.tolist())
        
        all_paths = []
        
        for branch in itertools.product(*branch_per_heads):
            path = [root_token]
            path.extend(branch)
            
            all_paths.append(path)
        
        tree_candidates = torch.tensor(
            all_paths,
            dtype=torch.long,
            device=self.device,
        )
        
        return tree_candidates
    
    @torch.no_grad()
    def _tree_attention_mask(self, tree_candidates: torch.Tensor):
        n_paths, length_sequences = tree_candidates.shape
        total_tokens = n_paths * length_sequences
        
        tree_mask = torch.full((total_tokens, total_tokens), float("-inf"), device=self.device)
        
        # MAKE TREE ATTENTION LIKE THIS
        #             Token Path 0 [0:4]   Token Path 1 [4:8]   ...
        #         ┌────────────────────┬─────────────────────┬─────┐
        # Path  │  0.0  -inf -inf -inf│ -inf  -inf -inf  -inf │     │
        # 0     │  0.0   0.0  -inf -inf│ -inf  -inf -inf  -inf│    │
        # [0:4] │  0.0   0.0   0.0 -inf│ -inf  -inf -inf  -inf│-inf│
        #       │  0.0   0.0   0.0  0.0│ -inf  -inf -inf  -inf│    │
        #         ├────────────────────┼─────────────────────┼─────┤
        # Path  │ -inf  -inf -inf -inf │  0.0  -inf -inf  -inf│    │
        # 1     │ -inf  -inf -inf -inf │  0.0   0.0  -inf -inf│    │
        # [4:8] │ -inf  -inf -inf -inf │  0.0   0.0   0.0 -inf│-inf│
        #       │ -inf  -inf -inf -inf │  0.0   0.0   0.0  0.0│    │
        #         ├────────────────────┼─────────────────────┼─────┤
        #     ... │        -inf        │         -inf        │ 0.0 │
        #         └────────────────────┴─────────────────────┴─────┘
        for path_idx in range(n_paths):
            start_idx = path_idx * length_sequences
            end_idx = start_idx + length_sequences
            
            causal_matrix = torch.tril(torch.ones(length_sequences, length_sequences, device=self.device))
            
            tree_mask[start_idx:end_idx, start_idx:end_idx] = torch.where(
                causal_matrix == 1, torch.tensor(0.0, device=self.device), torch.tensor(float("-inf"), device=self.device)
            )
        
        return tree_mask

    @torch.no_grad()
    def _verify_single_path(
        self,
        input_ids: torch.Tensor,
        path: torch.Tensor,
    ):
        """Naive medusa verification."""
        
        verify_input = torch.cat([input_ids, path.unsqueeze(0)], dim=-1)
        
        outputs = self.model(input_ids=verify_input)
        logits = outputs.logits
        
        context_len = input_ids.shape[-1]
        accepted = 0
        
        for i in range(path.shape[0]):
            predicted = torch.argmax(logits[0, context_len + i - 1], dim=-1)
            
            if predicted.item() == path[i].item():
                accepted += 1
            else:
                break
        
        return accepted

    @torch.no_grad()
    def _verify_all_paths(
        self,
        input_ids,
        tree_candidates,
    ):
        accepted_lengths = []
        
        for path in tree_candidates:
            accepted = self._verify_single_path(
                input_ids,
                path
            )
            
            accepted_lengths.append(accepted)
        
        return accepted_lengths

    @torch.no_grad()
    def _select_winner(
        self,
        tree_candidates,
        accepted_lengths,
    ):
        winner = torch.argmax(torch.tensor(accepted_lengths), dim=-1).item()
        winner_path = tree_candidates[winner]
        accept_len = accepted_lengths[winner]
        accepted_prefix = winner_path[:accept_len]
        
        return accepted_prefix
    
    @torch.no_grad()
    def generate(self, prompt: str, max_steps: int = 15):
        input_ids, attn_mask = self._get_input_and_mask(prompt)
        prompt_len = input_ids.shape[-1]
        
        step = 0
        while step < max_steps:
            outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            base_logits = outputs.logits
            last_hidden_states = outputs.hidden_states[-1]
            
            base_next_token = torch.argmax(base_logits[:, -1, :], dim=-1, keepdim=True)
            
            tree_candidates = self._generate_tree_candidates(last_hidden_states, base_next_token)
            n_paths, length_sequences = tree_candidates.shape
            
            # ============================================================
            # REAL MEDUSA
            #
            # Uncomment this section if using a patched transformer
            # supporting Tree Attention.
            #
            # flat_tree_tokens = tree_candidates.view(1, -1)
            # tree_mask = self._tree_attention_mask(tree_candidates)
            # combined_input_ids = torch.cat([input_ids, flat_tree_tokens], dim=-1)
            # outputs = self.model(
            #     input_ids=combined_input_ids,
            #     attention_mask=tree_mask,
            # )
            #
            # ============================================================
            
            accepted_lengths = self._verify_all_paths(
                input_ids,
                tree_candidates,
            )
            
            accepted_prefix = self._select_winner(
                tree_candidates,
                accepted_lengths
            )
            
            if accepted_prefix.numel() == 0:
                accepted_prefix = base_next_token.squeeze(0)
                
            accepted_prefix = accepted_prefix.unsqueeze(0)
            
            input_ids = torch.cat([input_ids, accepted_prefix], dim=-1)
            attn_mask = torch.ones_like(input_ids)
            
            if (accepted_prefix == self.tokenizer.eos_token_id).any():
                break
                
            step += 1
        
        output_text = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
        return output_text