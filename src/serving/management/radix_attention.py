'''
Build RadixAttention simulation implementation

INTUITION:
    Many request have a same prompt such as System Prompt that use over and over
    the motivation is, Why we not cache the prompt? We cache the prompt that often appear
    and if we meet again similiar prompt we just use that prompt cache
    
    if we dealing request and request ended, we usually wipe or clear the KV-cache for the prompt
    but RadixAttention do not that, RA keep the KV-cache of prompt user to evictable mode instead of
    clear or wipe kv-cache if request user ended during VRAM still not full, but save KV-cache instead of clear
    isn't it make memory save larger? yeah it's true, but RA present LRU (Least Recently Used) RA check if
    there is node prompt that rarely use back into new request or seldom visited (less accessed) we clear that kv-cache
    
    for Matchy the prompt if it is the same is with RadixTree which is search that have same tokens (meaning?)
    RadixTree use memoization algorithm but not use hashmap but activation tensor KV-cache at biner level VRAM
'''

import time

class RadixNode:
    
    def __init__(self, prefix_tokens, physical_blocks):
        self.prefix_tokens = prefix_tokens
        self.physical_blocks = physical_blocks
        self.children = {}
        self.last_accessed = time.time()

class RadixAttention:
    
    def __init__(self, total_vram_blocks=10, block_size=3):
        self.block_size = block_size
        
        self.free_physical_blocks = [i for i in range(total_vram_blocks)]
        self.physical_vram = {i: [] for i in range(total_vram_blocks)}
        
        self.root = RadixNode(prefix_tokens=[], physical_blocks=[])
    
    def _allocate_blocks_for_tokens(self, tokens):
        num_blocks_needed = (len(tokens) + self.block_size - 1) // self.block_size
        
        while num_blocks_needed > len(self.free_physical_blocks):
            print(f"VRAM Full! Running LRU Eviction for to free space")
            if not self._evict_least_recently_used(self.root):
                print(f"CRITICAL ERROR: VRAM truly run out and nothing cache that evictable (OOM)!")
        
        allocated = []
        for _ in range(num_blocks_needed):
            allocated.append(self.free_physical_blocks.pop(0))
        return allocated
    
    def match_prefix(self, node: RadixNode, tokens):
        """THE MAIN IDEA --> RADIX TREE"""
        node.last_accessed = time.time()
        
        if not tokens:
            return node, [], []
        
        first_token = tokens[0]
        if first_token in node.children:
            child = node.children[first_token]
            child_tokens = child.prefix_tokens
            
            match_len = 0
            for i in range(min(len(tokens), len(child_tokens))):
                if tokens[i] == child_tokens[i]:
                    match_len += 1
                else:
                    break
            
            if match_len == len(child_tokens):
                return self.match_prefix(child, tokens[match_len:])
            
            return child, tokens[:match_len], tokens[match_len:]

        return node, [], tokens
    
    def process_request(self, request_id, prompt_tokens):
        """MEMOIZATION FLOW"""
        print(f"\n===== [REQUEST] {request_id} Come Bring a Prompt: {prompt_tokens} =====")
        matched_node, matched_tokens, remaining_tokens = self.match_prefix(self.root, prompt_tokens)
        
        if matched_tokens or matched_node != self.root:
            cached_blocks = matched_node.physical_blocks
            print(f"[CACHE HIT] Part {prompt_tokens[:len(prompt_tokens)-len(remaining_tokens)]} DECISION TO USE PREFIX CACHE!")
            print(f"   -> GPU Save Prefill! Directly use Physic Block: {cached_blocks}")
            
        if remaining_tokens:
            print(f"[CACHE MISS] Part {remaining_tokens} forced to Prefill directly by GPU...")
            
            new_blocks = self._allocate_blocks_for_tokens(prompt_tokens)
            if not new_blocks: return
            
            curr_idx = 0
            for block_id in new_blocks:
                chunk = prompt_tokens[curr_idx : self.block_size + curr_idx]
                self.physical_vram[block_id] = chunk
                curr_idx += self.block_size
            
            new_node = RadixNode(prefix_tokens=remaining_tokens, physical_blocks=new_blocks)
            matched_node.children[remaining_tokens[0]] = new_node
        
        self.print_tree_and_vram()
    
    def _evict_least_recently_used(self, current_node: RadixNode):
        if not current_node.children:
            return False

        oldest_time = time.time()
        oldest_key = None
        oldest_node = None
        
        for key, child in current_node.children.items():
            if child.last_accessed < oldest_time:
                oldest_time = child.last_accessed
                oldest_key = key
                oldest_node = child
        
        if oldest_node:
            print(f"[LRU EVICTION] Cutting a Old Branch Cache: {oldest_node.prefix_tokens}")
            
            for b in oldest_node.physical_blocks:
                self.physical_vram[b] = []
                self.free_physical_blocks.append(b)
            
            del current_node.children[oldest_key]
            return True
        return False
    
    def print_tree_and_vram(self):
        """HELPER"""
        print("\n--- VISUALIZATION STRUCTURE RADIX TREE (PAGE TABLE HIERARCHIES) ---")
        def _traverse(node, depth=0):
            indent = "  " * depth
            if depth > 0:
                print(f"{indent} --- Text: {node.prefix_tokens} -> Pointing Physic Block: {node.physical_blocks}")
            for child in node.children.values():
                _traverse(child, depth + 1)
        _traverse(self.root)
        
        print("\n--- STORAGE LAYER: PHYSICAL VRAM STATUS ---")
        for b, tokens in self.physical_vram.items():
            status = "FREE" if b in self.free_physical_blocks else "CACHED/USED"
            print(f" [Physic Block #{b:02d}] Status: {status:<14} | Data Fill: {tokens}")
        print("="*80)

if __name__ == "__main__":
    server = RadixAttention(total_vram_blocks=4, block_size=2)
    
    server.process_request("User_1", ["Coffe", "Bitter", "Make", "Fun"])
    server.process_request("User_2", ["Coffe", "Bitter", "Without", "Sugar"])
    
    server.process_request("User_3", ["Tea", "Sweet"])