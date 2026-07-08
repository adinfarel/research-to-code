'''
Build PagedAttention Simulation implementation
'''

class PagedAttention:
    
    def __init__(self, total_vram_blocks=10, block_size=4):
        self.block_size = block_size
        
        # PHYSICAL BLOCKS
        self.free_physical_blocks = [i for i in range(total_vram_blocks)]
        self.physical_vram = {i: [] for i in range(total_vram_blocks)}
        # PAGE TABLE
        self.page_table = {}
    
    def register_new_request(self, request_id, prompt_tokens):
        print(f"\n===== [PREFILL] Request {request_id} sign in with {len(prompt_tokens)} Token =====")
        
        num_blocks_needed = (len(prompt_tokens) + self.block_size - 1) // self.block_size
        
        if num_blocks_needed > len(self.free_physical_blocks):
            print("ERROR: VRAM FULL! Can not accommodating new request")
            return False
        
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_physical_blocks.pop(0)
            allocated_blocks.append(block_id)
            
        self.page_table[request_id] = allocated_blocks
        
        curr_token_idx = 0
        for block_id in allocated_blocks:
            chunk = prompt_tokens[curr_token_idx : curr_token_idx + self.block_size]
            self.physical_vram[block_id] = chunk
            curr_token_idx += self.block_size
        
        self.print_status_table()
    
    def decode_next_token(self, request_id, new_token):
        print(f"\n===== [DECODE] Request {request_id} Generating New Token: '{new_token}' =====")
        
        user_assigned_blocks = self.page_table[request_id]
        last_physical_block_id = user_assigned_blocks[-1]
        
        if len(self.physical_vram[last_physical_block_id]) < self.block_size:
            self.physical_vram[last_physical_block_id].append(new_token)
            print(f"-> Empty slot found at Block Physic #{last_physical_block_id}")
        else:
            if not self.free_physical_blocks:
                print(f"OUT OT MEMORY (OOM): VRAM Genuinely run out while decode")
                return False
            
            new_block_id = self.free_physical_blocks.pop(0)
            print(f"-> Block Physic #{last_physical_block_id} full, take new Block Physic #{new_block_id}")
            
            self.page_table[request_id].append(new_block_id)
            self.physical_vram[new_block_id] = [new_token]
        
        self.print_status_table()
    
    def release_request(self, request_id):
        print(f"\n===== [EOS] Request {request_id} FINISH. Clear Memori... =====")
        if request_id not in self.page_table:
            return
        
        rented_blocks = self.page_table[request_id]
        
        for block_id in rented_blocks:
            self.physical_vram[block_id] = []
            self.free_physical_blocks.append(block_id)
        
        del self.page_table[request_id]
        
        print(f"-> Blocks {rented_blocks} success back into free pool.")
        self.print_status_table()
    
    def print_status_table(self):
        print("\n--- LAYER 1: PAGE TABLE (LOGICAL TO PHYSICAL MAP) ---")
        for req_id, blocks in self.page_table.items():
            logical_view = " -> ".join([f"LogicBlock_{i}(Physic #{b})" for i, b in enumerate(blocks)])
            print(f" Request {req_id} : {logical_view}")
            
        print("\n--- LAYER 2: PHYSICAL VRAM TABLE (KONDISI HARDWARE) ---")
        for block_id, tokens in self.physical_vram.items():
            status = "FREE" if block_id in self.free_physical_blocks else "ALLOCATED"
            slots = [tokens[i] if i < len(tokens) else "EMPTY" for i in range(self.block_size)]
            print(f" [Block Physic #{block_id:02d}] Status: {status:<11} | Filled: {slots}")
        print("-" * 75)

if __name__ == "__main__":
    server = PagedAttention(total_vram_blocks=5, block_size=3)
    server.register_new_request("User_A", ["Halo", "Adin", "Want", "Ask", "LLM"])
    
    server.decode_next_token("User_A", "It")
    
    server.register_new_request("User_B", ["Coffe", "Bitter"])
    
    server.decode_next_token("User_A", "What")
    server.release_request("User_B")