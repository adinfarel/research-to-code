'''
Build MHA atom level

NOTE: this is anti-mainstream implementation, cause i am dealing with 4D hell no

Exact same with intuition from 'scaled_dot_product.py __docs__'
but the main is not matrix again but tensor

Here some explanation:
    -> each tensor contain 4D, consist is (Batch, Head, Seq_Len, Embed)
        Exam:
            Q @ K.T
            --> (B, H, T, C) @ (B, H, C, T)
            --> (B, H, T, T)
            
            Wei @ V
            --> (B, H, T, T) @ (B, H, T, C)
            --> (B, H, T, C)
    
    that's the whole main idea behind this, not scary at all if we understand under the hood

NOTE: if there mistake from intuition or implementation, just call me >.<
'''

# tomorrow implementation cause i sleepy