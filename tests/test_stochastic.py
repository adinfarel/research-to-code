'''
testing Stochastic Decoder
'''

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from src.decoding.stochastic import StochasticDecoder
from tests.test_deterministic import _tiny_model_and_tokenizer

def test_temperature_sampling_output_type():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = StochasticDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    output = decoder.temperature_sampling("Hello", temperature=1.0)
    assert isinstance(output, str)

def test_top_k_restricts_to_exactly_k_candidates():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = StochasticDecoder(model, tokenizer, max_new_tokens=1, device=DEVICE) #type: ignore

    input_ids, attn_mask = decoder._get_input_and_mask("Hello")
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    logits_temp = logits[:, -1, :]

    k = 10
    top_values, top_indices = torch.topk(logits_temp, k=k, dim=-1)
    masked = torch.full_like(logits_temp, float('-inf'))
    masked.scatter_(dim=-1, index=top_indices, src=top_values)

    non_inf_count = (masked != float('-inf')).sum().item()
    assert non_inf_count == k

def test_top_p_sampling_output_type():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = StochasticDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    output = decoder.top_p_sampling("Hello", p=0.9)
    assert isinstance(output, str)

def test_top_p_lower_p_keeps_fewer_candidates():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = StochasticDecoder(model, tokenizer, max_new_tokens=1, device=DEVICE) #type: ignore

    input_ids, attn_mask = decoder._get_input_and_mask("Hello")
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    probs = torch.softmax(logits[:, -1, :], dim=-1)

    sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    n_tokens_low_p = (cumulative <= 0.5).sum().item() + 1
    n_tokens_high_p = (cumulative <= 0.95).sum().item() + 1

    assert n_tokens_low_p <= n_tokens_high_p