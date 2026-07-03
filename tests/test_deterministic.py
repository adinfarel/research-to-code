'''
testing Deterministic Decoding
'''

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

from src.decoding.deterministic import DeterministicDecoder

def _tiny_model_and_tokenizer():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config=config)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    return model, tokenizer

def test_greedy_search_output_type():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = DeterministicDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    output = decoder.greedy_search("Hello")
    assert isinstance(output, str)

def test_greedy_search_is_deterministic():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = DeterministicDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    out1 = decoder.greedy_search("Hello")
    out2 = decoder.greedy_search("Hello")

    assert out1 == out2

def test_beam_search_output_type():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = DeterministicDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    output = decoder.beam_search("Hello", num_beams=2)
    assert isinstance(output, str)


def test_beam_search_runs_with_more_beams():
    model, tokenizer = _tiny_model_and_tokenizer()
    decoder = DeterministicDecoder(model, tokenizer, max_new_tokens=5, device=DEVICE) #type: ignore

    output = decoder.beam_search("Hi", num_beams=4)
    assert isinstance(output, str)