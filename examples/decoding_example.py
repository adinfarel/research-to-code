'''
Example for see what happen if use different approach decoding
'''

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from src.decoding.model_loader import load_model
from src.decoding.deterministic import DeterministicDecoder
from src.decoding.stochastic import StochasticDecoder


def run_comparison(prompts: list[str], model_name: str = "meta-llama/Llama-3.2-1B",
                    device: str = "cpu", max_new_tokens: int = 30):
    model, tokenizer = load_model(model_name, device=device)

    det = DeterministicDecoder(model, tokenizer, max_new_tokens=max_new_tokens, device=device) #type: ignore
    stoch = StochasticDecoder(model, tokenizer, max_new_tokens=max_new_tokens, device=device) #type: ignore

    results = []
    for prompt in prompts:
        row = {"prompt": prompt}
        row["greedy"] = det.greedy_search(prompt)
        row["beam_search (n=3)"] = det.beam_search(prompt, num_beams=3)
        row["temperature=0.7"] = stoch.temperature_sampling(prompt, temperature=0.7)
        row["temperature=1.3"] = stoch.temperature_sampling(prompt, temperature=1.3)
        row["top_k=50"] = stoch.top_k_sampling(prompt, k=50)
        row["top_p=0.9"] = stoch.top_p_sampling(prompt, p=0.9)
        results.append(row)

    return results


def print_summary(results: list[dict]):
    for row in results:
        print("=" * 80)
        print(f"PROMPT: {row['prompt']}")
        print("-" * 80)
        for strategy, output in row.items():
            if strategy == "prompt":
                continue
            print(f"[{strategy}]\n  {output}\n")


if __name__ == "__main__":
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a small village,",
    ]

    results = run_comparison(prompts, device=DEVICE, max_new_tokens=30)
    print_summary(results)