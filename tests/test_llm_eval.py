import torch

from src.evaluation.llm_eval import EvaluateLLM

evaluator = EvaluateLLM()


def test_perplexity_confident_correct_prediction_is_low():
    torch.manual_seed(0)
    B, T, C = 1, 3, 5
    labels = torch.tensor([[1, 2, 3]])

    logits = torch.full((B, T, C), -10.0)
    for t in range(T):
        logits[0, t, labels[0, t]] = 10.0  

    ppl = evaluator.perplexity(logits, labels)
    assert ppl < 1.1  


def test_perplexity_uniform_distribution_equals_vocab_size():
    B, T, C = 1, 4, 10
    labels = torch.randint(0, C, (B, T))
    logits = torch.zeros(B, T, C) 

    ppl = evaluator.perplexity(logits, labels)
    assert abs(ppl - C) < 0.5


def test_bleu_identical_sentences_scores_near_one():
    preds = "the cat sat on the mat"
    truth = "the cat sat on the mat"

    result = evaluator.BLEU(preds, truth)
    assert result["final_bleu_score"] > 0.99


def test_bleu_completely_different_sentences_scores_zero():
    preds = "quantum entanglement violates locality"
    truth = "the cat sat on the mat"

    result = evaluator.BLEU(preds, truth)
    assert result["final_bleu_score"] == 0.0


def test_rouge_recall_higher_when_pred_is_superset():
    preds = "the cat sat on the mat quietly today"
    truth = "the cat sat on the mat"

    result = evaluator.ROUGE(preds, truth)
    assert result["rouge-1"]["recall"] > 0.9
    assert result["rouge-1"]["precision"] < result["rouge-1"]["recall"]


def test_dpo_loss_lower_when_policy_prefers_chosen_more_than_reference():
    torch.manual_seed(42)
    B, T, C = 1, 4, 8
    chosen_labels = torch.randint(0, C, (B, T))
    rejected_labels = torch.randint(0, C, (B, T))

    reference_chosen_logits = torch.randn(B, T, C)
    reference_rejected_logits = torch.randn(B, T, C)

    policy_chosen_same = reference_chosen_logits.clone()
    policy_rejected_same = reference_rejected_logits.clone()

    policy_chosen_better = reference_chosen_logits.clone()

    for t in range(T - 1):
        policy_chosen_better[0, t, chosen_labels[0, t + 1]] += 5.0

    loss_same = evaluator.DPO(
        policy_chosen_same, reference_chosen_logits,
        policy_rejected_same, reference_rejected_logits,
        chosen_labels, rejected_labels,
    )
    loss_better = evaluator.DPO(
        policy_chosen_better, reference_chosen_logits,
        policy_rejected_same, reference_rejected_logits,
        chosen_labels, rejected_labels,
    )

    assert loss_better.item() < loss_same.item()