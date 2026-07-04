'''
LLM eval (ROUGE, BLEU, BERTScore, Perplexity, DPO)
'''

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

class EvaluateLLM:
    
    def perplexity(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        '''
        INTUITION:
        Perplexity measure how confuse model,
        for example if we get perplexity = 10.0
        it's mean model at least confuse how exactly word/token for the next token
        in 10 choice, because if model confident to predict next token then prob
        that token become higher and if probs token higher log(probs around ~1.0) loss lower around ~0.0
        if model given 10 choice it's mean each token have similiar probs and probs real next token
        become lower and got loss higher
        '''
        assert logits.ndim == 3
        assert labels.ndim == 2
        
        B, T, C = logits.shape
        
        loss = F.cross_entropy(logits.view(-1, C), labels.view(-1), ignore_index=-100)
        ppl = torch.exp(loss)
        return float(ppl)

    def BLEU(self, preds: str, truth: str, n_grams=None):
        '''Just accepted one sample. re-call if need many samples.'''
        '''
        INTUITION:
        BLEU (Bilingual Evaluation Understudy)
        Understudy it's mean change human role in evaluation become ground truth text
        BLEU matching each word whether this token prediction exist in the ground truth
        for example
        preds = ["I", "Really", "Love", "You"]
        truth = ["I", "Love", "You"]
        preds guessing all the words that exists in the ground truth
        because BLEU is precision-oriented then measure its score use precision approach
        Precision = (Num word matches with ground truth / Num word in generation predictions)
        Precision = (3 / 4)
        it's mean how good model capture or predict token that exist in ground truth
        why we must divide with num word generations? it's prevent model adding useless token and wordy
        But what if model spam word that matches with one word in ground truth at least one token
        then modified precision formula exists,
        we calculate how exactly freq that words exist in ground truth, we restrict the preds token generation
        are limited only to the total number present token in ground truth
        
        then Brevity Penalty for a stingy-worded model
        '''

        chunk_preds = preds.split()
        chunk_truth = truth.split()
        
        if len(chunk_preds) == 0 or len(chunk_truth) == 0:
            return {"bleu_score": 0.0}
        
        if n_grams is None:
            n_grams = [1, 2, 3, 4] # use all n_grams
        
        p_n_scores = []
        scores_report = {}
        
        for n in n_grams:
            truth_ngrams = [tuple(chunk_truth[i:i+n]) for i in range(len(chunk_truth) - n + 1)]
            truth_counts = Counter(truth_ngrams)
            
            pred_ngrams = [tuple(chunk_preds[i:i+n]) for i in range(len(chunk_preds) - n + 1)]
            pred_counts = Counter(pred_ngrams)
            
            total_pred_ngrams = len(pred_ngrams)
            if total_pred_ngrams == 0:
                p_n_scores.append(0.0)
                scores_report[f"{n}-gram_precision"] = 0.0
                continue
            
            clipped_match = 0
            for ngram, count in pred_counts.items():
                if ngram in truth_counts:
                    clipped_match += min(count, truth_counts[ngram])
            
            
            precision = clipped_match / total_pred_ngrams
            p_n_scores.append(precision)
            scores_report[f"{n}-gram_precision"] = precision
        
        import math
        if min(p_n_scores) == 0:
            geometric_mean = 0.0
        else:
            weight = 1.0 / len(n_grams)
            geometric_mean = math.exp(sum(weight * math.log(p) for p in p_n_scores))
        
        c = len(chunk_preds)
        r = len(chunk_truth)
        
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - (r / c))
        
        score_bleu = bp * geometric_mean
        scores_report["brevity_penalty"] = bp
        scores_report["final_bleu_score"] = score_bleu
        
        return scores_report
        
        # dictionary = {}
        # for n_gram in n_grams: #type: ignore
        #     dictionary[n_gram] = []
        #     for i in range(0, len(chunk_truth)):
        #         if i + n_gram <= len(chunk_truth):
        #             dictionary[n_gram].append(chunk_truth[i:i+n_gram])
        
        # scores = {}
        # for n_gram in n_grams:
        #     match = 0
        #     for i in range(0, len(chunk_preds)):
        #         if i + n_gram <= len(chunk_preds):
        #             if chunk_preds[i:i+n_gram] in dictionary[n_gram]:
        #                 match += 1
            
        #     result = float(match / (len(chunk_preds) + 1e-8))
        #     scores[f"n_gram={n_gram}"] = result
        
        # return scores
    
    def _calc_lcs(self, X: list, Y: list):
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1): # 2
            for j in range(1, n + 1): # 2
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    
    def _calc_ngram_rouge(self, pred_tokens: list, truth_tokens: list, n: int):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        truth_ngrams = [tuple(truth_tokens[i:i+n]) for i in range(len(truth_tokens) - n + 1)] 
        
        total_pred = len(pred_ngrams)
        total_truth = len(truth_ngrams)
        
        if total_pred == 0 or total_truth == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        pred_counts = Counter(pred_ngrams)
        truth_counts = Counter(truth_ngrams)
        
        overlap = 0
        for ngram, count in pred_counts.items():
            if ngram in truth_counts:
                overlap += min(count, truth_counts[ngram])
        
        recall = overlap / total_truth
        precision = overlap / total_pred
        
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return {"precision": precision, "recall": recall, "f1_score": f1}
    
    def ROUGE(self, preds: str, truth: str):
        '''
        INTUITION:
        ROUGE (Recall-Oriented Understudy Gisting Evaluation)
        same as BLEU but ROUGE provided the recall which is how good model capture
        all of word in the ground truth
        '''
        pred_tokens = preds.split()
        truth_tokens = truth.split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            empty_res = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            return {"rouge-1": empty_res, "rouge-2": empty_res, "rouge-l": empty_res}

        rouge1 = self._calc_ngram_rouge(pred_tokens, truth_tokens, n=1)
        rouge2 = self._calc_ngram_rouge(pred_tokens, truth_tokens, n=2)
        
        lcs_length = self._calc_lcs(pred_tokens, truth_tokens)
        
        r_lcs = lcs_length / len(truth_tokens)
        p_lcs = lcs_length / len(pred_tokens)
        
        if (p_lcs + r_lcs) == 0:
            f1_lcs = 0.0
        else:
            f1_lcs = 2 * (p_lcs * r_lcs) / (p_lcs + r_lcs)
            
        rougel = {"precision": p_lcs, "recall": r_lcs, "f1_score": f1_lcs}
        
        return {
            "rouge-1": rouge1,
            "rouge-2": rouge2,
            "rouge-l": rougel
        }
    
    def _get_log_softmax(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index=-100,
    ):
        '''
        Calculate total log-probability per-sentence
        '''
        # Shifting in logits approach, but we can shifting before put in x and y to model
        logits = logits[:, :-1, :]
        labels = labels[:, 1:].clone()
        
        loss_mask = (labels != ignore_index)
        
        labels[labels == ignore_index] = 0
        
        per_token_logits = F.log_softmax(logits, dim=-1)
        per_token_logs = torch.gather(per_token_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        per_token_logs *= loss_mask
        
        return per_token_logs.sum(-1) # dealing with sentence so sum for each sentence
    
    def DPO(
        self,
        policy_chosen_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        beta: float = 0.1
    ):
        '''Bradley-Terry equation'''
        # pi_chosen_logits = self._get_log_softmax(policy_chosen_logits, chosen_labels)
        # pi_rejected_logits = self._get_log_softmax(policy_rejected_logits, rejected_labels)
        
        with torch.no_grad():
            pi_chosen_logits = self._get_log_softmax(policy_chosen_logits.detach(), chosen_labels)
            pi_rejected_logits = self._get_log_softmax(policy_rejected_logits.detach(), rejected_labels)
            ref_chosen_logits = self._get_log_softmax(reference_chosen_logits.detach(), chosen_labels)
            ref_rejected_logits = self._get_log_softmax(reference_rejected_logits.detach(), rejected_labels)
        
        pi_logratios = pi_chosen_logits - pi_rejected_logits
        ref_logratios = ref_chosen_logits - ref_rejected_logits
        
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits)
        
        return loss.mean()