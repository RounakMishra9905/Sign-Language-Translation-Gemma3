"""
Metrics for Sign Language Translation
"""

import numpy as np
from typing import List, Dict
from collections import Counter
import math


def compute_bleu(references: List[str], hypotheses: List[str], max_order: int = 4) -> Dict[str, float]:
    """
    Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
        max_order: Maximum n-gram order
    
    Returns:
        Dict with BLEU scores
    """
    
    def _get_ngrams(segment, max_n):
        """Extract n-grams from segment"""
        ngram_counts = Counter()
        for order in range(1, max_n + 1):
            for i in range(len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts
    
    # Tokenize
    ref_tokens = [ref.split() for ref in references]
    hyp_tokens = [hyp.split() for hyp in hypotheses]
    
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    
    for ref, hyp in zip(ref_tokens, hyp_tokens):
        reference_length += len(ref)
        translation_length += len(hyp)
        
        ref_ngrams = _get_ngrams(ref, max_order)
        hyp_ngrams = _get_ngrams(hyp, max_order)
        
        overlap = hyp_ngrams & ref_ngrams
        
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        
        for order in range(1, max_order + 1):
            possible_matches = len(hyp) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches
    
    # Compute precisions
    precisions = []
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precision = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precision = 0.0
        precisions.append(precision)
    
    # Compute BLEU scores
    bleu_scores = {}
    for n in range(1, max_order + 1):
        if min(precisions[:n]) > 0:
            log_precision_sum = sum(math.log(p) for p in precisions[:n]) / n
            geo_mean = math.exp(log_precision_sum)
        else:
            geo_mean = 0.0
        
        # Brevity penalty
        ratio = translation_length / reference_length if reference_length > 0 else 0
        if ratio > 1.0:
            bp = 1.0
        else:
            bp = math.exp(1 - 1.0 / ratio) if ratio > 0 else 0.0
        
        bleu_scores[f'bleu{n}'] = geo_mean * bp
    
    return bleu_scores


def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
    
    Returns:
        Dict with ROUGE scores
    """
    
    def _lcs(x, y):
        """Longest common subsequence"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_l_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        
        if len(hyp_tokens) == 0:
            rouge_l_scores.append(0.0)
            continue
        
        lcs_length = _lcs(ref_tokens, hyp_tokens)
        
        # Precision and recall
        precision = lcs_length / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        rouge_l_scores.append(f1)
    
    return {
        'rouge_l': np.mean(rouge_l_scores) if rouge_l_scores else 0.0
    }


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER)
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
    
    Returns:
        WER score
    """
    
    def _edit_distance(ref, hyp):
        """Compute edit distance"""
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    total_errors = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        
        errors = _edit_distance(ref_tokens, hyp_tokens)
        total_errors += errors
        total_words += len(ref_tokens)
    
    wer = total_errors / total_words if total_words > 0 else 0.0
    return wer


def compute_all_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute all metrics at once
    
    Args:
        references: List of reference sentences
        hypotheses: List of hypothesis sentences
    
    Returns:
        Dict with all metrics
    """
    bleu_scores = compute_bleu(references, hypotheses)
    rouge_scores = compute_rouge(references, hypotheses)
    wer_score = compute_wer(references, hypotheses)
    
    return {
        **bleu_scores,
        **rouge_scores,
        'wer': wer_score
    }


# Test
if __name__ == "__main__":
    refs = [
        "the cat is on the mat",
        "there is a cat on the mat"
    ]
    hyps = [
        "the cat is on the mat",
        "there is a dog on the mat"
    ]
    
    metrics = compute_all_metrics(refs, hyps)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
