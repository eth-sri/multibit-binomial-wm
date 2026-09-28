from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from typing import List
from fast_bleu import SelfBLEU


def compute_self_bleu(
    generated_sequences: List[List[int]],
    weights: dict = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
) -> dict:
    """Compute Self-BLEU scores for the generated sequences.

    Args:
        generated_sequences (List[List[int]]): A list of generated sequences, where each sequence is a list of token IDs.
        weights (dict): A dictionary specifying the n-gram weights for BLEU computation.

    Returns:
        dict: A dictionary containing Self-BLEU scores for each n-gram specified in weights.
    """
    self_bleu = SelfBLEU(generated_sequences, weights=weights)
    scores = self_bleu.get_score()
    
    return scores


def compute_perplexity(
    perplexity_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[float]:   
    """Compute the perplexity for each sample in the batch.

    Args:
        perplexity_model (AutoModelForCausalLM): The language model to use.
        input_ids (torch.Tensor): Batch of input token IDs. Shape: [batch_size, sequence_length]
        attention_mask (torch.Tensor): Batch of attention masks. Shape: [batch_size, sequence_length]

    Returns:
        List[float]: A list of perplexity values, one for each sample in the batch.
    """
    with torch.no_grad():
        outputs = perplexity_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        
        shift_labels[shift_attention_mask == 0] = -100

        batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        loss = loss.view(batch_size, seq_len_minus_1)
        
        sum_loss_per_sequence = loss.sum(dim=1)
        num_valid_tokens_per_sequence = (shift_labels != -100).sum(dim=1).float()
        
        avg_loss_per_sequence = torch.zeros_like(sum_loss_per_sequence, dtype=torch.float)
        valid_mask = num_valid_tokens_per_sequence > 0
        
        avg_loss_per_sequence[valid_mask] = sum_loss_per_sequence[valid_mask] / num_valid_tokens_per_sequence[valid_mask]

        perplexities = torch.exp(avg_loss_per_sequence)

        perplexities[~valid_mask] = 0.0

    return perplexities.detach().cpu().tolist()