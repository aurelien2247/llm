"""Utilitaires pour le modèle GPT."""

import torch
import tiktoken
from typing import Optional


def text_to_token_ids(text: str, tokenizer=None) -> torch.Tensor:
    """Encoder le texte en tenseur de token ids groupé (1 x T). Si tokenizer est None,
    utilise par défaut `tiktoken.get_encoding('gpt2')`."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer=None) -> str:
    """Décoder un tenseur groupé (1 x T) ou 1D de token ids en texte."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    """Calculer la perte d'entropie croisée pour un batch. Retourne un scalaire sur CPU."""
    if device is None:
        device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model: torch.nn.Module, device: Optional[torch.device] = None, num_batches: Optional[int] = None) -> float:
    """Calculer la perte moyenne sur un dataloader.
    
    Args:
        data_loader: DataLoader avec les données d'entraînement/validation.
        model: Le modèle GPT.
        device: Dispositif Torch (cpu ou cuda).
        num_batches: Nombre de batches à évaluer. Si None, évaluer tous.
    
    Returns:
        Perte moyenne.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


__all__ = ["text_to_token_ids", "token_ids_to_text", "calc_loss_batch", "calc_loss_loader"]
