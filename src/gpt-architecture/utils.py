import torch
import tiktoken
from typing import Optional


def text_to_token_ids(text: str, tokenizer=None) -> torch.Tensor:
    """Encode text to a batched tensor of token ids (1 x T). If tokenizer is None,
    defaults to `tiktoken.get_encoding('gpt2')`."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer=None) -> str:
    """Decode a batched tensor (1 x T) or 1D tensor of token ids back into text."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute cross entropy loss for a single batch. Returns a scalar tensor on CPU."""
    if device is None:
        device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model: torch.nn.Module, device: Optional[torch.device] = None, num_batches: Optional[int] = None) -> float:
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
