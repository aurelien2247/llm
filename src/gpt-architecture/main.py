import torch
import tiktoken

from data import create_dataloader_v1
from model import GPTModel
from generate import generate_text_simple


def main():
    # Configuration du modèle GPT-124M
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # Texte d'entrée pour la génération
    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}ENTRÉE\n{50*'='}")
    print("\nTexte d'entrée:", start_context)
    print("Texte d'entrée encodé:", encoded)
    print("Forme du tenseur encodé:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}SORTIE\n{50*'='}")
    print("\nSortie:", out)
    print("Longueur de la sortie:", len(out[0]))
    print("Texte de sortie:", decoded_text)


if __name__ == "__main__":
    main()
