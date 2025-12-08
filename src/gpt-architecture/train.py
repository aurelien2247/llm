import time
import torch
import tiktoken

from model import GPTModel
from data import create_dataloader_v1
from training import train_model_simple, plot_losses


def main():
    # Configuration
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Configuration du dispositif
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du dispositif: {device}")

    # Charger le texte
    print("Chargement des données textuelles...")
    try:
        with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("ERREUR: data/the-verdict.txt introuvable")
        return

    print(f"Texte chargé: {len(raw_text)} caractères")

    # Diviser en entraînement et validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    print(f"Données train: {len(train_data)} caractères | Données val: {len(val_data)} caractères")

    # Créer les dataloaders
    print("Création des dataloaders...")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        tokenizer=tokenizer
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=False,
        drop_last=False,
        tokenizer=tokenizer
    )

    print(f"Dataloader train: {len(train_loader)} batches | Dataloader val: {len(val_loader)} batches")

    # Initialiser le modèle et l'optimiseur
    print("\nInitialisation du modèle...")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Entraîner
    print("Démarrage de l'entraînement...\n")
    start_time = time.time()

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nEntraînement terminé en {execution_time_minutes:.2f} minutes.")

    # Sauvegarder le checkpoint du modèle
    print("\nSauvegarde du checkpoint du modèle...")
    checkpoint_path = "gpt-model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Modèle sauvegardé dans {checkpoint_path}")

    # Tracer les pertes (optionnel, nécessite matplotlib)
    try:
        print("\nTraçage des pertes...")
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, savepath="loss-plot.pdf")
    except ImportError:
        print("matplotlib n'est pas installé; skip du graphique des pertes")
        print("Pour installer: pip install matplotlib")


if __name__ == "__main__":
    main()
