"""
Script principal d'entraînement du modèle GPT.

Ce script:
1. Charge les données textuelles
2. Crée les dataloaders train/val
3. Entraîne le modèle GPT-124M
4. Sauvegarde le checkpoint
5. Trace les courbes de perte (optional, nécessite matplotlib)

Utilisation:
    python3 train.py
"""

import sys
import time
import torch
import tiktoken

# Importer depuis les modules locaux (chemin relatif)
sys.path.insert(0, '..')
from config import GPT_CONFIG_124M, TRAINING_CONFIG, PATHS
from core import GPTModel
from data import create_dataloader_v1
from training import train_model_simple, plot_losses


def main():
    """Fonction principale d'entraînement."""
    # Configuration du dispositif
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du dispositif: {device}")

    # Charger le texte
    print("Chargement des données textuelles...")
    try:
        with open(f"../{PATHS['data_file']}", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"ERREUR: {PATHS['data_file']} introuvable")
        return

    print(f"Texte chargé: {len(raw_text)} caractères")

    # Diviser en entraînement et validation
    train_ratio = PATHS['train_split']
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    print(f"Données train: {len(train_data)} caractères | Données val: {len(val_data)} caractères")

    # Créer les dataloaders
    print("Création des dataloaders...")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=TRAINING_CONFIG['batch_size'],
        max_length=TRAINING_CONFIG['max_length'],
        stride=TRAINING_CONFIG['stride'],
        shuffle=True,
        drop_last=True,
        tokenizer=tokenizer
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=TRAINING_CONFIG['batch_size'],
        max_length=TRAINING_CONFIG['max_length'],
        stride=TRAINING_CONFIG['stride'],
        shuffle=False,
        drop_last=False,
        tokenizer=tokenizer
    )

    print(f"Dataloader train: {len(train_loader)} batches | Dataloader val: {len(val_loader)} batches")

    # Initialiser le modèle et l'optimiseur
    print("\nInitialisation du modèle...")
    torch.manual_seed(TRAINING_CONFIG['seed'])
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # Entraîner
    print("Démarrage de l'entraînement...\n")
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        eval_freq=TRAINING_CONFIG['eval_freq'],
        eval_iter=TRAINING_CONFIG['eval_iter'],
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nEntraînement terminé en {execution_time_minutes:.2f} minutes.")

    # Sauvegarder le checkpoint du modèle
    print("\nSauvegarde du checkpoint du modèle...")
    checkpoint_path = PATHS['model_checkpoint']
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Modèle sauvegardé dans {checkpoint_path}")

    # Tracer les pertes (optionnel, nécessite matplotlib)
    try:
        print("\nTraçage des pertes...")
        epochs_tensor = torch.linspace(0, TRAINING_CONFIG['num_epochs'], len(train_losses))
        plot_losses(
            epochs_tensor, tokens_seen, train_losses, val_losses,
            savepath=PATHS['loss_plot']
        )
    except ImportError:
        print("matplotlib n'est pas installé; skip du graphique des pertes")
        print("Pour installer: pip install matplotlib")


if __name__ == "__main__":
    main()
