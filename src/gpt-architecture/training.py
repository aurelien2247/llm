import torch

from utils import calc_loss_batch, calc_loss_loader, text_to_token_ids, token_ids_to_text
from generate import generate_text_simple


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Générer et afficher un exemple de texte du modèle.
    
    Args:
        model: Le modèle GPT (sera basculé en mode eval).
        tokenizer: Objet tokenizer avec méthode encode.
        device: Dispositif Torch (cpu ou cuda).
        start_context: Texte initial pour la génération.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Format compact
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter=5):
    """Évaluer le modèle sur les ensembles d'entraînement et de validation.
    
    Args:
        model: Le modèle GPT.
        train_loader: Dataloader d'entraînement.
        val_loader: Dataloader de validation.
        device: Dispositif Torch.
        eval_iter: Nombre de batches à évaluer.
    
    Returns:
        Tuple de (train_loss, val_loss).
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """Boucle d'entraînement simple pour le modèle GPT.
    
    Args:
        model: Le modèle GPT à entraîner.
        train_loader: Dataloader d'entraînement.
        val_loader: Dataloader de validation.
        optimizer: Optimiseur Torch (ex: AdamW).
        device: Dispositif Torch (cpu ou cuda).
        num_epochs: Nombre d'epochs d'entraînement.
        eval_freq: Évaluer tous les N steps.
        eval_iter: Nombre de batches par évaluation.
        start_context: Texte initial pour la génération d'exemples.
        tokenizer: Tokenizer pour l'encodage/décodage.
    
    Returns:
        Tuple de (train_losses, val_losses, track_tokens_seen).
    """
    # Initialiser les listes pour suivre les pertes et tokens vus
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Boucle d'entraînement principale
    for epoch in range(num_epochs):
        model.train()  # Mettre le modèle en mode entraînement

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Réinitialiser les gradients de la perte
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculer les gradients de la perte
            optimizer.step()  # Mettre à jour les poids du modèle
            tokens_seen += input_batch.numel()  # Nombre total de tokens
            global_step += 1

            # Étape d'évaluation optionnelle
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Perte train {train_loss:.3f}, Perte val {val_loss:.3f}")

        # Afficher un exemple de texte après chaque epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, savepath="loss-plot.pdf"):
    """Tracer les pertes d'entraînement et de validation.
    
    Args:
        epochs_seen: Tensor ou liste des numéros d'epochs.
        tokens_seen: Liste des comptes de tokens lors de l'enregistrement des pertes.
        train_losses: Liste des pertes d'entraînement.
        val_losses: Liste des pertes de validation.
        savepath: Chemin pour sauvegarder le graphique PDF.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        raise ImportError("matplotlib est requis pour tracer les pertes. Installez-le avec: pip install matplotlib")

    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Tracer les pertes d'entraînement et de validation par epochs
    ax1.plot(epochs_seen, train_losses, label="Perte d'entraînement")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Perte de validation")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Perte")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Afficher uniquement les labels entiers sur l'axe x

    # Créer un deuxième axe x pour les tokens vus
    ax2 = ax1.twiny()  # Créer un deuxième axe x qui partage le même axe y
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Graphique invisible pour aligner les ticks
    ax2.set_xlabel("Tokens vus")

    fig.tight_layout()  # Ajuster la disposition pour faire de la place
    plt.savefig(savepath)
    print(f"Graphique sauvegardé dans {savepath}")
    plt.show()


__all__ = ["train_model_simple", "evaluate_model", "generate_and_print_sample", "plot_losses"]
