"""Configuration centralisée du projet GPT."""

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

# Configuration de l'entraînement
TRAINING_CONFIG = {
    "batch_size": 4,
    "max_length": 256,
    "stride": 128,
    "num_epochs": 10,
    "eval_freq": 5,
    "eval_iter": 5,
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
    "seed": 123,
}

# Configuration des chemins
PATHS = {
    "model_checkpoint": "gpt-model.pt",
    "loss_plot": "loss-plot.pdf",
    "data_file": "data/the-verdict.txt",
    "train_split": 0.90,
}

# Configuration pour la génération
GENERATION_CONFIG = {
    "max_tokens_default": 50,
    "default_strategy": "greedy",
    "default_temperature": 1.0,
    "default_k": 50,
    "default_p": 0.9,
}

__all__ = ["GPT_CONFIG_124M", "TRAINING_CONFIG", "PATHS", "GENERATION_CONFIG"]
