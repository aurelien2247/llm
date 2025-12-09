"""
Messages réutilisables pour les scripts de `scripts/`.
Centralise les messages imprimés pour éviter les duplications.
"""

MODEL_LOADED = "✓ Modèle chargé avec succès\n"

def model_loaded_from(path: str) -> str:
    return f"✓ Modèle chargé depuis {path}\n"

def error_not_found(path: str) -> str:
    return f"✗ ERREUR: {path} non trouvé"

MISSING_CHECKPOINT_HELP = "  Veuillez d'abord entraîner le modèle avec: python3 train.py"

def graph_saved(filename: str) -> str:
    return f"✓ Graphique sauvegardé dans '{filename}'\n"

def header(title: str, width: int = 80) -> None:
    sep = "=" * width
    print(sep)
    print(title)
    print(sep + "\n")
