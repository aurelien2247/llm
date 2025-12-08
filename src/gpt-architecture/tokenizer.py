import re
from typing import Dict, List


class SimpleTokenizerV1:
    """Tokenizer simple basé sur un vocabulaire personnalisé.
    Utilise des expressions régulières pour diviser le texte sur la ponctuation et les espaces.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encoder le texte en liste d'identifiants de jetons."""
        preprocessed = re.split(r'([,.:;?_!"()\'\'']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Décoder une liste d'identifiants de jetons en texte."""
        text = " ".join([self.int_to_str[i] for i in ids])
        # Supprimer les espaces avant la ponctuation spécifiée
        text = re.sub(r'\s+([,.:;?!)'\'\"])', r"\1", text)
        return text


class SimpleTokenizerV2(SimpleTokenizerV1):
    """Version améliorée du tokenizer avec support pour les jetons inconnus.
    Remplace les jetons hors-vocabulaire par un jeton <|unk|>.
    """
    def __init__(self, vocab: Dict[str, int], unk_token: str = "<|unk|>"):
        super().__init__(vocab)
        self.unk = unk_token

    def encode(self, text: str) -> List[int]:
        """Encoder le texte en liste d'identifiants, en remplaçant les mots inconnus par <|unk|>."""
        preprocessed = re.split(r'([,.:;?_!"()\'\'']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else self.unk for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids


__all__ = ["SimpleTokenizerV1", "SimpleTokenizerV2"]
