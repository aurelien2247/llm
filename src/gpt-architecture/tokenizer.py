import re
from typing import Dict, List


class SimpleTokenizerV1:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.:;?_!"()\'\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!)\'\"])', r"\1", text)
        return text


class SimpleTokenizerV2(SimpleTokenizerV1):
    def __init__(self, vocab: Dict[str, int], unk_token: str = "<|unk|>"):
        super().__init__(vocab)
        self.unk = unk_token

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.:;?_!"()\'\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else self.unk for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids


__all__ = ["SimpleTokenizerV1", "SimpleTokenizerV2"]
