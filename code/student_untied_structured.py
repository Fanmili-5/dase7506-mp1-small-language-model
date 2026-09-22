"""Prefix-copy Transformer with independent input and output embeddings."""
import hashlib
from pathlib import Path

from torch import nn

import student_structured

PARENT_SHA256 = "6bc2e61a2a53e25416bba3818b8bc41af72c1bef5221244306027ea7bf3faa18"


class UntiedStructuredLM(student_structured.StructuredLM):
    def __init__(self, config):
        if config.get("tie_embeddings") is not False:
            raise ValueError("Untied model requires tie_embeddings=false")
        super().__init__(config)
        self.head = nn.Linear(int(config["width"]), self.vocab, bias=False)
        self.head.apply(self._initialize)


def build_model(config):
    actual = hashlib.sha256(Path(student_structured.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Structured parent source changed")
    return UntiedStructuredLM(config)
