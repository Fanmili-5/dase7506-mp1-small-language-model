"""Stage44 training-only DS/MTP model with untied vocabulary output."""
import hashlib
from pathlib import Path

from torch import nn

import student_multi_token

PARENT_SHA256 = "92b3ff8bd3d457d9ec8c6f8a0a8fc5019908471a3dd1d0bc7d0c3bce945ff9b3"


class UntiedMultiTokenLM(student_multi_token.MultiTokenLM):
    def __init__(self, config):
        if config.get("tie_embeddings") is not False:
            raise ValueError("Untied training requires tie_embeddings=false")
        super().__init__(config)
        self.head = nn.Linear(int(config["width"]), self.vocab, bias=False)
        self.head.apply(self._initialize)


def build_model(config):
    actual = hashlib.sha256(Path(student_multi_token.__file__).read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        raise ValueError("Multi-token parent source changed")
    return UntiedMultiTokenLM(config)
