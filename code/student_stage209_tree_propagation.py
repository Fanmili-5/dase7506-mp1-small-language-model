"""Mathematically identical lexical hierarchy with level-wise propagation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

import student_stage208_lexical_hierarchy as parent


PARENT_SHA256 = "c582f803333b201614be3021f5844fd44d0315b2043448989a34c6cfc51e4279"


class PropagatedLexicalHierarchy(parent.LexicalHierarchy):
    def __init__(self, vocab: int, width: int):
        super().__init__(vocab, width)
        tokenizer = Tokenizer.from_file(str(parent.TOKENIZER_PATH))
        ordered = sorted(range(vocab), key=lambda token_id:
                         (tokenizer.id_to_token(token_id), token_id))
        inverse = torch.empty(vocab, dtype=torch.long)
        for lexical_index, token_id in enumerate(ordered):
            inverse[token_id] = lexical_index
        self.register_buffer("id_to_lexical", inverse, persistent=True)
        for depth in range(self.path_nodes.shape[1]):
            width_at_level = 1 << depth
            span = vocab // width_at_level
            positions = [ordered[index * span] for index in range(width_at_level)]
            node_ids = self.path_nodes[positions, depth].clone()
            if torch.unique(node_ids).numel() != width_at_level:
                raise ValueError("Invalid Stage208 tree level")
            self.register_buffer(f"level_nodes_{depth}", node_ids, persistent=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden.float(), self.node_weight, self.node_bias)
        partial = torch.zeros((*hidden.shape[:2], 1),
                              dtype=logits.dtype, device=logits.device)
        for depth in range(self.path_nodes.shape[1]):
            nodes = getattr(self, f"level_nodes_{depth}")
            decisions = logits.index_select(-1, nodes)
            left = partial + F.logsigmoid(decisions)
            right = partial + F.logsigmoid(-decisions)
            partial = torch.stack((left, right), dim=-1).flatten(-2)
        return partial.index_select(-1, self.id_to_lexical)


class TreePropagationHybridLM(parent.LexicalHierarchyHybridLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.lexical = PropagatedLexicalHierarchy(
            int(config["vocab"]), int(config["width"]))


def build_model(config: dict) -> TreePropagationHybridLM:
    if hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("Stage209 parent source changed")
    return TreePropagationHybridLM(config)
