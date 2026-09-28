"""Joint causal hybrid LM with a normalized lexical-tree output branch.

The tree is a deterministic property of the supplied fixed tokenizer.  It
never reads benchmark text or target tokens when making a prediction.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer

import student_hybrid_conv_rdrop


PARENT_SHA256 = "2185db04f5f1ec44ad68005e8f7482e11114b3528528572c4497f6e9c07cbfe1"
TOKENIZER_SHA256 = "020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e"
TOKENIZER_PATH = Path(__file__).resolve().parent / "data/tokenizer.json"


def lexical_tree_paths(vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced lexical partition of the unchanged BPE vocabulary."""
    if vocab != 2048 or vocab & (vocab - 1):
        raise ValueError("Stage208 requires the fixed 2,048-token vocabulary")
    if hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Changed supplied tokenizer")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    ordered = sorted(range(vocab), key=lambda token_id: (tokenizer.id_to_token(token_id), token_id))
    depth = int(math.log2(vocab))
    nodes = torch.empty((vocab, depth), dtype=torch.long)
    signs = torch.empty((vocab, depth), dtype=torch.float32)
    next_node = 0

    def split(leaves: list[int], level: int) -> None:
        nonlocal next_node
        if len(leaves) == 1:
            if level != depth:
                raise ValueError("Lexical tree is unbalanced")
            return
        node = next_node
        next_node += 1
        half = len(leaves) // 2
        for token_id in leaves[:half]:
            nodes[token_id, level] = node
            signs[token_id, level] = 1.0
        for token_id in leaves[half:]:
            nodes[token_id, level] = node
            signs[token_id, level] = -1.0
        split(leaves[:half], level + 1)
        split(leaves[half:], level + 1)

    split(ordered, 0)
    if next_node != vocab - 1:
        raise ValueError("Incomplete lexical tree")
    return nodes, signs


class LexicalHierarchy(nn.Module):
    def __init__(self, vocab: int, width: int):
        super().__init__()
        nodes, signs = lexical_tree_paths(vocab)
        self.register_buffer("path_nodes", nodes, persistent=True)
        self.register_buffer("path_signs", signs, persistent=True)
        self.node_weight = nn.Parameter(torch.zeros(vocab - 1, width))
        self.node_bias = nn.Parameter(torch.zeros(vocab - 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden.float(), self.node_weight, self.node_bias)
        result = torch.zeros((*hidden.shape[:2], self.path_nodes.shape[0]),
                             dtype=logits.dtype, device=logits.device)
        for depth in range(self.path_nodes.shape[1]):
            nodes = self.path_nodes[:, depth]
            signs = self.path_signs[:, depth]
            result = result + F.logsigmoid(logits.index_select(-1, nodes) * signs)
        return result


class LexicalHierarchyHybridLM(student_hybrid_conv_rdrop.HybridConvRDropLM):
    def __init__(self, config: dict):
        super().__init__(config)
        width = int(config["width"])
        self.lexical = LexicalHierarchy(int(config["vocab"]), width)
        self.lexical_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.lexical_gate.weight)
        nn.init.constant_(self.lexical_gate.bias, -math.log(4.0))

    def _prefix_log_probs(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        ordinary = F.log_softmax(self.head(hidden), dim=-1)
        if self.training and hidden.requires_grad:
            lexical = checkpoint(self.lexical, hidden, use_reentrant=False)
        else:
            lexical = self.lexical(hidden)
        gate = self.lexical_gate(hidden)
        vocabulary = torch.logaddexp(F.logsigmoid(-gate) + ordinary,
                                     F.logsigmoid(gate) + lexical)
        copy = self.copy_distribution(hidden, ids)
        log_copy = copy.clamp_min(torch.finfo(copy.dtype).tiny).log()
        log_copy = log_copy.masked_fill(copy == 0, float("-inf"))
        copy_gate = self.copy_gate(hidden)
        return torch.logaddexp(F.logsigmoid(-copy_gate) + vocabulary,
                               F.logsigmoid(copy_gate) + log_copy)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.features(ids)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self._prefix_log_probs(hidden.float(), ids)


def build_model(config: dict) -> LexicalHierarchyHybridLM:
    parent = Path(student_hybrid_conv_rdrop.__file__)
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("Stage208 hybrid parent source changed")
    return LexicalHierarchyHybridLM(config)
