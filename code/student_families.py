"""Independent-window LSTM and dilated gated-convolution comparison families.

These are small custom baselines, not reproductions of AWD-LSTM or GCNN papers.
No attention, copied Transformer blocks, external weights, or persistent state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalConvBlock(nn.Module):
    def __init__(self, width, kernel, dilation, dropout, depth):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, 2 * width, kernel, dilation=dilation)
        self.output = nn.Linear(width, width)
        self.left = (kernel - 1) * dilation
        self.dropout = dropout
        nn.init.normal_(self.output.weight, std=0.02 / math.sqrt(2 * depth))
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        z = F.glu(self.conv(F.pad(z, (self.left, 0))), dim=1).transpose(1, 2)
        return x + F.dropout(self.output(z), self.dropout, self.training)


class FamilyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context, self.vocab = int(config["context"]), int(config["vocab"])
        if (self.context, self.vocab) != (256, 2048):
            raise ValueError("Fixed context/vocabulary required")
        self.family = config["family"]
        width, depth = int(config["width"]), int(config["depth"])
        self.dropout = float(config.get("dropout", 0.1))
        self.token = nn.Embedding(self.vocab, width)
        nn.init.normal_(self.token.weight, std=0.02)
        if self.family == "lstm":
            hidden = int(config["hidden"])
            self.recurrent = nn.LSTM(width, hidden, depth, batch_first=True,
                                     dropout=self.dropout if depth > 1 else 0.0)
            self.project = nn.Linear(hidden, width, bias=False)
            for name, parameter in self.recurrent.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(parameter)
                    if "bias_ih" in name:
                        with torch.no_grad():
                            parameter[hidden:2 * hidden].fill_(1)
        elif self.family == "gated_conv":
            dilations = list(config["dilations"])
            if len(dilations) != depth or min(dilations) < 1:
                raise ValueError("One positive dilation per block required")
            self.blocks = nn.ModuleList([CausalConvBlock(width, int(config["kernel"]),
                d, self.dropout, depth) for d in dilations])
        else:
            raise ValueError("Unknown family")
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, self.vocab, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("Expected nonempty independent windows <=256")
        x = F.dropout(self.token(ids), self.dropout, self.training)
        if self.family == "lstm":
            # Omitted h/c explicitly means fresh zero states for every call/row.
            x, _ = self.recurrent(x)
            x = self.project(x)
        else:
            for block in self.blocks:
                x = block(x)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self.head(F.dropout(self.norm(x.float()), self.dropout, self.training))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), -1)


def build_model(config):
    return FamilyLM(config)
