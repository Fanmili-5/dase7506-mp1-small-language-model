"""Stage105 fused predictor with a five-order count expert."""
from __future__ import annotations

import student_stage105_gated_singlepass


class Order5GatedLM(student_stage105_gated_singlepass.SinglePassGatedLM):
    def __init__(self, config: dict):
        if config.get("max_order") != 5 or len(config.get("order_shapes", ())) != 4:
            raise ValueError("Expected the pinned five-order count model")
        parent_config = dict(config, max_order=6)
        super().__init__(parent_config)
        self.ngram = student_stage105_gated_singlepass.SinglePassNgramLM(config)


def build_model(config: dict) -> Order5GatedLM:
    return Order5GatedLM(config)
