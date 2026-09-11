"""Reconstruct v3 deployment from raw tokenized inputs, not cached embeddings."""
from __future__ import annotations
import json
import torch
from torch import nn
from common import ROOT, sha256
from model import Selector
from prepare import START
from e26_adaptation_pilot import load_base

class Deployed(nn.Module):
    def __init__(self, run, device="cpu"):
        super().__init__()
        self.meta=json.loads((run/"provenance.json").read_text())
        ck=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
        if sha256(START)!=self.meta["inputs_sha256"][str(START.relative_to(ROOT))]:
            raise ValueError("Starting checkpoint changed")
        self.base=load_base(device)
        self.arm=self.meta["args"]["arm"]
        self.original=ck["best_step"]==0
        self.selector=None
        self.h=None
        self.base.head.register_forward_pre_hook(self.capture)
        if not self.original:
            state=ck["state"]
            if state["base"] is not None:
                self.base.load_state_dict(state["base"])
            if state["selector"] is not None:
                dim=self.base.head[0].normalized_shape[0]
                self.selector=Selector(dim,residual=self.arm in ("E","F","G","F_replay")).to(device)
                self.selector.load_state_dict(state["selector"])
        self.eval()

    def capture(self,module,x):
        self.h=x[0]

    def forward(self,batch):
        out=self.base(batch)
        pred=self.base.efficiency_from_output(out)
        q=(pred-self.meta["q_mean"])/self.meta["q_sd"]
        # Only frozen-backbone arms use a residual, so q is the frozen source score.
        s=q if self.selector is None else self.selector(self.h,q)
        return s
