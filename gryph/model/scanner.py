from collections import ChainMap

import torch.nn as nn
from mmengine.model import BaseModel

from gryph.utils import MODELS, build


@MODELS.register_module()
class FormulaScanner(BaseModel):
    def __init__(
        self,
        batcher: dict,
        encoder: dict,
        decoder: dict,
        handler: dict,
        losses: list,
    ):
        super().__init__()

        # init modules
        batcher = build(batcher)
        encoder = build(encoder)
        decoder = build(decoder, batcher=batcher)
        handler = build(handler, batcher=batcher)

        # core modules
        self.encoder = encoder
        self.decoder = decoder
        self.handler = handler

        # loss modules
        self.losses = nn.ModuleList(map(build, losses))

    def init_weights(self):
        pass

    def forward(self, mode: str, **kwargs):
        if mode == "loss":
            return self._train(**kwargs)
        else:
            return self._valid(**kwargs)

    def _train(self, **targets):
        targets = self.handler(**targets, train=True)
        outputs = self.encoder(**targets, train=True)

        # invoke decoder
        output1 = self.decoder(**outputs, train=True)
        output2 = self.decoder(**outputs, train=True)

        # rename outputs
        output1 = self.handler.rename(output1, suffix="1")
        output2 = self.handler.rename(output2, suffix="2")

        # rename targets
        target1 = self.handler.rename(targets, suffix="1")
        target2 = self.handler.rename(targets, suffix="2")

        # merge
        outputs = dict(output1, **output2)
        targets = dict(target1, **target2)

        return self.loss(outputs, targets)

    def _valid(self, **targets):
        outputs = self.handler(**targets, train=False)
        outputs = self.encoder(**outputs, train=False)
        outputs = self.decoder(**outputs, train=False)

        return self.handler.reverse(**outputs)

    def loss(self, outputs, targets):
        return ChainMap(*[f(outputs, targets) for f in self.losses])
