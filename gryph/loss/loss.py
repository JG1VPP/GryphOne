from abc import ABC, abstractmethod

import torch.nn as nn
from mmengine import MODELS


class Loss(nn.Module, ABC):
    label = "loss_{}"

    def __init__(self, key: str, **kwargs):
        super().__init__()

        # keys
        self.key = key
        self.base = self.function(**kwargs)
        self.label = self.label.format(key)

    @abstractmethod
    def function(self, **kwargs):
        raise NotImplementedError

    def forward(self, outputs, targets):
        inputs = self.format(outputs, targets)
        return {self.label: self.loss(*inputs)}

    @abstractmethod
    def format(self, outputs, targets):
        raise NotImplementedError

    def loss(self, norm, *inputs):
        return self.base(*inputs).div(norm)


@MODELS.register_module()
class CELoss(Loss):
    label = "loss_ce_{}"

    def function(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)

    def format(self, outputs, targets):
        # outputs [N, L, C]
        # targets [N, L]
        logit = outputs[self.key]
        label = targets[self.key]

        return 1, logit.mT, label


@MODELS.register_module()
class KLLoss(Loss):
    label = "loss_kl_{}"

    def function(self, mut: str, **kwargs):
        # key
        self.mut = mut

        # prob
        self.p = nn.Softmax(dim=2)
        self.q = nn.LogSoftmax(dim=2)

        # loss
        return nn.KLDivLoss(reduction="sum")

    def format(self, outputs, targets):
        # outputs [N, L, C]
        logit_f = outputs[self.key]
        logit_b = outputs[self.mut]
        label_f = targets[self.key]

        # P: target
        # Q: output
        p = self.p(logit_b).detach()
        q = self.q(logit_f)

        return label_f.numel(), q, p
