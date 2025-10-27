import torch
from mmengine.model import BaseModule

from gryph.utils import MODELS, build


@MODELS.register_module()
class FormulaRefiner(BaseModule):
    def __init__(self, model: dict, steps: int, **kwargs):
        super().__init__()

        # model
        self.model = build(model, **kwargs)

        # param
        self.steps = steps

    def forward(self, train: bool, **kwargs):
        if train:
            return self._train(**kwargs, train=train)
        else:
            return self._valid(**kwargs, train=train)

    def _train(self, **kwargs):
        return self.model(**kwargs)

    def _valid(self, **kwargs):
        for time in torch.linspace(1, 0, self.steps):
            kwargs = self.model(**kwargs, time=time)

        return kwargs
