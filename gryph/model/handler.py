from typing import Sequence

import torch
from mmengine import MODELS
from mmengine.model import BaseModule
from more_itertools import transpose
from toolz.dicttoolz import keymap

from gryph.codec import sym_to_tex


@MODELS.register_module()
class FormulaHandler(BaseModule):
    def __init__(
        self,
        batcher,
        len_tex: int,
        outputs: list,
        targets: list,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(len_tex, int)

        assert isinstance(outputs, Sequence)
        assert isinstance(targets, Sequence)

        # vocab
        self.batcher = batcher

        # params
        self.len_tex = len_tex

        # labels
        self.outputs = outputs
        self.targets = targets

    @property
    def MASK(self):
        return self.batcher.MASK

    def forward(self, targets, train: bool):
        if train:
            return self._train(targets)
        else:
            return self._valid(targets)

    def reverse(self, targets, train: bool, **items):
        outputs = []

        for item in transpose(items.values()):
            outputs.append(dict(zip(items.keys(), item)))

        return tuple(map(self.itemize, outputs, targets))

    def itemize(self, outputs, targets):
        # labels
        outputs = {k: outputs.get(k) for k in self.outputs}
        targets = {k: targets.get(k) for k in self.targets}

        outputs = self.batcher.reverse(outputs)

        # codec
        outputs.update(sym_to_tex(**outputs))
        targets.update(sym_to_tex(**targets))

        return dict(outputs=outputs, targets=targets)

    def _train(self, batch):
        img = self.tensor("img", batch, self.tensor_img)
        sym = self.tensor("sym", batch, self.tensor_tex)
        mod = self.tensor("mod", batch, self.tensor_tex)

        # batch
        item = dict(targets=batch, time=self.step(img))

        # tasks
        item.update(img=img)
        item.update(sym=sym)
        item.update(mod=mod)

        return item

    def _valid(self, batch):
        img = self.tensor("img", batch, self.tensor_img)
        sym = self.MASK.expand(len(batch), self.len_tex)
        mod = self.MASK.expand(len(batch), self.len_tex)

        # batch
        item = dict(targets=batch)

        # tasks
        item.update(img=img)
        item.update(sym=sym)
        item.update(mod=mod)

        return item

    def tensor(self, key, batch, op):
        return op(tuple(m.get(key) for m in batch))

    def tensor_img(self, batch):
        return torch.stack(batch)

    def tensor_tex(self, batch):
        return self.batcher(batch, length=self.len_tex)

    def step(self, img):
        return torch.rand(len(img), 1).to(img.device)

    def rename(self, tensors, suffix: str):
        return keymap(lambda k: f"{k}{suffix}", tensors)
