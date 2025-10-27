from functools import partial

import torch
import torch.nn as nn
from toolz.dicttoolz import valmap

from gryph.utils import MODELS, build


@MODELS.register_module()
class FormulaBatcher(nn.Module):
    def __init__(self, lexicon: dict, **kwargs):
        super().__init__()

        # vocab
        self.lexicon = build(lexicon, **kwargs)

    @property
    def num_class(self):
        return self.lexicon.num_class

    @property
    def MASK(self):
        return self.lexicon.MASK

    def forward(self, batch, length: int):
        lexicon = partial(self.lexicon, length=length)
        return torch.stack(tuple(map(lexicon, batch)))

    def reverse(self, outputs):
        return valmap(self.lexicon.reverse, outputs)
