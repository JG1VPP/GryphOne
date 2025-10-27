from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gryph.utils import MODELS


@MODELS.register_module()
class FormulaVocab(nn.Module):
    def __init__(
        self,
        load: str,
        skip: str,
        mask: str,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(load, str)

        assert isinstance(skip, str)
        assert isinstance(mask, str)

        self.fwd, self.rev = self.parse(load)
        assert len(self.fwd) == len(self.rev)

        # special tokens
        self.register_buffer("SKIP", torch.tensor(self.fwd[skip]))
        self.register_buffer("MASK", torch.tensor(self.fwd[mask]))

    def parse(self, path: str):
        rev_list = list(Path(path).read_text().splitlines())
        fwd_dict = dict(zip(rev_list, range(len(rev_list))))

        return fwd_dict, rev_list

    @property
    def num_class(self):
        return len(self.fwd)

    def forward(self, data, length: int):
        data = map(self.get_number, data)
        data = map(self.get_tensor, data)

        data = torch.stack(list(data))
        size = (0, length - len(data))

        return F.pad(data, pad=size, value=self.SKIP)

    def reverse(self, data):
        skip = data.eq(self.SKIP).cumsum(dim=0).logical_not()
        data = map(self.get_string, data.masked_select(skip))

        return tuple(data)

    def get_tensor(self, value):
        return torch.tensor(value).to(self.SKIP)

    def get_number(self, token):
        return self.fwd[token]

    def get_string(self, token):
        return self.rev[token]
