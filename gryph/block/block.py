import abc
import math

import torch
import torch.nn as nn

from gryph.utils import MODELS, build


class Linear(nn.Sequential):
    def __init__(self, d: int, h: int, *, act=nn.Identity):
        super().__init__(nn.LayerNorm(d), nn.Linear(d, h), act())


class Attention(nn.Module, abc.ABC):
    def __init__(self, heads: int, d_model: int, **kwargs):
        super().__init__()

        assert d_model % heads == 0

        # params
        self.dim = int(d_model // heads)
        self.lhd = (-1, heads, self.dim)

        # layers
        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.w = Linear(d_model, d_model)

    def forward(self, q, k, v, **kwargs):
        # inputs
        x = self.q(q).view(len(q), *self.lhd).swapaxes(1, 2)
        k = self.k(k).view(len(k), *self.lhd).swapaxes(1, 2)
        v = self.v(v).view(len(v), *self.lhd).swapaxes(1, 2)

        # output
        x = self.attention(x, k, v, **kwargs).swapaxes(1, 2)
        return self.w(x.contiguous().flatten(-2)).view_as(q)

    @abc.abstractmethod
    def attention(self, q, k, v, **kwargs):
        raise NotImplementedError


@MODELS.register_module()
class GlobalAttention(Attention):
    def __init__(self, dropout: float, **kwargs):
        super().__init__(**kwargs)

        self.drop = nn.Dropout(dropout)

    def attention(self, q, k, v, **kwargs):
        p = q.matmul(k.mT.div(math.sqrt(v.size(-1))))
        return self.drop(self.weight(p, q)).matmul(v)

    def weight(self, p, q):
        return p.softmax(dim=-1)


@MODELS.register_module()
class AbsentAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q, k, v, **kwargs):
        return torch.zeros_like(q)


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.append(Linear(d_model, d_model, act=nn.ReLU))
        self.append(Linear(d_model, d_model, act=nn.Identity))


@MODELS.register_module()
class Block(nn.Module):
    def __init__(self, att1, att2, **kwargs):
        super().__init__()

        # inputs
        self.att1 = build(att1, **kwargs)
        self.att2 = build(att2, **kwargs)

        # output
        self.feed = FeedForward(**kwargs)

    def forward(self, x, y):
        x = self.att1(x, x, x).add(x)
        x = self.att2(x, y, y).add(x)

        return self.feed(x).add(x)


class Blocks(nn.Sequential):
    def __init__(self, blocks: list, **kwargs):
        super().__init__()

        # blocks
        block = lambda param: Block(**param, **kwargs)
        self.model = nn.ModuleList(map(block, blocks))

    def forward(self, tex, img, **kwargs):
        for block in self.model:
            tex = block(tex, img)

        return dict(tex=tex)
