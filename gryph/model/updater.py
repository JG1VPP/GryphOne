import positional_encodings.torch_encodings as P
import torch
import torch.nn as nn

from gryph.block import Linear
from gryph.utils import MODELS


@MODELS.register_module()
class FormulaUpdater(nn.Module):
    def __init__(self, batcher, d_model: int, **kwargs):
        super().__init__()

        # embeddings
        self.emb = nn.Embedding(batcher.num_class, d_model)
        self.pos = P.PositionalEncoding1D(channels=d_model)

        # heads
        self.sym = Linear(d_model, self.emb.num_embeddings)
        self.mod = Linear(d_model, self.emb.num_embeddings)

        # special token
        self.register_buffer("MASK", batcher.MASK)

    def feature(self, sym, mod, **kwargs):
        assert sym.ndim == 2
        assert mod.ndim == 2

        sym = self.emb(sym)
        mod = self.emb(mod)

        tex = self.pos(sym).add(sym).add(mod)

        return dict(kwargs, tex=tex)

    def diffuse(self, sym, mod, time, **kwargs):
        assert sym.ndim == 2
        assert mod.ndim == 2

        sym = self.remask(sym, time=time)
        mod = self.remask(mod, time=time)

        return dict(kwargs, sym=sym, mod=mod)

    def forward(self, sym, mod, tex, **kwargs):
        assert sym.ndim == 2
        assert mod.ndim == 2

        assert tex.ndim == 3

        sym = self.unmask(sym, self.sym(tex))
        mod = self.unmask(mod, self.mod(tex))

        return dict(kwargs, sym=sym, mod=mod)

    def predict(self, tex, **kwargs):
        assert tex.ndim == 3

        sym = self.sym(tex)
        mod = self.mod(tex)

        return dict(kwargs, sym=sym, mod=mod)

    def remask(self, batch, time):
        rand = torch.rand(*batch.shape, device=batch.device)
        return batch.masked_fill(rand.less(time), self.MASK)

    def unmask(self, batch, output):
        return batch.where(self.MASK.ne(batch), output.argmax(dim=2))
