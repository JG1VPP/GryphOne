from mmengine.model import BaseModule

from gryph.block import Blocks
from gryph.utils import MODELS, build


@MODELS.register_module()
class FormulaDecoder(BaseModule):
    def __init__(self, updater: dict, **kwargs):
        super().__init__()

        # vocab
        self.updater = build(updater, **kwargs)

        # blocks
        self.decoder = Blocks(**kwargs)

    def forward(self, train: bool, **kwargs):
        if train:
            return self._train(**kwargs, train=train)
        else:
            return self._valid(**kwargs, train=train)

    def _train(self, **data):
        outs = self.updater.diffuse(**data)
        outs = self.updater.feature(**outs)
        outs = self.decoder.forward(**outs)
        outs = self.updater.predict(**outs)

        return outs

    def _valid(self, **data):
        outs = self.updater.feature(**data)
        outs = self.decoder.forward(**outs)
        outs = self.updater(**data, **outs)
        outs = self.updater.diffuse(**outs)

        return outs
