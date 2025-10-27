from abc import ABC, abstractmethod
from typing import Sequence

from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from mmengine.structures import BaseDataElement

from gryph.codec import tex_to_sym
from gryph.inkml import paint_inkml, scale_inkml


class FormulaTransform(ABC, BaseTransform):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, results):
        return results | self.progress(**results)

    @abstractmethod
    def progress(self, **kwargs) -> dict:
        raise NotImplementedError


@TRANSFORMS.register_module()
class Annotate(BaseTransform):
    def __init__(self, *, keys, meta):
        super().__init__()

        assert isinstance(keys, Sequence)
        assert isinstance(meta, Sequence)

        self.keys = keys
        self.meta = meta

    def transform(self, results):
        # collect inputs
        data = {k: results[k] for k in self.keys}
        meta = {k: results[k] for k in self.meta}

        # create element
        data = BaseDataElement(metainfo=meta, **data)

        return dict(targets=data)


@TRANSFORMS.register_module()
class LabelTeX(FormulaTransform):
    def progress(self, tex, **kwargs):
        return tex_to_sym(tex, **self.kwargs)


@TRANSFORMS.register_module()
class PaintInk(FormulaTransform):
    def progress(self, ink, **kwargs):
        return dict(img=paint_inkml(ink, **self.kwargs))


@TRANSFORMS.register_module()
class ScaleInk(FormulaTransform):
    def progress(self, ink, **kwargs):
        return dict(ink=scale_inkml(ink, **self.kwargs))
