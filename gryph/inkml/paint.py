from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor


def paint_inkml(traces, w: int, h: int, fill: int, line: int):
    img = Image.new(mode="RGB", size=(w, h))
    ink = ImageDraw.Draw(img)

    for trace in traces:
        ink.line(trace, fill=fill, width=line)

    return pil_to_tensor(img).float()
