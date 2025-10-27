from pathlib import Path
from typing import List

from lxml.etree import XMLParser, parse


def parse_point(node: List[str]):
    return tuple(map(float, node.strip().split(" ")[:2]))


def parse_trace(node: List[List[str]]):
    return tuple(map(parse_point, node.text.split(",")))


def parse_inkml(path: Path, hook=None):
    root = parse(path, parser=XMLParser(recover=True)).getroot()

    # get nodes
    math = root.find('./annotation[@type="truth"]', root.nsmap)
    base = root.find('./annotation[@type="label"]', root.nsmap)
    norm = root.find('./annotation[@type="normalizedLabel"]', root.nsmap)
    traces = tuple(map(parse_trace, root.findall("./trace", root.nsmap)))

    # get label
    math = None if math is None else math.text.strip()
    base = None if base is None else base.text.strip()
    norm = None if norm is None else norm.text.strip()
    norm = norm or base or math

    # normalize
    norm = norm if hook is None else hook(path, norm)
    return dict(name=path.stem, tex=norm, ink=traces)
