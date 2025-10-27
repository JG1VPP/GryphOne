#!/usr/bin/python3
import argparse
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List

from dacite import from_dict
from more_itertools import flatten
from tqdm import tqdm
from yaml import safe_load

from gryph.inkml import parse_inkml


@dataclass
class Load:
    type: str
    path: str
    glob: str


@dataclass
class Dump:
    path: str


@dataclass
class Config:
    load: List[Load]
    dump: Dump
    word: str
    norm: Dict[str, List[str]]


def options():
    args = argparse.ArgumentParser()
    args.add_argument("cfg")
    args = args.parse_args()

    path = Path(args.cfg).expanduser()
    yaml = safe_load(path.read_text())

    return from_dict(Config, yaml)


def normalize(token: str, params):
    return params.norm.get(token, [token])


def handle(path: Path, formula: str, params):
    pattern = re.compile(params.word)
    tokens = pattern.findall(formula)
    assert formula == "".join(tokens)

    norm = partial(normalize, params=params)
    return tuple(flatten(map(norm, tokens)))


def process(params, sets):
    hook = partial(handle, params=params)
    ink = partial(parse_inkml, hook=hook)

    for load in params.load:
        batch = Path(load.path).expanduser()
        batch = list(batch.rglob(load.glob))
        batch = map(ink, tqdm(batch))
        sets[load.type].extend(batch)

    dump = Path(params.dump.path).expanduser()
    dump.write_bytes(pickle.dumps(dict(sets)))


if __name__ == "__main__":
    process(options(), sets=defaultdict(list))
