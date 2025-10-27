from typing import List, Optional

PARSERS = list()


def parser(token: Optional[str] = None):
    return lambda p: PARSERS.append((token, p))


@parser("{")
def lhs(pre, suf, mod: str):
    pre.append(mod)


@parser("}")
def rhs(pre, suf, mod: str):
    suf.append(mod)


@parser("^")
def sup(pre, suf, mod: str):
    pre.append(mod)


@parser("_")
def sub(pre, suf, mod: str):
    pre.append(mod)


def parse(pre, suf, mod: str):
    for token, parser in PARSERS:
        if mod == token or token is None:
            return parser(pre, suf, mod)


def sym_to_tex(sym: List[str], mod: List[str], **kwargs):
    output = []

    for sym, mod in zip(sym, mod):
        pre = []
        suf = []

        for mod in mod:
            parse(pre, suf, mod)

        output.extend(pre)
        output.append(sym)
        output.extend(suf)

    return dict(tex=output)


if __name__ == "__main__":
    import json
    from pathlib import Path

    tex = json.loads(Path("sample_tex.json").read_text())
    sym = json.loads(Path("sample_sym.json").read_text())

    assert tex == list(sym_to_tex(**item) for item in sym)
