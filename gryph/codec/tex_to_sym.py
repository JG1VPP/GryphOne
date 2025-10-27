from typing import List, Optional

PARSERS = list()


def parser(token: Optional[str] = None):
    return lambda p: PARSERS.append((token, p))


@parser("{")
def lhs(sym, mod, pre, tex: str):
    pre.append(tex)


@parser("}")
def rhs(sym, mod, pre, tex: str):
    if len(pre):
        sym.append("")
        mod.append(list(pre))
        pre.clear()

    else:
        mod[-1].append(tex)


@parser("^")
def sup(sym, mod, pre, tex: str):
    pre.append(tex)


@parser("_")
def sub(sym, mod, pre, tex: str):
    pre.append(tex)


@parser()
def one(sym, mod, pre, tex: str):
    sym.append(tex)
    mod.append(list(pre))
    pre.clear()


def parse(sym, mod, pre, tex: str):
    for token, parser in PARSERS:
        if tex == token or token is None:
            return parser(sym, mod, pre, tex)


def tex_to_sym(tex: List[str], split=True, **kwargs):
    if split:
        sym = []
        mod = []
        pre = []

        for tex in tex:
            parse(sym, mod, pre, tex)

        sym = list(map("".join, sym))
        mod = list(map("".join, mod))

    else:
        sym = tex
        mod = [""] * len(sym)

    assert len(sym) == len(mod)

    return dict(sym=sym, mod=mod)


if __name__ == "__main__":
    import json
    from pathlib import Path

    tex = json.loads(Path("sample_tex.json").read_text())
    sym = json.loads(Path("sample_sym.json").read_text())

    assert sym == list(tex_to_sym(**item) for item in tex)
