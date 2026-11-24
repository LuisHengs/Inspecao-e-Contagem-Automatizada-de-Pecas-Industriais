"""Ajusta as faixas do classificador e estima o acerto com validação cruzada.

Uso:
    python -m experiments.fit_class_ranges
    python -m experiments.fit_class_ranges --sweep     # sensibilidade a Q e margem

Lê outputs/component_descriptors.csv (gerado por collect_descriptors, que já
EXCLUI as imagens do conjunto de teste) e grava outputs/class_ranges.json.

A estimativa de acerto usa validação cruzada POR IMAGEM: as imagens de cada
dobra de validação nunca participam do ajuste das faixas. Mesmo assim,
Q e margem foram escolhidos olhando esse mesmo número, então ele é levemente
otimista. O número que vale é o da avaliação no conjunto de teste (Iteração 6).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.classification import ESPURIO, classify, fit_class_ranges, group_name  # noqa: E402


def load_rows(path: Path) -> List[dict]:
    """Lê o CSV de collect_descriptors convertendo os tipos."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "image": r["image"],
                "class_id": int(r["class_id"]),
                "fused": r["fused"].strip().lower() == "true",
                "area": float(r["area"]),
                "aspect_ratio": float(r["aspect_ratio"]),
            })
    return rows


def cross_validate(rows: List[dict], k: int = 5, seed: int = 0,
                   quantile: float = config.CLASSIFIER_QUANTILE,
                   margin: float = config.CLASSIFIER_MARGIN) -> dict:
    """Validação cruzada com dobras formadas por IMAGENS inteiras.

    Avalia só componentes não fundidos (os fundidos são tratados à parte).
    Devolve acerto por grupo e por classe, matriz de confusão por grupo
    {real: {previsto: n}} e suporte por grupo.
    """
    rows = [r for r in rows if not r["fused"]]
    images = sorted({r["image"] for r in rows})
    if len(images) < k:
        raise ValueError(f"há {len(images)} imagens, menos que as {k} dobras pedidas")
    order = np.random.default_rng(seed).permutation(len(images))
    fold_of = {images[i]: j % k for j, i in enumerate(order)}

    confusion: Dict[str, Dict[str, int]] = {}
    support: Dict[str, int] = {}
    ok_group = ok_class = 0
    for f in range(k):
        train = [r for r in rows if fold_of[r["image"]] != f]
        test = [r for r in rows if fold_of[r["image"]] == f]
        ranges = fit_class_ranges(train, quantile=quantile, margin=margin)
        for r in test:
            pred = classify(r["area"], r["aspect_ratio"], ranges)
            g_true, g_pred = group_name(r["class_id"]), group_name(pred)
            confusion.setdefault(g_true, {}).setdefault(g_pred, 0)
            confusion[g_true][g_pred] += 1
            support[g_true] = support.get(g_true, 0) + 1
            ok_group += g_true == g_pred
            ok_class += r["class_id"] == pred
    n = len(rows)
    return {"n": n, "group_accuracy": ok_group / n, "class_accuracy": ok_class / n,
            "confusion": confusion, "support": support}


def _print_confusion(res: dict) -> None:
    groups = ["porca", "parafuso", "espurio"]
    print(f"{'real \\ previsto':>16s} " + " ".join(f"{g:>9s}" for g in groups) + f" {'recall':>8s}")
    for g in groups:
        row = res["confusion"].get(g, {})
        tot = res["support"].get(g, 0)
        rec = row.get(g, 0) / tot if tot else float("nan")
        print(f"{g:>16s} " + " ".join(f"{row.get(p, 0):>9d}" for p in groups) + f" {rec:>8.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", type=Path, default=config.OUTPUT_DIR / "component_descriptors.csv")
    ap.add_argument("--out", type=Path, default=config.CLASS_RANGES_FILE)
    ap.add_argument("--quantile", type=float, default=config.CLASSIFIER_QUANTILE)
    ap.add_argument("--margin", type=float, default=config.CLASSIFIER_MARGIN)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="varre Q e margem e mostra o acerto de cada par")
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"ERRO: {args.csv} não encontrado. Rode antes: python -m experiments.collect_descriptors",
              file=sys.stderr)
        return 1
    rows = load_rows(args.csv)
    n_img = len({r["image"] for r in rows})
    print(f"{len(rows)} componentes de {n_img} imagens ({sum(r['fused'] for r in rows)} fundidos, fora do ajuste)")

    if args.sweep:
        print(f"\n{'Q':>5s} {'margem':>7s} {'acerto grupo':>13s} {'acerto classe':>14s}")
        for q in (0.02, 0.05, 0.10, 0.15):
            for m in (0.05, 0.10, 0.15, 0.25):
                r = cross_validate(rows, args.folds, quantile=q, margin=m)
                print(f"{q:5.2f} {m:7.2f} {r['group_accuracy']:13.1%} {r['class_accuracy']:14.1%}")

    res = cross_validate(rows, args.folds, quantile=args.quantile, margin=args.margin)
    print(f"\nValidação cruzada por imagem ({args.folds} dobras), Q={args.quantile}, margem={args.margin}:")
    print(f"  acerto por grupo (porca/parafuso/espúrio): {res['group_accuracy']:.1%}")
    print(f"  acerto por classe (1..13 + espúrio):       {res['class_accuracy']:.1%}")
    print()
    _print_confusion(res)

    ranges = fit_class_ranges(rows, quantile=args.quantile, margin=args.margin)
    ranges.save(args.out)
    print("\nFaixas ajustadas (todas as imagens de treino):")
    print(f"{'classe':>6s} {'grupo':9s} {'n':>4s} {'área (px)':>20s} {'razão de aspecto':>20s}")
    for cid, e in ranges.to_dict()["classes"].items():
        print(f"{cid:>6s} {e['group']:9s} {e['n']:>4d} "
              f"{e['area_min']:>9.0f} – {e['area_max']:<8.0f} {e['aspect_min']:>9.2f} – {e['aspect_max']:<8.2f}")
    print(f"\nFaixas salvas em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
