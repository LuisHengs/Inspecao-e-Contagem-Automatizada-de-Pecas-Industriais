"""Experimento: qual canal de cor segmenta melhor as peças sobre a madeira?

Uso:
    python -m experiments.compare_channels                 # todas as imagens do JSON
    python -m experiments.compare_channels --limit 20      # só as 20 primeiras
    python -m experiments.compare_channels --annotations data/mvtec_screws/mvtec_screws_test.json

Métricas por configuração (médias sobre as imagens):
  * acerto_objetos : fração de caixas GT com >= 10% de pixels de primeiro plano
                     (a peça foi "tocada" pela segmentação);
  * precisao_px    : fração do primeiro plano que cai dentro de alguma caixa GT;
  * espurios       : componentes (>= 300 px) cujo centro NÃO está em nenhuma caixa;
  * comp_por_gt    : nº de componentes / nº de peças reais (ideal ~1.0).

IMPORTANTE: as caixas orientadas são um proxy do contorno real (parafusos finos
não preenchem a caixa), então as métricas servem para COMPARAR configurações,
não como acurácia absoluta. A avaliação final (P/R/F1 com IoU) vem na iteração 6.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Permite rodar também como "python experiments/compare_channels.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.evaluation import box_mask, load_annotations  # noqa: E402
from src.preprocessing import load_image  # noqa: E402
from src.segmentation import segment_mask  # noqa: E402

MIN_AREA = 300          # px; ignora poeira ao contar componentes
HIT_FRACTION = 0.10     # fração mínima da caixa coberta para contar como "acerto"

CONFIGS = [
    ("gray", False), ("L", False), ("b", False), ("S", False), ("V", False),
    ("dist_lab", False), ("gray", True), ("dist_lab", True),
]


def evaluate_image(mask: np.ndarray, gt_masks: list) -> dict:
    fg = mask > 0
    any_gt = np.zeros_like(fg)
    hits = 0
    for g in gt_masks:
        any_gt |= g
        if g.sum() > 0 and (fg & g).sum() / g.sum() >= HIT_FRACTION:
            hits += 1
    precision = (fg & any_gt).sum() / max(fg.sum(), 1)

    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = spurious = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        comps += 1
        cx, cy = int(cents[i][0]), int(cents[i][1])
        if not any_gt[cy, cx]:
            spurious += 1
    return {
        "acerto": hits / max(len(gt_masks), 1),
        "precisao": float(precision),
        "espurios": spurious,
        "comp_por_gt": comps / max(len(gt_masks), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=config.ANNOTATIONS_FILE)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--limit", type=int, default=None, help="máximo de imagens")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "channel_comparison.csv")
    args = ap.parse_args()

    try:
        ds = load_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERRO ao ler anotações: {exc}", file=sys.stderr)
        return 1
    if not args.images_dir.is_dir():
        print(f"ERRO: pasta de imagens não encontrada: {args.images_dir}", file=sys.stderr)
        return 1

    # Só usa imagens que existem em disco (permite rodar com um subconjunto).
    ids = [i for i, im in ds.images.items() if (args.images_dir / im.file_name).is_file()]
    ids = sorted(ids)[: args.limit] if args.limit else sorted(ids)
    if not ids:
        print("ERRO: nenhuma imagem das anotações foi encontrada em disco.", file=sys.stderr)
        return 1
    print(f"Imagens avaliadas: {len(ids)}\n")

    cache = {}
    for i in ids:
        img = load_image(args.images_dir / ds.images[i].file_name)
        gts = [box_mask(a.box, img.shape[:2]) for a in ds.annotations_by_image[i]]
        cache[i] = (img, gts)

    rows = []
    header = f"{'canal':9s} {'CLAHE':5s} {'acerto':>7s} {'prec_px':>8s} {'espurios':>9s} {'comp/GT':>8s} {'ms/img':>7s}"
    print(header)
    print("-" * len(header))
    for channel, clahe in CONFIGS:
        res, t0 = [], time.perf_counter()
        for i in ids:
            img, gts = cache[i]
            mask, _ = segment_mask(img, channel=channel, clahe=clahe)
            res.append(evaluate_image(mask, gts))
        ms = (time.perf_counter() - t0) / len(ids) * 1000
        row = {
            "canal": channel, "clahe": clahe, "n_imagens": len(ids),
            "acerto": np.mean([r["acerto"] for r in res]),
            "precisao": np.mean([r["precisao"] for r in res]),
            "espurios": np.mean([r["espurios"] for r in res]),
            "comp_por_gt": np.mean([r["comp_por_gt"] for r in res]),
            "ms_por_img": ms,
        }
        rows.append(row)
        print(f"{channel:9s} {'sim' if clahe else 'não':5s} {row['acerto']:7.2f} "
              f"{row['precisao']:8.2f} {row['espurios']:9.1f} {row['comp_por_gt']:8.2f} {ms:7.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResultados salvos em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
