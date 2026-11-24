"""Coleta os descritores de cada componente segmentado, agrupados por classe.

Objetivo: tirar os LIMIARES do classificador por regras (Iteração 5) dos dados,
e não de chute. Para cada componente (>= MIN_AREA px) o script descobre, por
sobreposição de pixels com as caixas GT, a que peça ele corresponde, e resume
área, razão de aspecto, circularidade etc. por classe.

Uso:
    python -m experiments.collect_descriptors
    python -m experiments.collect_descriptors --limit 80     # amostra espaçada
    python -m experiments.collect_descriptors --include-test # NÃO recomendado

POR PADRÃO as imagens do mvtec_screws_test.json são EXCLUÍDAS da coleta: os
limiares saem do restante do dataset e o conjunto de teste fica intocado para
a avaliação final (Iteração 6). Usar o teste para escolher limiares e depois
avaliar nele infla o resultado.

Rótulos de cada componente:
  * classe 1..13 : >= 50 % dos pixels do componente caem dentro de caixas GT e
                   a caixa com mais sobreposição é daquela classe;
  * classe 0     : espúrio (mancha de madeira, sombra...);
  * fused=True   : o componente cobre >= 20 % de DUAS ou mais caixas, ou seja,
                   é um bloco de peças encostadas (o descritor não representa
                   uma peça só e fica fora do resumo por padrão).

Saídas (em outputs/):
  * component_descriptors.csv : uma linha por componente;
  * descriptor_summary.csv    : mediana e percentis 10/90 por classe;
  * gt_box_sizes.csv          : lado maior/menor das caixas GT por classe
                                (independe da segmentação).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Permite rodar também como "python experiments/collect_descriptors.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.evaluation import box_mask, load_annotations  # noqa: E402
from src.features import component_features, label_components  # noqa: E402
from src.preprocessing import load_image  # noqa: E402
from src.segmentation import segment_mask  # noqa: E402

ESPURIO = 0
MIN_AREA = 300         # px; mesmo valor do experimento de canais
MIN_INSIDE = 0.5       # fração mínima do componente dentro de caixas GT
FUSED_HIT = 0.2        # fração mínima de uma caixa coberta para "tocá-la"

FEATURES = ("area", "perimeter", "circularity", "aspect_ratio",
            "eccentricity", "extent", "major_axis_length", "minor_axis_length")


# --------------------------------------------------------------------------
# Lógica testável
# --------------------------------------------------------------------------
def assign_components(
    labels: np.ndarray,
    n: int,
    gt_masks: Sequence[np.ndarray],
    gt_classes: Sequence[int],
    min_area: int = MIN_AREA,
    min_inside: float = MIN_INSIDE,
    fused_hit: float = FUSED_HIT,
) -> Dict[int, dict]:
    """Casa cada componente (>= min_area) com as caixas GT por pixels.

    Devolve {rótulo: {class_id, inside_frac, n_gt, fused}}. `inside_frac` é a
    fração do componente que cai em alguma caixa; `n_gt` é quantas caixas o
    componente cobre em >= `fused_hit` da área delas.
    """
    if len(gt_masks) != len(gt_classes):
        raise ValueError("gt_masks e gt_classes devem ter o mesmo tamanho")

    areas = np.bincount(labels.ravel(), minlength=n + 1)
    overlap = np.zeros((n + 1, len(gt_masks)), dtype=np.int64)  # px comp ∩ caixa
    inside = np.zeros(n + 1, dtype=bool)
    in_any = np.zeros(labels.shape, dtype=bool)
    gt_area = np.zeros(len(gt_masks))
    for k, g in enumerate(gt_masks):
        overlap[:, k] = np.bincount(labels[g], minlength=n + 1)
        gt_area[k] = max(int(g.sum()), 1)
        in_any |= g
    inside = np.bincount(labels[in_any], minlength=n + 1)

    out: Dict[int, dict] = {}
    for lab in range(1, n + 1):
        if areas[lab] < min_area:
            continue
        frac = inside[lab] / areas[lab]
        n_gt = int((overlap[lab] / gt_area >= fused_hit).sum()) if len(gt_masks) else 0
        if len(gt_masks) and frac >= min_inside:
            class_id = int(gt_classes[int(np.argmax(overlap[lab]))])
        else:
            class_id = ESPURIO
        out[lab] = {
            "class_id": class_id,
            "inside_frac": float(frac),
            "n_gt": n_gt,
            "fused": n_gt >= 2,
        }
    return out


def summarize(
    rows: Iterable[dict],
    features: Sequence[str] = FEATURES,
    include_fused: bool = False,
) -> Dict[int, dict]:
    """Mediana e percentis 10/90 de cada descritor, por class_id."""
    groups: Dict[int, List[dict]] = {}
    for r in rows:
        if r.get("fused") and not include_fused:
            continue
        groups.setdefault(int(r["class_id"]), []).append(r)
    summary: Dict[int, dict] = {}
    for cid, items in sorted(groups.items()):
        entry: dict = {"n": len(items)}
        for f in features:
            v = np.array([it[f] for it in items if f in it], dtype=float)
            if v.size:
                entry[f] = {
                    "median": float(np.median(v)),
                    "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90)),
                }
        summary[cid] = entry
    return summary


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
def _group_name(cid: int) -> str:
    return "espurio" if cid == ESPURIO else config.group_of(cid)


def _pick_images(ds, images_dir: Path, exclude_names: set, limit: Optional[int]) -> List[int]:
    ids = sorted(
        i for i, im in ds.images.items()
        if im.file_name not in exclude_names and (images_dir / im.file_name).is_file()
    )
    if limit and limit < len(ids):  # amostra espaçada: evita pegar só um tipo de cena
        ids = [ids[j] for j in np.linspace(0, len(ids) - 1, limit).round().astype(int)]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=config.ANNOTATIONS_FILE)
    ap.add_argument("--test-annotations", type=Path, default=config.ANNOTATIONS_TEST_FILE)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--channel", default="b")
    ap.add_argument("--limit", type=int, default=None, help="nº de imagens (amostra espaçada)")
    ap.add_argument("--include-test", action="store_true",
                    help="inclui as imagens do conjunto de teste (contamina a avaliação final)")
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = ap.parse_args()

    try:
        ds = load_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERRO ao ler anotações: {exc}", file=sys.stderr)
        return 1
    if not args.images_dir.is_dir():
        print(f"ERRO: pasta de imagens não encontrada: {args.images_dir}", file=sys.stderr)
        return 1

    exclude: set = set()
    if not args.include_test:
        if not args.test_annotations.is_file():
            print(f"ERRO: {args.test_annotations} não encontrado. Sem ele não dá para separar o teste; "
                  "aponte --test-annotations ou use --include-test (não recomendado).", file=sys.stderr)
            return 1
        test_ds = load_annotations(args.test_annotations)
        exclude = {im.file_name for im in test_ds.images.values()}

    ids = _pick_images(ds, args.images_dir, exclude, args.limit)
    if not ids:
        print("ERRO: nenhuma imagem utilizável encontrada em disco.", file=sys.stderr)
        return 1
    print(f"Imagens usadas: {len(ids)} (excluídas do teste: {len(exclude)})  canal: {args.channel}")

    rows: List[dict] = []
    gt_sizes: Dict[int, List[tuple]] = {}
    for n_done, i in enumerate(ids, 1):
        info = ds.images[i]
        img = load_image(args.images_dir / info.file_name)
        anns = ds.annotations_by_image[i]
        gt_masks = [box_mask(a.box, img.shape[:2]) for a in anns]
        for a in anns:
            gt_sizes.setdefault(a.category_id, []).append(
                (max(a.box.width, a.box.height), min(a.box.width, a.box.height)))

        mask, _ = segment_mask(img, channel=args.channel)
        labels, n = label_components(mask)
        assign = assign_components(labels, n, gt_masks, [a.category_id for a in anns])
        for f in component_features(labels, n):
            a = assign.get(f.label)
            if a is None:
                continue
            rows.append({
                "image": info.file_name, "label": f.label, **a,
                "group": _group_name(a["class_id"]),
                **{k: getattr(f, k) for k in FEATURES if hasattr(f, k)},
            })
        if n_done % 10 == 0 or n_done == len(ids):
            print(f"  {n_done}/{len(ids)} imagens", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image", "label", "class_id", "group", "inside_frac", "n_gt", "fused", *FEATURES]
    with open(args.out_dir / "component_descriptors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    key = ("area", "aspect_ratio", "circularity", "extent")
    print("\nResumo por classe (componentes NÃO fundidos): mediana [p10 – p90]")
    head = f"{'classe':>6s} {'grupo':9s} {'n':>5s} " + " ".join(f"{k:>26s}" for k in key)
    print(head)
    print("-" * len(head))
    with open(args.out_dir / "descriptor_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "group", "n"] + [f"{k}_{s}" for k in FEATURES for s in ("median", "p10", "p90")])
        for cid, e in summary.items():
            cells = []
            for k in FEATURES:
                s = e.get(k, {})
                cells += [s.get("median", ""), s.get("p10", ""), s.get("p90", "")]
            w.writerow([cid, _group_name(cid), e["n"]] + cells)
            line = f"{cid:>6d} {_group_name(cid):9s} {e['n']:>5d} "
            line += " ".join(
                f"{e[k]['median']:>10.2f} [{e[k]['p10']:>6.2f} – {e[k]['p90']:>6.2f}]" for k in key)
            print(line)

    n_fused = sum(1 for r in rows if r["fused"])
    print(f"\nComponentes: {len(rows)} no total, {n_fused} fundidos "
          f"({100 * n_fused / max(len(rows), 1):.1f} %), "
          f"{sum(1 for r in rows if r['class_id'] == ESPURIO)} espúrios.")

    with open(args.out_dir / "gt_box_sizes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "group", "n", "lado_maior_mediana", "lado_menor_mediana"])
        print("\nTamanho das caixas GT (px), independe da segmentação:")
        for cid in sorted(gt_sizes):
            v = np.array(gt_sizes[cid])
            big, small = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
            w.writerow([cid, _group_name(cid), len(v), round(big, 1), round(small, 1)])
            print(f"  classe {cid:>2d} {_group_name(cid):9s} n={len(v):>4d}  "
                  f"maior={big:7.1f}  menor={small:6.1f}")
    print(f"\nArquivos salvos em {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
