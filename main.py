"""Ponto de entrada principal para contagem, inspeção e visualização do pipeline.

Executa o fluxo completo em duas passadas (segmentação no canal `b` ->
Otsu + morfologia -> rotulagem e descritores -> Watershed seletivo em blocos
fundidos -> classificação nas 13 classes) e exporta painéis visuais das etapas.

Exemplos:
    python main.py --image data/mvtec_screws/images/screws_001.png
    python main.py --limit 10 --save-vis 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

import config
from experiments.evaluate_test import run_two_pass_inference
from src.classification import ClassRanges
from src.evaluation import (
    Annotation,
    Detection,
    box_corners,
    evaluate_dataset_predictions,
    load_annotations,
)
from src.preprocessing import gaussian_blur, get_channel, load_image
from src.segmentation import clean_mask, otsu_threshold


def _draw_oriented_boxes(
    img_bgr: np.ndarray,
    boxes_and_labels: Sequence[tuple],
    thickness: int = 3,
) -> np.ndarray:
    """Desenha caixas delimitadoras orientadas e rótulos sobre uma cópia da imagem."""
    canvas = img_bgr.copy()
    for box, text, color in boxes_and_labels:
        pts = np.round(box_corners(box)).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
        if text:
            anchor = (int(pts[:, 0].min()), max(22, int(pts[:, 1].min()) - 6))
            cv2.putText(
                canvas,
                text,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    return canvas


def save_step_visualizations(
    img_bgr: np.ndarray,
    stem: str,
    channel: str,
    detections: List[Detection],
    meta: dict,
    out_dir: Path,
    gts: Optional[Sequence[Annotation]] = None,
) -> int:
    """Salva as etapas intermediárias do processamento e a comparação com GT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_bgr.shape[:2]
    scale = config.resolution_scale(h, w)

    blurred = gaussian_blur(img_bgr, config.odd_kernel(5, scale))
    ch_img = get_channel(blurred, channel)
    _, raw_otsu = otsu_threshold(ch_img, invert=True)
    morph_mask = clean_mask(
        raw_otsu, config.odd_kernel(3, scale), config.odd_kernel(5, scale), "ellipse"
    )
    final_mask = meta["mask"]
    labels = meta["labels"]

    rng = np.random.default_rng(42)
    n_lbl = int(labels.max())
    palette = np.zeros((n_lbl + 1, 3), dtype=np.uint8)
    if n_lbl > 0:
        palette[1:] = rng.integers(60, 255, size=(n_lbl, 3), dtype=np.uint8)
    labeled_vis = palette[labels]

    pred_items = []
    for d in detections:
        color = (0, 220, 0) if d.group == config.GROUP_NUT else (255, 140, 0)
        pred_items.append((d.box, f"C{d.class_id} ({d.group})", color))
    result_vis = _draw_oriented_boxes(img_bgr, pred_items)

    n_screws = sum(1 for d in detections if d.group == config.GROUP_SCREW)
    n_nuts = sum(1 for d in detections if d.group == config.GROUP_NUT)
    banner = f"Parafusos: {n_screws} | Porcas: {n_nuts} | Total: {len(detections)}"
    cv2.rectangle(result_vis, (0, 0), (w, 46), (30, 30, 30), -1)
    cv2.putText(
        result_vis, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )

    panel = np.hstack([
        cv2.resize(img_bgr, (w // 2, h // 2)),
        cv2.resize(result_vis, (w // 2, h // 2)),
    ])

    files = [
        (f"{stem}_00_painel_antes_depois.png", panel),
        (f"{stem}_01_original.png", img_bgr),
        (f"{stem}_02_canal_{channel}.png", ch_img),
        (f"{stem}_03_otsu_bruto.png", raw_otsu),
        (f"{stem}_04_morfologia.png", final_mask if meta["n_fused_candidates"] else morph_mask),
        (f"{stem}_05_componentes_rotulados.png", labeled_vis),
        (f"{stem}_06_resultado_final.png", result_vis),
    ]

    if gts is not None:
        gt_items = [(a.box, f"C{a.category_id}", (0, 230, 255)) for a in gts]
        gt_vis = _draw_oriented_boxes(img_bgr, gt_items)
        gt_s = sum(1 for a in gts if config.group_of(a.category_id) == config.GROUP_SCREW)
        gt_n = sum(1 for a in gts if config.group_of(a.category_id) == config.GROUP_NUT)
        cv2.rectangle(gt_vis, (0, 0), (w, 46), (30, 30, 30), -1)
        cv2.putText(
            gt_vis,
            f"GT -> Parafusos: {gt_s} | Porcas: {gt_n} | Total: {len(gts)}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 230, 255),
            2,
        )
        comp_gt = np.hstack([
            cv2.resize(gt_vis, (w // 2, h // 2)),
            cv2.resize(result_vis, (w // 2, h // 2)),
        ])
        files.append((f"{stem}_07_comparacao_gt.png", comp_gt))

    for fname, arr in files:
        cv2.imwrite(str(out_dir / fname), arr)
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--image", type=Path, default=None, help="caminho de uma imagem única")
    ap.add_argument("--annotations", type=Path, default=config.ANNOTATIONS_TEST_FILE)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--ranges", type=Path, default=config.CLASS_RANGES_FILE)
    ap.add_argument("--channel", default="b")
    ap.add_argument("--no-split-fused", action="store_true", help="desativa Watershed na 2ª passada")
    ap.add_argument("--limit", type=int, default=10, help="nº de imagens no modo lote")
    ap.add_argument("--save-vis", type=int, default=5, help="nº de imagens com figuras salvas")
    ap.add_argument("--vis-dir", type=Path, default=config.OUTPUT_DIR / "visualizations")
    args = ap.parse_args()

    ranges = ClassRanges.load(args.ranges)

    if args.image is not None:
        img = load_image(args.image)
        dets, meta = run_two_pass_inference(
            img, ranges, channel=args.channel, split_fused=not args.no_split_fused
        )
        n_screws = sum(1 for d in dets if d.group == config.GROUP_SCREW)
        n_nuts = sum(1 for d in dets if d.group == config.GROUP_NUT)
        n_saved = save_step_visualizations(
            img, args.image.stem, args.channel, dets, meta, args.vis_dir
        )
        print(f"Imagem: {args.image.name}")
        print(f"  Parafusos detectados : {n_screws}")
        print(f"  Porcas detectadas    : {n_nuts}")
        print(f"  Total de peças       : {len(dets)}")
        print(f"  Tempo total          : {meta['ms_total']:.1f} ms")
        print(f"  Figuras salvas ({n_saved}) em: {args.vis_dir}")
        return 0

    ds = load_annotations(args.annotations)
    ids = sorted(
        iid for iid, im in ds.images.items() if (args.images_dir / im.file_name).is_file()
    )[: args.limit]
    if not ids:
        print(f"ERRO: nenhuma imagem encontrada em {args.images_dir}", file=sys.stderr)
        return 1

    preds_by_image = {}
    times_ms = []
    for idx, iid in enumerate(ids):
        info = ds.images[iid]
        img = load_image(args.images_dir / info.file_name)
        dets, meta = run_two_pass_inference(
            img, ranges, channel=args.channel, split_fused=not args.no_split_fused
        )
        preds_by_image[iid] = dets
        times_ms.append(meta["ms_total"])
        if idx < args.save_vis:
            save_step_visualizations(
                img,
                Path(info.file_name).stem,
                args.channel,
                dets,
                meta,
                args.vis_dir,
                gts=ds.annotations_by_image[iid],
            )

    gts_by_image = {iid: ds.annotations_by_image[iid] for iid in ids}
    report = evaluate_dataset_predictions(preds_by_image, gts_by_image)
    macro_f1 = report["by_group"]["macro"]["f1"]
    mae_tot = report["counting"]["total"]["mae"]
    print(f"Processadas {len(ids)} imagens | Tempo médio: {np.mean(times_ms):.1f} ms/img")
    print(f"F1 (macro grupos): {macro_f1:.1%} | MAE contagem total: {mae_tot:.2f} peças/img")
    if args.save_vis > 0:
        print(f"Visualizações salvas em: {args.vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())