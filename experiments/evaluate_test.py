"""Avaliação final do pipeline completo na partição de teste isolada.

Integra as melhorias de robustez:
  1. Escalonamento proporcional de kernels e áreas pela resolução da imagem;
  2. Pipeline de duas passadas: separa componentes grandes rejeitados como
     espúrios via Transformada de Distância + Watershed antes da contagem final;
  3. Descritores topológicos (`solidity` e `num_holes`) na classificação.

Uso:
    python -m experiments.evaluate_test
    python -m experiments.evaluate_test --channel ab_chrom --no-split-fused
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.classification import ESPURIO, ClassRanges, classify, group_name  # noqa: E402
from src.evaluation import (  # noqa: E402
    Detection,
    component_to_box,
    evaluate_dataset_predictions,
    load_annotations,
)
from src.features import ComponentFeatures, component_features, label_components  # noqa: E402
from src.preprocessing import load_image  # noqa: E402
from src.segmentation import segment_mask, split_fused_components  # noqa: E402

BASE_MIN_AREA = 300
BASE_MIN_FUSED_AREA = 4000


def run_two_pass_inference(
    img_bgr: np.ndarray,
    ranges: ClassRanges,
    channel: str = "b",
    split_fused: bool = True,
) -> Tuple[List[Detection], dict]:
    """Executa detecção completa em duas passadas com invariância de escala."""
    h, w = img_bgr.shape[:2]
    scale = config.resolution_scale(h, w)
    area_scale = scale * scale

    blur_k = config.odd_kernel(5, scale)
    open_k = config.odd_kernel(3, scale)
    close_k = config.odd_kernel(5, scale)
    min_area = max(50, int(round(BASE_MIN_AREA * area_scale)))
    min_fused_area = max(500, int(round(BASE_MIN_FUSED_AREA * area_scale)))

    # --- Etapa 1: Segmentação ---
    t0 = time.perf_counter()
    mask, thresh = segment_mask(
        img_bgr,
        channel=channel,
        blur_ksize=blur_k,
        open_ksize=open_k,
        close_ksize=close_k,
    )
    t_seg = (time.perf_counter() - t0) * 1000.0

    # --- Etapa 2: Passada 1 (rotulagem + descritores + classificação) ---
    t1 = time.perf_counter()
    labels, n = label_components(mask, connectivity=8)
    feats = component_features(labels, n)
    t_feat = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    rejected = set()
    for f in feats:
        if f.area < min_area:
            continue
        cid = classify(
            f.area / area_scale,
            f.aspect_ratio,
            ranges,
            solidity=f.solidity,
            num_holes=f.num_holes,
        )
        if cid == ESPURIO and f.area >= min_fused_area:
            rejected.add(f.label)
    t_clf = (time.perf_counter() - t2) * 1000.0

    # --- Etapa 3: Passada 2 (Watershed seletivo em componentes fundidos) ---
    n_splits = 0
    if split_fused and rejected:
        t_ws0 = time.perf_counter()
        mask = split_fused_components(
            mask,
            labels,
            feats,
            rejected_labels=rejected,
            min_fused_area=min_fused_area,
            min_core_dist=6.5 * scale,
        )
        t_seg += (time.perf_counter() - t_ws0) * 1000.0

        t_f2 = time.perf_counter()
        labels, n = label_components(mask, connectivity=8)
        feats = component_features(labels, n)
        t_feat += (time.perf_counter() - t_f2) * 1000.0
        n_splits = len(rejected)

    # --- Consolidação das detecções finais ---
    t_c2 = time.perf_counter()
    detections: List[Detection] = []
    valid_feats: List[ComponentFeatures] = []
    for f in feats:
        if f.area < min_area:
            continue
        cid = classify(
            f.area / area_scale,
            f.aspect_ratio,
            ranges,
            solidity=f.solidity,
            num_holes=f.num_holes,
        )
        if cid == ESPURIO:
            continue
        valid_feats.append(f)
        detections.append(
            Detection(
                box=component_to_box(f),
                class_id=cid,
                group=group_name(cid),
                area=f.area,
                aspect_ratio=f.aspect_ratio,
                label=f.label,
            )
        )
    t_clf += (time.perf_counter() - t_c2) * 1000.0

    meta = {
        "threshold": thresh,
        "mask": mask,
        "labels": labels,
        "features": valid_feats,
        "n_fused_candidates": n_splits,
        "ms_seg": t_seg,
        "ms_feat": t_feat,
        "ms_clf": t_clf,
        "ms_total": t_seg + t_feat + t_clf,
    }
    return detections, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--test-annotations", type=Path, default=config.ANNOTATIONS_TEST_FILE)
    ap.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    ap.add_argument("--ranges", type=Path, default=config.CLASS_RANGES_FILE)
    ap.add_argument("--channel", default="b")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--no-split-fused", action="store_true", help="desativa Watershed na 2ª passada")
    ap.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = ap.parse_args()

    try:
        ds = load_annotations(args.test_annotations)
        ranges = ClassRanges.load(args.ranges)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERRO ao carregar entradas: {exc}", file=sys.stderr)
        return 1

    image_ids = sorted(
        iid for iid, im in ds.images.items() if (args.images_dir / im.file_name).is_file()
    )
    if not image_ids:
        print(f"ERRO: nenhuma imagem encontrada em {args.images_dir}", file=sys.stderr)
        return 1

    preds_by_image = {}
    timings = []
    for idx, iid in enumerate(image_ids, 1):
        info = ds.images[iid]
        img = load_image(args.images_dir / info.file_name)
        dets, meta = run_two_pass_inference(
            img,
            ranges,
            channel=args.channel,
            split_fused=not args.no_split_fused,
        )
        preds_by_image[iid] = dets
        timings.append(meta)
        if idx % 15 == 0 or idx == len(image_ids):
            print(f"  {idx}/{len(image_ids)} imagens avaliadas", flush=True)

    gts_by_image = {iid: ds.annotations_by_image[iid] for iid in image_ids}
    report = evaluate_dataset_predictions(preds_by_image, gts_by_image, iou_threshold=args.iou)

    loc = report["localization"]
    bg = report["by_group"]
    cnt = report["counting"]

    print(f"\n=== RESULTADOS NO CONJUNTO DE TESTE ({len(image_ids)} imagens, IoU >= {args.iou}) ===")
    print(
        f"Localização pura (qualquer classe): P={loc['precision']:.1%}  "
        f"R={loc['recall']:.1%}  F1={loc['f1']:.1%}  IoU={loc['mean_iou']:.3f}"
    )
    print("\nDetecção por grupo (IoU + grupo correto):")
    print(f"{'Grupo':12s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Precisão':>10s} {'Recall':>8s} {'F1':>8s} {'IoU':>7s}")
    for grp in ("porca", "parafuso", "micro"):
        m = bg[grp]
        print(
            f"{grp:12s} {m['tp']:5d} {m['fp']:5d} {m['fn']:5d} "
            f"{m['precision']:10.1%} {m['recall']:8.1%} {m['f1']:8.1%} {m['mean_iou']:7.3f}"
        )
    mc = bg["macro"]
    print(
        f"{'macro':12s} {'-':>5s} {'-':>5s} {'-':>5s} "
        f"{mc['precision']:10.1%} {mc['recall']:8.1%} {mc['f1']:8.1%} {mc['mean_iou']:7.3f}"
    )
    print(f"\nAcerto da classe exata (1..13) nos TPs: {report['fine_class_accuracy_on_tp']:.1%}")

    print("\nContagem por imagem:")
    for label, key in (("Total", "total"), ("Porcas", "porca"), ("Parafusos", "parafuso")):
        c = cnt[key]
        print(
            f"  {label:9s}: exata={c['exact_match_rate']:.1%} | "
            f"MAE={c['mae']:.2f} peças/img | previsto={c['total_pred']} vs real={c['total_gt']}"
        )

    avg_tot = float(np.mean([t["ms_total"] for t in timings]))
    avg_seg = float(np.mean([t["ms_seg"] for t in timings]))
    avg_feat = float(np.mean([t["ms_feat"] for t in timings]))
    avg_clf = float(np.mean([t["ms_clf"] for t in timings]))
    print(
        f"\nTempo médio por imagem: {avg_tot:.1f} ms "
        f"(seg={avg_seg:.1f} ms, feat={avg_feat:.1f} ms, clf={avg_clf:.1f} ms)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report["timing_ms"] = {
        "total": round(avg_tot, 2),
        "segmentation": round(avg_seg, 2),
        "features": round(avg_feat, 2),
        "classification": round(avg_clf, 2),
    }
    report["config"] = {
        "channel": args.channel,
        "split_fused": not args.no_split_fused,
    }

    metrics_path = args.out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.out_dir / "test_per_image.csv"
    if report["per_image"]:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report["per_image"][0].keys()))
            writer.writeheader()
            writer.writerows(report["per_image"])

    print(f"Salvo em {metrics_path} e {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())