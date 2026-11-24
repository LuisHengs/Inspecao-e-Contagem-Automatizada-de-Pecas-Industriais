"""Visualização das etapas intermediárias e painéis 'Antes e Depois' (Iteração 7).

Gera figuras prontas para o relatório e slides:
  * 01_original.png             : imagem de entrada;
  * 02_canal_b.png              : canal 'b' do espaço CIE Lab (suprime madeira);
  * 03_otsu_bruto.png           : máscara binária logo após Otsu do zero;
  * 04_morfologia.png           : máscara limpa após abertura e fechamento;
  * 05_componentes_rotulados.png: todos os componentes conectados coloridos;
  * 06_resultado_final.png      : caixas orientadas + classes + placar de contagem;
  * 00_painel_antes_depois.png  : grade 2x2 resumindo o pipeline completo;
  * 07_comparacao_gt.png        : lado a lado [Ground Truth | Predição] (opcional).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

import config
from src.classification import ESPURIO, ClassRanges, classify, group_name
from src.evaluation import Annotation, Detection, box_corners, component_to_box
from src.features import component_features, label_components
from src.preprocessing import gaussian_blur, get_channel
from src.segmentation import CHANNEL_FOREGROUND, clean_mask, otsu_threshold

# Cores BGR por grupo
COLOR_SCREW = (230, 120, 20)    # Azul vibrante
COLOR_NUT = (40, 195, 60)       # Verde vibrante
COLOR_GT = (0, 215, 255)        # Amarelo ouro para caixas GT


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def colorize_labels(labels: np.ndarray, n: int) -> np.ndarray:
    """Pinta cada componente conectado (1..n) com uma cor distinta determinística."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if n <= 0:
        return out
    rng = np.random.default_rng(42)
    palette = np.zeros((n + 1, 3), dtype=np.uint8)
    palette[1:] = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
    out = palette[labels]
    return out


def draw_detections(
    img_bgr: np.ndarray,
    detections: Sequence[Detection],
    show_hud: bool = True,
) -> np.ndarray:
    """Desenha as caixas orientadas, rótulos e placar de contagem na imagem."""
    canvas = img_bgr.copy()
    n_screws = sum(1 for d in detections if d.group == config.GROUP_SCREW)
    n_nuts = sum(1 for d in detections if d.group == config.GROUP_NUT)

    for det in detections:
        cor = COLOR_SCREW if det.group == config.GROUP_SCREW else COLOR_NUT
        pts = np.round(box_corners(det.box)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=cor, thickness=3, lineType=cv2.LINE_AA)

        # Posição do texto próximo ao canto superior da caixa
        top_pt = pts[np.argmin(pts[:, 0, 1]), 0]
        tx = int(np.clip(top_pt[0] - 20, 8, max(8, canvas.shape[1] - 180)))
        ty = int(np.clip(top_pt[1] - 8, 24, canvas.shape[0] - 10))
        label_txt = f"C{det.class_id} ({det.group})"
        cv2.putText(canvas, label_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2, cv2.LINE_AA)

    if show_hud:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (14, 14), (620, 74), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        hud = f"Parafusos: {n_screws}  |  Porcas: {n_nuts}  |  Total: {len(detections)}"
        cv2.putText(canvas, hud, (28, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_ground_truth(img_bgr: np.ndarray, annotations: Sequence[Annotation]) -> np.ndarray:
    """Desenha as caixas orientadas originais (Ground Truth) na imagem."""
    canvas = img_bgr.copy()
    n_screws = sum(1 for a in annotations if config.group_of(a.category_id) == config.GROUP_SCREW)
    n_nuts = sum(1 for a in annotations if config.group_of(a.category_id) == config.GROUP_NUT)

    for ann in annotations:
        grp = config.group_of(ann.category_id)
        pts = np.round(box_corners(ann.box)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_GT, thickness=3, lineType=cv2.LINE_AA)
        top_pt = pts[np.argmin(pts[:, 0, 1]), 0]
        tx = int(np.clip(top_pt[0] - 20, 8, max(8, canvas.shape[1] - 180)))
        ty = int(np.clip(top_pt[1] - 8, 24, canvas.shape[0] - 10))
        txt = f"GT C{ann.category_id} ({grp})"
        cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (14, 14), (660, 74), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    hud = f"GT -> Parafusos: {n_screws}  |  Porcas: {n_nuts}  |  Total: {len(annotations)}"
    cv2.putText(canvas, hud, (28, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_GT, 2, cv2.LINE_AA)
    return canvas


def _add_banner(img: np.ndarray, title: str, width: int = 800) -> np.ndarray:
    """Redimensiona mantendo proporção e adiciona faixa de título no topo."""
    bgr = _to_bgr(img)
    scale = width / float(bgr.shape[1])
    new_h = max(1, int(round(bgr.shape[0] * scale)))
    resized = cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
    banner = np.full((44, width, 3), 30, dtype=np.uint8)
    cv2.putText(banner, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, resized])


def render_pipeline_stages(
    img_bgr: np.ndarray,
    ranges: ClassRanges,
    annotations: Optional[Sequence[Annotation]] = None,
    channel: str = "b",
) -> Dict[str, np.ndarray]:
    """Executa o pipeline passo a passo e retorna o dicionário de imagens visuais."""
    blurred = gaussian_blur(img_bgr, ksize=5)
    ch = get_channel(blurred, channel)
    invert = CHANNEL_FOREGROUND[channel] == "dark"
    t_otsu, mask_raw = otsu_threshold(ch, invert=invert)
    mask_clean = clean_mask(mask_raw, open_ksize=3, close_ksize=5, shape="ellipse")

    labels, n = label_components(mask_clean)
    feats = component_features(labels, n)

    detections: List[Detection] = []
    for f in feats:
        cid = classify(f.area, f.aspect_ratio, ranges)
        if cid == ESPURIO:
            continue
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

    img_labels = colorize_labels(labels, n)
    img_final = draw_detections(img_bgr, detections, show_hud=True)

    # Painel 2x2 "Antes e Depois" para slides
    q1 = _add_banner(img_bgr, "1. Imagem Original (RGB)")
    q2 = _add_banner(ch, f"2. Canal '{channel}' (CIE Lab)")
    q3 = _add_banner(mask_clean, f"3. Otsu (t={t_otsu}) + Morfologia")
    q4 = _add_banner(img_final, "4. Classificacao e Contagem Final")
    painel = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])

    stages: Dict[str, np.ndarray] = {
        "00_painel_antes_depois": painel,
        "01_original": img_bgr.copy(),
        "02_canal_b": ch,
        "03_otsu_bruto": mask_raw,
        "04_morfologia": mask_clean,
        "05_componentes_rotulados": img_labels,
        "06_resultado_final": img_final,
    }

    if annotations is not None:
        left = _add_banner(draw_ground_truth(img_bgr, annotations), "Anotacao Real (Ground Truth)")
        right = _add_banner(img_final, "Predicao do Pipeline Classico")
        stages["07_comparacao_gt"] = np.hstack([left, right])

    return stages


def save_visualizations(
    stages: Dict[str, np.ndarray],
    out_dir: Path,
    prefix: str = "",
) -> List[Path]:
    """Salva todas as imagens geradas em `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for name, img in stages.items():
        fname = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        path = out_dir / fname
        cv2.imwrite(str(path), img)
        saved.append(path)
    return saved