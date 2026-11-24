"""Avaliação: parser das anotações do MVTec Screws, geometria e métricas.

Inclui:
  * Iteração 1: parser do JSON em formato "COCO-like" com caixas orientadas,
    geometria das caixas (cantos, área, rasterização) e IoU rotacionado;
  * Iteração 6: conversão de componentes segmentados em caixas orientadas,
    casamento 1-para-1 por IoU rotacionado, métricas de detecção (Precisão,
    Recall, F1, IoU médio) por grupo/classe e métricas de contagem por imagem.

Convenção do dataset (conforme o README do MVTec Screws v1.1):
    bbox = [row, col, width, height, phi]
      row, col : centro da caixa (LINHA = eixo vertical, COLUNA = eixo horizontal)
      width    : comprimento paralelo à orientação da caixa
      height   : comprimento perpendicular à orientação
      phi      : orientação em radianos, sentido matemático positivo medido a
                 partir do eixo das colunas (phi=0 -> direita; phi=pi/2 -> CIMA
                 da imagem; como o eixo y da imagem cresce para baixo, o vetor
                 direção em (x, y) é (cos phi, -sin phi)).
    As coordenadas estão na convenção COCO (origem no canto superior esquerdo
    do primeiro pixel). Para rasterizar em índices de pixel, subtrair 0.5;
    inversamente, ao converter índices de pixel (0..H-1, 0..W-1) para COCO,
    soma-se 0.5.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import config
from src.features import ComponentFeatures


class AnnotationError(ValueError):
    """Erro de conteúdo/estrutura no arquivo de anotações."""


# --------------------------------------------------------------------------
# Estruturas de dados
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OrientedBox:
    """Caixa orientada no formato nativo do dataset (row, col, w, h, phi)."""

    row: float
    col: float
    width: float
    height: float
    phi: float


@dataclass(frozen=True)
class Annotation:
    ann_id: int
    image_id: int
    category_id: int
    box: OrientedBox


@dataclass(frozen=True)
class ImageInfo:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass
class Dataset:
    categories: Dict[int, str]
    images: Dict[int, ImageInfo]
    # Toda imagem aparece como chave (lista vazia se não tiver anotações).
    annotations_by_image: Dict[int, List[Annotation]] = field(default_factory=dict)


@dataclass(frozen=True)
class Detection:
    """Predição de uma peça segmentada e classificada."""

    box: OrientedBox
    class_id: int
    group: str
    area: int = 0
    aspect_ratio: float = 1.0
    label: int = 0


@dataclass(frozen=True)
class MatchPair:
    """Par casado entre uma predição e uma anotação GT."""

    pred_idx: int
    gt_idx: int
    iou: float


@dataclass
class MatchResult:
    """Resultado do casamento 1-para-1 entre predições e anotações de uma imagem."""

    matches: List[MatchPair]
    unmatched_preds: List[int]
    unmatched_gts: List[int]

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.unmatched_preds)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gts)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        if not self.matches:
            return 0.0
        return float(sum(m.iou for m in self.matches) / len(self.matches))


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def box_corners(box: OrientedBox) -> np.ndarray:
    """Retorna os 4 cantos da caixa como array (4, 2) em coordenadas (x, y).

    x = coluna, y = linha. Os cantos saem em ordem de percurso (sem cruzar),
    o que é necessário para a interseção de polígonos convexos.
    """
    # Eixo paralelo à caixa (u) e perpendicular (v), em (x, y) de imagem.
    u = np.array([math.cos(box.phi), -math.sin(box.phi)])
    v = np.array([math.sin(box.phi), math.cos(box.phi)])
    centro = np.array([box.col, box.row])
    hw, hh = box.width / 2.0, box.height / 2.0
    return np.stack([
        centro + hw * u + hh * v,
        centro + hw * u - hh * v,
        centro - hw * u - hh * v,
        centro - hw * u + hh * v,
    ])


def box_area(box: OrientedBox) -> float:
    """Área da caixa (width * height)."""
    return float(box.width * box.height)


def component_to_box(feat: ComponentFeatures) -> OrientedBox:
    """Converte descritores de um componente em caixa orientada (OrientedBox).

    Conversões de convenção:
      * Centroide: `feat.centroid_row/col` estão em índices de pixel (0..H-1);
        soma-se 0.5 para voltar à convenção COCO do dataset.
      * Dimensões: `major_axis_length` e `minor_axis_length` da elipse
        equivalente estimam comprimento e largura da peça.
      * Ângulo: `feat.orientation` é medido no sentido horário a partir do eixo
        das colunas (y cresce para baixo), enquanto `OrientedBox.phi` usa o
        sentido matemático positivo (anti-horário); por isso `phi = -orientation`.
    """
    return OrientedBox(
        row=float(feat.centroid_row + 0.5),
        col=float(feat.centroid_col + 0.5),
        width=float(feat.major_axis_length),
        height=float(feat.minor_axis_length),
        phi=float(-feat.orientation),
    )


def rotated_iou(a: OrientedBox, b: OrientedBox) -> float:
    """IoU (interseção sobre união) entre duas caixas orientadas.

    Usa cv2.intersectConvexConvex para a área de interseção (as caixas são
    retângulos, logo convexas). As áreas individuais são calculadas de forma
    analítica (width*height), que é exata. Os polígonos são transladados para
    perto da origem antes de converter a float32, o que evita perda de
    precisão com coordenadas grandes (~1000 px).
    """
    area_a, area_b = box_area(a), box_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    origem = np.array([(a.col + b.col) / 2.0, (a.row + b.row) / 2.0])
    pa = (box_corners(a) - origem).astype(np.float32)
    pb = (box_corners(b) - origem).astype(np.float32)

    inter, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(inter)
    # Proteção numérica: a interseção nunca excede a menor das áreas.
    inter = max(0.0, min(inter, area_a, area_b))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def box_mask(box: OrientedBox, shape: Tuple[int, int]) -> np.ndarray:
    """Rasteriza a caixa em uma máscara booleana (altura, largura).

    Os cantos estão em coordenadas COCO (origem no canto do primeiro pixel);
    subtrai-se 0.5 para obter índices de pixel. O preenchimento usa precisão
    de 1/16 de pixel (shift=4) para não perder área em peças pequenas.
    """
    m = np.zeros(shape, np.uint8)
    pts = np.round((box_corners(box) - 0.5) * 16.0).astype(np.int32)
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


# --------------------------------------------------------------------------
# Casamento 1-para-1 e métricas de detecção / contagem (Iteração 6)
# --------------------------------------------------------------------------
def match_detections(
    predictions: Sequence[Detection],
    ground_truths: Sequence[Annotation],
    iou_threshold: float = 0.5,
    match_by: str = "group",
) -> MatchResult:
    """Casa predições e anotações GT (1-para-1) maximizando o IoU rotacionado.

    Parâmetros
    ----------
    predictions   : lista de detecções previstas para uma imagem.
    ground_truths : lista de anotações GT da mesma imagem.
    iou_threshold : limiar mínimo de IoU rotacionado para considerar acerto.
    match_by      : critério de compatibilidade de rótulo:
                      * "any"   : avalia apenas localização (ignora classe);
                      * "group" : exige mesmo grupo ('porca' ou 'parafuso');
                      * "class" : exige mesmo category_id (1..13).
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold deve estar em (0, 1], recebeu {iou_threshold}")
    if match_by not in ("any", "group", "class"):
        raise ValueError(f"match_by inválido: '{match_by}'. Use 'any', 'group' ou 'class'")

    n_pred = len(predictions)
    n_gt = len(ground_truths)
    if n_pred == 0 or n_gt == 0:
        return MatchResult(
            matches=[],
            unmatched_preds=list(range(n_pred)),
            unmatched_gts=list(range(n_gt)),
        )

    candidates: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if match_by == "group" and pred.group != config.group_of(gt.category_id):
                continue
            if match_by == "class" and pred.class_id != gt.category_id:
                continue
            iou = rotated_iou(pred.box, gt.box)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    # Ordena por IoU decrescente (desempate determinístico por índice)
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_preds = set()
    used_gts = set()
    matches: List[MatchPair] = []
    for iou, pi, gi in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matches.append(MatchPair(pred_idx=pi, gt_idx=gi, iou=iou))

    unmatched_preds = [i for i in range(n_pred) if i not in used_preds]
    unmatched_gts = [i for i in range(n_gt) if i not in used_gts]
    return MatchResult(
        matches=matches,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
    )


def _pr_f1(tp: int, fp: int, fn: int, ious: Sequence[float]) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def evaluate_dataset_predictions(
    preds_by_image: Dict[int, Sequence[Detection]],
    gts_by_image: Dict[int, Sequence[Annotation]],
    iou_threshold: float = 0.5,
) -> dict:
    """Agrega métricas de detecção, classificação e contagem em um conjunto de imagens.

    Retorna um dicionário estruturado pronto para exibição e serialização JSON:
      * `localization` : métricas de detecção ignorando classe (`match_by="any"`);
      * `by_group`     : métricas para `'porca'`, `'parafuso'`, `'micro'` e `'macro'`;
      * `fine_class_accuracy_on_tp` : acerto da classe exata (1..13) entre os TPs
                                      casados por grupo;
      * `counting`     : métricas de contagem por imagem (total, porca, parafuso):
                         `exact_match_rate`, `mae`, `rmse`, `total_pred`, `total_gt`;
      * `per_image`    : lista com resumo de contagem e P/R/F1 por imagem.
    """
    image_ids = sorted(gts_by_image.keys())
    groups = (config.GROUP_NUT, config.GROUP_SCREW)

    loc_tp = loc_fp = loc_fn = 0
    loc_ious: List[float] = []

    grp_counts = {g: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for g in groups}
    fine_tp_total = 0
    fine_tp_exact = 0

    tot_errors: List[int] = []
    tot_abs_errors: List[int] = []
    tot_sq_errors: List[int] = []
    exact_total_hits = 0
    sum_pred_total = 0
    sum_gt_total = 0

    grp_count_stats = {
        g: {"abs_err": [], "exact": 0, "pred": 0, "gt": 0} for g in groups
    }

    per_image: List[dict] = []

    for iid in image_ids:
        preds = list(preds_by_image.get(iid, []))
        gts = list(gts_by_image[iid])

        # 1. Localização pura
        m_loc = match_detections(preds, gts, iou_threshold=iou_threshold, match_by="any")
        loc_tp += m_loc.tp
        loc_fp += m_loc.fp
        loc_fn += m_loc.fn
        loc_ious.extend(mp.iou for mp in m_loc.matches)

        # 2. Por grupo ('porca' vs 'parafuso')
        img_tp = img_fp = img_fn = 0
        for g in groups:
            p_g = [p for p in preds if p.group == g]
            gt_g = [a for a in gts if config.group_of(a.category_id) == g]
            mg = match_detections(p_g, gt_g, iou_threshold=iou_threshold, match_by="group")
            grp_counts[g]["tp"] += mg.tp
            grp_counts[g]["fp"] += mg.fp
            grp_counts[g]["fn"] += mg.fn
            grp_counts[g]["ious"].extend(mp.iou for mp in mg.matches)
            img_tp += mg.tp
            img_fp += mg.fp
            img_fn += mg.fn

            for mp in mg.matches:
                fine_tp_total += 1
                if p_g[mp.pred_idx].class_id == gt_g[mp.gt_idx].category_id:
                    fine_tp_exact += 1

            # Contagem do grupo nesta imagem
            n_pg, n_gtg = len(p_g), len(gt_g)
            grp_count_stats[g]["pred"] += n_pg
            grp_count_stats[g]["gt"] += n_gtg
            grp_count_stats[g]["abs_err"].append(abs(n_pg - n_gtg))
            if n_pg == n_gtg:
                grp_count_stats[g]["exact"] += 1

        # 3. Contagem total nesta imagem
        n_p, n_gt = len(preds), len(gts)
        diff = n_p - n_gt
        tot_errors.append(diff)
        tot_abs_errors.append(abs(diff))
        tot_sq_errors.append(diff * diff)
        if diff == 0:
            exact_total_hits += 1
        sum_pred_total += n_p
        sum_gt_total += n_gt

        img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0.0
        img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0.0
        img_f1 = (2 * img_prec * img_rec / (img_prec + img_rec)) if (img_prec + img_rec) > 0 else 0.0
        per_image.append({
            "image_id": iid,
            "gt_total": n_gt,
            "pred_total": n_p,
            "count_error": diff,
            "gt_porca": sum(1 for a in gts if config.group_of(a.category_id) == config.GROUP_NUT),
            "pred_porca": sum(1 for p in preds if p.group == config.GROUP_NUT),
            "gt_parafuso": sum(1 for a in gts if config.group_of(a.category_id) == config.GROUP_SCREW),
            "pred_parafuso": sum(1 for p in preds if p.group == config.GROUP_SCREW),
            "tp": img_tp,
            "fp": img_fp,
            "fn": img_fn,
            "precision": img_prec,
            "recall": img_rec,
            "f1": img_f1,
        })

    by_group = {}
    for g in groups:
        c = grp_counts[g]
        by_group[g] = _pr_f1(c["tp"], c["fp"], c["fn"], c["ious"])

    micro_tp = sum(by_group[g]["tp"] for g in groups)
    micro_fp = sum(by_group[g]["fp"] for g in groups)
    micro_fn = sum(by_group[g]["fn"] for g in groups)
    micro_ious = [iou for g in groups for iou in grp_counts[g]["ious"]]
    by_group["micro"] = _pr_f1(micro_tp, micro_fp, micro_fn, micro_ious)
    by_group["macro"] = {
        "precision": float(np.mean([by_group[g]["precision"] for g in groups])),
        "recall": float(np.mean([by_group[g]["recall"] for g in groups])),
        "f1": float(np.mean([by_group[g]["f1"] for g in groups])),
        "mean_iou": float(np.mean([by_group[g]["mean_iou"] for g in groups])),
    }

    n_imgs = max(len(image_ids), 1)
    counting = {
        "total": {
            "exact_match_rate": exact_total_hits / n_imgs,
            "mae": float(np.mean(tot_abs_errors)) if tot_abs_errors else 0.0,
            "rmse": float(np.sqrt(np.mean(tot_sq_errors))) if tot_sq_errors else 0.0,
            "bias": float(np.mean(tot_errors)) if tot_errors else 0.0,
            "total_pred": sum_pred_total,
            "total_gt": sum_gt_total,
        }
    }
    for g in groups:
        st = grp_count_stats[g]
        counting[g] = {
            "exact_match_rate": st["exact"] / n_imgs,
            "mae": float(np.mean(st["abs_err"])) if st["abs_err"] else 0.0,
            "total_pred": st["pred"],
            "total_gt": st["gt"],
        }

    return {
        "n_images": len(image_ids),
        "iou_threshold": iou_threshold,
        "localization": _pr_f1(loc_tp, loc_fp, loc_fn, loc_ious),
        "by_group": by_group,
        "fine_class_accuracy_on_tp": (
            fine_tp_exact / fine_tp_total if fine_tp_total > 0 else 0.0
        ),
        "counting": counting,
        "per_image": per_image,
    }


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def _require(d: dict, key: str, where: str):
    if key not in d:
        raise AnnotationError(f"chave obrigatória ausente em {where}: '{key}'")
    return d[key]


def load_annotations(path: Union[str, Path]) -> Dataset:
    """Lê um JSON do MVTec Screws (completo, train, val ou test).

    Valida a estrutura e levanta erros explícitos em vez de deixar o pipeline
    falhar mais adiante com uma mensagem obscura:
      * FileNotFoundError  : arquivo não existe;
      * AnnotationError    : JSON inválido, chave ausente, bbox que não tem 5
                             valores, categoria ou imagem referenciada que não
                             existe.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"arquivo de anotações não encontrado: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"JSON inválido em {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise AnnotationError("a raiz do JSON deve ser um objeto")

    categories = {
        int(_require(c, "id", "categories")): str(c.get("name", ""))
        for c in _require(raw, "categories", "JSON")
    }

    images: Dict[int, ImageInfo] = {}
    for im in _require(raw, "images", "JSON"):
        iid = int(_require(im, "id", "images"))
        images[iid] = ImageInfo(
            image_id=iid,
            file_name=str(_require(im, "file_name", "images")),
            width=int(im.get("width", 0)),
            height=int(im.get("height", 0)),
        )

    by_image: Dict[int, List[Annotation]] = {iid: [] for iid in images}
    for ann in _require(raw, "annotations", "JSON"):
        aid = _require(ann, "id", "annotations")
        bbox = _require(ann, "bbox", f"anotação {aid}")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 5:
            n = len(bbox) if isinstance(bbox, (list, tuple)) else "?"
            raise AnnotationError(
                f"anotação {aid}: bbox deve ter 5 valores "
                f"(row, col, width, height, phi); recebeu {n}"
            )
        try:
            row, col, w, h, phi = (float(v) for v in bbox)
        except (TypeError, ValueError) as exc:
            raise AnnotationError(f"anotação {aid}: bbox não numérico") from exc

        cid = int(_require(ann, "category_id", f"anotação {aid}"))
        if cid not in categories:
            raise AnnotationError(f"anotação {aid}: categoria desconhecida {cid}")
        iid = int(_require(ann, "image_id", f"anotação {aid}"))
        if iid not in images:
            raise AnnotationError(f"anotação {aid}: referencia image_id inexistente {iid}")

        by_image[iid].append(
            Annotation(int(aid), iid, cid, OrientedBox(row, col, w, h, phi))
        )

    return Dataset(categories=categories, images=images, annotations_by_image=by_image)