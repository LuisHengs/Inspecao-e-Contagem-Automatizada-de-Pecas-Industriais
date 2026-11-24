"""Classificador por faixas de descritores (Iteração 5).

Ideia: cada tipo de peça (as 13 classes do dataset) tem uma "assinatura" em
(área, razão de aspecto). Os dados de treino dão, para cada classe, uma faixa
em (log área, log razão de aspecto): do percentil Q ao 1-Q, alargada por uma
margem. Um componente recebe a classe cuja faixa o contém. Se nenhuma faixa o
contém, é ESPÚRIO (mancha de madeira, sombra, fragmento). Se várias o contêm
(classes de tamanho quase igual, como 5 e 6), vence a de centro mais próximo.

Por que log: os tamanhos das peças variam por ordens de grandeza (de ~2,7 mil a
~44 mil px²) e a variação dentro de cada classe é proporcional, não aditiva.

Por que SÓ área e razão de aspecto: foram os descritores que separaram os
tipos nos dados reais. A circularidade ficou inútil em máscaras reais (porcas
com circularidade 0,33 a 0,49, parafusos 0,04 a 0,66, sobrepostas), porque o
perímetro inclui furos e bordas serrilhadas.

Componentes FUNDIDOS (peças encostadas) não entram no ajuste: seus
descritores não representam uma peça só.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import config

ESPURIO = 0
_LOG_ASPECT_FLOOR = 0.0  # razão de aspecto >= 1, logo log >= 0


def group_name(class_id: int) -> str:
    """'espurio' para a classe 0; 'porca'/'parafuso' pelo mapeamento do config."""
    if class_id == ESPURIO:
        return "espurio"
    return config.group_of(class_id)


def _log_features(area: float, aspect: float) -> Tuple[float, float]:
    return math.log(area), math.log(max(float(aspect), 1.0))


@dataclass
class ClassRanges:
    """Faixas por classe: {class_id: {'log_area': (lo, hi), 'log_aspect': (lo, hi)}}."""

    boxes: Dict[int, Dict[str, Tuple[float, float]]]
    quantile: float
    margin: float
    counts: Dict[int, int] = field(default_factory=dict)

    # --- persistência ------------------------------------------------------
    def to_dict(self) -> dict:
        classes = {}
        for cid, b in sorted(self.boxes.items()):
            la, lr = b["log_area"], b["log_aspect"]
            classes[str(cid)] = {
                "group": group_name(cid),
                "n": self.counts.get(cid, 0),
                # legível (unidades originais), para citar no relatório:
                "area_min": round(math.exp(la[0]), 1),
                "area_max": round(math.exp(la[1]), 1),
                "aspect_min": round(max(1.0, math.exp(lr[0])), 3),
                "aspect_max": round(math.exp(lr[1]), 3),
                # exato (logs), é o que `load` usa:
                "log_area": list(la),
                "log_aspect": list(lr),
            }
        return {
            "features": ["area", "aspect_ratio"],
            "quantile": self.quantile,
            "margin": self.margin,
            "classes": classes,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassRanges":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"arquivo de faixas não encontrado: {path}")
        d = json.loads(path.read_text(encoding="utf-8"))
        boxes, counts = {}, {}
        for cid, c in d["classes"].items():
            boxes[int(cid)] = {
                "log_area": tuple(c["log_area"]),
                "log_aspect": tuple(c["log_aspect"]),
            }
            counts[int(cid)] = int(c.get("n", 0))
        return cls(boxes=boxes, quantile=float(d["quantile"]),
                   margin=float(d["margin"]), counts=counts)


def fit_class_ranges(
    rows: Iterable[Mapping],
    quantile: float = config.CLASSIFIER_QUANTILE,
    margin: float = config.CLASSIFIER_MARGIN,
    min_samples: int = config.CLASSIFIER_MIN_SAMPLES,
) -> ClassRanges:
    """Ajusta uma faixa por classe a partir de linhas com `class_id`, `area`,
    `aspect_ratio` (e opcionalmente `fused`).

    Ignora linhas de classe 0 (espúrios) e fundidas. Falha alto se não sobrar
    nada ou se alguma classe tiver menos de `min_samples` amostras.
    """
    if not 0.0 <= quantile < 0.5:
        raise ValueError(f"quantile deve estar em [0, 0.5), recebeu {quantile}")
    if margin < 0:
        raise ValueError(f"margin deve ser >= 0, recebeu {margin}")

    by_class: Dict[int, list] = {}
    for r in rows:
        cid = int(r["class_id"])
        if cid == ESPURIO or r.get("fused"):
            continue
        by_class.setdefault(cid, []).append(_log_features(float(r["area"]), float(r["aspect_ratio"])))
    if not by_class:
        raise ValueError("nenhuma linha utilizável (classe > 0 e não fundida) para ajustar as faixas")

    few = {c: len(v) for c, v in by_class.items() if len(v) < min_samples}
    if few:
        raise ValueError(f"classes com poucas amostras (mínimo {min_samples}): {few}")

    boxes, counts = {}, {}
    for cid, vals in by_class.items():
        counts[cid] = len(vals)
        boxes[cid] = {}
        for k, name in enumerate(("log_area", "log_aspect")):
            col = sorted(v[k] for v in vals)
            lo = _percentile(col, quantile * 100)
            hi = _percentile(col, (1 - quantile) * 100)
            boxes[cid][name] = (lo - margin, hi + margin)
    return ClassRanges(boxes=boxes, quantile=quantile, margin=margin, counts=counts)


def _percentile(sorted_vals, p: float) -> float:
    """Percentil com interpolação linear (igual ao padrão do NumPy)."""
    n = len(sorted_vals)
    pos = (n - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def classify(
    area: float,
    aspect_ratio: float,
    ranges: ClassRanges,
    solidity: Optional[float] = None,
    num_holes: Optional[int] = None,
) -> int:
    """Devolve o class_id (1..13) da faixa que contém o componente, ou
    ESPURIO (0) se nenhuma contiver."""
    if area <= 0:
        return ESPURIO
    if solidity is not None and solidity < 0.35:
        return ESPURIO

    x = _log_features(area, aspect_ratio)
    best_cid, best_dist = ESPURIO, math.inf

    for cid, box in ranges.boxes.items():
        # Se o componente possui furo interno, prioriza classes de porca
        if num_holes is not None and num_holes >= 1 and group_name(cid) != config.GROUP_NUT:
            continue

        dist = 0.0
        for v, name in zip(x, ("log_area", "log_aspect")):
            lo, hi = box[name]
            if not lo <= v <= hi:
                dist = math.inf
                break
            half = (hi - lo) / 2.0
            dist = max(dist, abs(v - (lo + hi) / 2.0) / half if half > 0 else 0.0)
        if dist < best_dist:
            best_cid, best_dist = cid, dist

    # Fallback: se tinha furo (ex.: reflexo especular) mas não casou com porca,
    # tenta novamente sem restrição de grupo antes de descartar
    if best_cid == ESPURIO and num_holes is not None and num_holes >= 1:
        return classify(area, aspect_ratio, ranges, solidity=solidity, num_holes=None)

    return best_cid
