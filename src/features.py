"""Componentes conectados e descritores geométricos, do zero (só NumPy).

Segunda (e última) etapa "do zero" do projeto, junto com o Otsu.

Rotulagem
---------
Em vez de percorrer pixel a pixel (lento em Python para 1920x1440 = 2,7 M de
pixels), a máscara é reduzida a *runs*: trechos horizontais contíguos de
primeiro plano, extraídos de forma vetorizada. Dois runs de linhas vizinhas
pertencem ao mesmo componente se se tocam (por coluna em 4-conectividade, ou
também pela diagonal em 8-conectividade). A fusão é feita com union-find sobre
os runs (alguns milhares, não milhões de pixels).

Os rótulos são 1..n, na ordem em que o primeiro pixel de cada componente
aparece na varredura linha a linha; 0 é o fundo.

Descritores
-----------
Todos calculados de forma vetorizada para todos os componentes de uma vez
(np.bincount), sem laço sobre pixels:

* area, centroide, caixa delimitadora, extent (área / área da caixa);
* perímetro: estimador de Crofton em 4 direções (ver `_perimeters`). Soma a
  borda externa e a dos furos. Erro típico de alguns por cento (até ~6 % em
  retângulos alinhados aos eixos), por isso a circularidade é limitada a 1;
* momentos centrais -> eixos da elipse equivalente, razão de aspecto,
  excentricidade e orientação. As variâncias recebem +1/12 (variância de um
  pixel de lado 1), o que deixa um pixel isolado "redondo" e mantém finita a
  razão de uma linha de 1 pixel de largura.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

_SQRT2 = float(np.sqrt(2.0))


# --------------------------------------------------------------------------
# Rotulagem de componentes conectados
# --------------------------------------------------------------------------
def _find(parent: list, x: int) -> int:
    """Raiz de x no union-find, com compressão de caminho."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def label_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Rotula os componentes conectados de uma máscara binária.

    Parâmetros
    ----------
    mask : array 2D (bool ou numérico); qualquer valor != 0 é primeiro plano.
    connectivity : 4 ou 8.

    Retorna
    -------
    (labels, n): `labels` é int32 com 0 = fundo e 1..n = componentes (ordem de
    varredura linha a linha); `n` é o número de componentes.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("a máscara deve ser um numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"a máscara deve ter 2 dimensões, recebeu {mask.ndim}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity deve ser 4 ou 8, recebeu {connectivity}")

    h, w = mask.shape
    fg = mask != 0
    if h == 0 or w == 0 or not fg.any():
        return np.zeros((h, w), dtype=np.int32), 0

    # --- 1. runs horizontais (vetorizado) ---------------------------------
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = fg
    d = np.diff(padded, axis=1)
    run_row, run_start = np.nonzero(d == 1)          # primeiro pixel do run
    _, run_end_excl = np.nonzero(d == -1)            # fim exclusivo
    run_end = run_end_excl - 1                       # último pixel (inclusivo)
    n_runs = run_row.size

    # --- 2. pares de runs que se tocam em linhas vizinhas -----------------
    # Chaves globais crescentes (linha * BIG + coluna + 2) permitem achar, com
    # searchsorted, o intervalo contíguo de runs da linha de cima que tocam
    # cada run. O deslocamento +2 evita colunas negativas.
    big = w + 4
    start_key = run_row * big + run_start + 2
    end_key = run_row * big + run_end + 2
    off = 1 if connectivity == 8 else 0
    prev_base = (run_row - 1) * big
    lo = np.searchsorted(end_key, prev_base + run_start + 2 - off, side="left")
    hi = np.searchsorted(start_key, prev_base + run_end + 2 + off, side="right")
    cnt = np.maximum(hi - lo, 0)
    total_pairs = int(cnt.sum())

    parent = list(range(n_runs))
    if total_pairs:
        upper = np.repeat(np.arange(n_runs), cnt)
        first = np.cumsum(cnt) - cnt
        lower = np.repeat(lo, cnt) + (np.arange(total_pairs) - np.repeat(first, cnt))
        # --- 3. union-find: a raiz é sempre o menor índice de run ---------
        for a, b in zip(upper.tolist(), lower.tolist()):
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    roots = np.fromiter((_find(parent, i) for i in range(n_runs)), dtype=np.int64, count=n_runs)
    # raiz = menor run do componente => ordem crescente = ordem de varredura
    uniq, run_label = np.unique(roots, return_inverse=True)
    run_label = run_label.astype(np.int32) + 1

    # --- 4. pinta os runs na imagem de rótulos (vetorizado) ---------------
    lengths = run_end - run_start + 1
    offsets = np.cumsum(lengths) - lengths
    flat_start = run_row.astype(np.int64) * w + run_start
    pixels = np.repeat(flat_start - offsets, lengths) + np.arange(int(lengths.sum()))
    labels = np.zeros(h * w, dtype=np.int32)
    labels[pixels] = np.repeat(run_label, lengths)
    return labels.reshape(h, w), int(uniq.size)


# --------------------------------------------------------------------------
# Descritores
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentFeatures:
    """Descritores geométricos de um componente (pixels; ângulos em rad)."""

    label: int
    area: int
    centroid_row: float
    centroid_col: float
    bbox: Tuple[int, int, int, int]  # (r0, c0, r1, c1), fim exclusivo
    extent: float                    # área / área da caixa delimitadora
    perimeter: float
    circularity: float               # 4*pi*A / P^2, limitada a 1
    major_axis_length: float         # elipse com os mesmos momentos de 2ª ordem
    minor_axis_length: float
    aspect_ratio: float              # eixo maior / eixo menor (>= 1)
    eccentricity: float              # sqrt(1 - menor^2 / maior^2), em [0, 1)
    orientation: float               # ângulo do eixo maior com o eixo das colunas
    solidity: float = 1.0            # área / área do fecho convexo, em (0, 1]
    num_holes: int = 0               # nº de furos internos (buracos de fundo)


def _perimeters(labels: np.ndarray, n: int) -> np.ndarray:
    """Perímetro de cada componente pelo estimador de Crofton em 4 direções.

    Conta, em cada direção (horizontal, vertical e as duas diagonais), os
    pares de pixels vizinhos com rótulos diferentes em que um deles pertence
    ao componente. Pela fórmula de Cauchy-Crofton,

        P ~ (pi / 8) * (n_h + n_v + (n_d1 + n_d2) / sqrt(2)),

    onde 1/sqrt(2) é o espaçamento das retas diagonais. Para um círculo o
    resultado é exato no limite; para um quadrado alinhado aos eixos subestima
    cerca de 5 %. Uma moldura de fundo evita perder as bordas da imagem.
    """
    pad = np.pad(labels, 1)
    pairs = (
        (pad[:, :-1], pad[:, 1:], 1.0),                 # horizontal
        (pad[:-1, :], pad[1:, :], 1.0),                 # vertical
        (pad[:-1, :-1], pad[1:, 1:], 1.0 / _SQRT2),     # diagonal \
        (pad[:-1, 1:], pad[1:, :-1], 1.0 / _SQRT2),     # diagonal /
    )
    acc = np.zeros(n + 1, dtype=np.float64)
    for a, b, weight in pairs:
        diff = a != b
        acc += weight * np.bincount(a[diff], minlength=n + 1)
        acc += weight * np.bincount(b[diff], minlength=n + 1)
    return (np.pi / 8.0) * acc[1:]

def _convex_hull_area(sub: np.ndarray) -> float:
    """Área do fecho convexo 2D em NumPy puro (Andrew's monotone chain)."""
    pad = np.pad(sub, 1, constant_values=False)
    border = sub & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    ys, xs = np.nonzero(border)
    if xs.size < 3:
        return float(sub.sum())

    order = np.lexsort((ys, xs))
    pts = list(zip(xs[order].tolist(), ys[order].tolist()))

    def cross(o: tuple, a: tuple, b: tuple) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return float(sub.sum())

    hx = np.array([p[0] for p in hull], dtype=np.float64)
    hy = np.array([p[1] for p in hull], dtype=np.float64)
    poly_area = 0.5 * abs(float(np.dot(hx, np.roll(hy, -1)) - np.dot(hy, np.roll(hx, -1))))
    edge_len = float(np.hypot(np.diff(hx, append=hx[0]), np.diff(hy, append=hy[0])).sum())
    return poly_area + 0.5 * edge_len

def component_features(labels: np.ndarray, n: Optional[int] = None) -> List[ComponentFeatures]:
    """Calcula os descritores de cada componente de uma imagem de rótulos.

    `labels` vem de `label_components` (0 = fundo, 1..n). Se `n` não for
    dado, usa labels.max(). Devolve uma lista com um item por componente, na
    ordem dos rótulos.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D")
    if n is None:
        n = int(labels.max()) if labels.size else 0
    if n == 0:
        return []

    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols].astype(np.int64)
    r = rows.astype(np.float64)
    c = cols.astype(np.float64)

    area = np.bincount(lab, minlength=n + 1)[1:]
    cy = np.bincount(lab, weights=r, minlength=n + 1)[1:] / area
    cx = np.bincount(lab, weights=c, minlength=n + 1)[1:] / area

    # momentos centrais em duas passadas (numericamente mais estável)
    dy = r - cy[lab - 1]
    dx = c - cx[lab - 1]
    var_x = np.bincount(lab, weights=dx * dx, minlength=n + 1)[1:] / area + 1 / 12
    var_y = np.bincount(lab, weights=dy * dy, minlength=n + 1)[1:] / area + 1 / 12
    cov_xy = np.bincount(lab, weights=dx * dy, minlength=n + 1)[1:] / area

    half_sum = (var_x + var_y) / 2
    root = np.sqrt(((var_x - var_y) / 2) ** 2 + cov_xy**2)
    lam1 = half_sum + root           # variância ao longo do eixo maior
    lam2 = half_sum - root           # ... do eixo menor
    major = 4.0 * np.sqrt(lam1)
    minor = 4.0 * np.sqrt(lam2)
    aspect = np.sqrt(lam1 / lam2)
    ecc = np.sqrt(np.clip(1.0 - lam2 / lam1, 0.0, 1.0))
    orient = 0.5 * np.arctan2(2.0 * cov_xy, var_x - var_y)

    # caixa delimitadora
    r0 = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
    c0 = r0.copy()
    r1 = np.full(n + 1, -1, dtype=np.int64)
    c1 = r1.copy()
    np.minimum.at(r0, lab, rows)
    np.minimum.at(c0, lab, cols)
    np.maximum.at(r1, lab, rows)
    np.maximum.at(c1, lab, cols)
    r0, c0, r1, c1 = r0[1:], c0[1:], r1[1:] + 1, c1[1:] + 1
    extent = area / ((r1 - r0) * (c1 - c0))

    perim = _perimeters(labels, n)
    circ = np.minimum(1.0, 4.0 * np.pi * area / perim**2)

    solidities = np.ones(n, dtype=np.float64)
    holes = np.zeros(n, dtype=np.int32)
    for i in range(n):
        r_a, c_a, r_b, c_b = int(r0[i]), int(c0[i]), int(r1[i]), int(c1[i])
        sub = labels[r_a:r_b, c_a:c_b] == (i + 1)

        # Fundo externo tem 1 componente 4-conectado; componentes extras são furos
        _, n_bg = label_components(~np.pad(sub, 1, constant_values=False), connectivity=4)
        holes[i] = max(0, n_bg - 1)

        hull_area = _convex_hull_area(sub)
        if hull_area > 0.0:
            solidities[i] = min(1.0, float(area[i]) / hull_area)

    return [
        ComponentFeatures(
            label=i + 1,
            area=int(area[i]),
            centroid_row=float(cy[i]),
            centroid_col=float(cx[i]),
            bbox=(int(r0[i]), int(c0[i]), int(r1[i]), int(c1[i])),
            extent=float(extent[i]),
            perimeter=float(perim[i]),
            circularity=float(circ[i]),
            major_axis_length=float(major[i]),
            minor_axis_length=float(minor[i]),
            aspect_ratio=float(aspect[i]),
            eccentricity=float(ecc[i]),
            orientation=float(orient[i]),
            solidity=float(solidities[i]),
            num_holes=int(holes[i]),
        )
        for i in range(n)
    ]
