"""Segmentação: limiarização de Otsu implementada do zero (somente NumPy).

Contém:
  * Otsu do zero em NumPy (a primeira das duas etapas "do zero" do projeto;
    a segunda é a rotulagem + descritores, em src/features.py);
  * morfologia matemática (abertura/fechamento) via OpenCV, por decisão do
    projeto: o foco "do zero" fica no Otsu e nos descritores;
  * `segment_mask`, que encadeia suavização -> canal -> (CLAHE) -> Otsu ->
    morfologia.
A rotulagem de componentes conectados e os descritores estão em src/features.py.

Convenção (idêntica à do OpenCV, para permitir comparação direta):
    classe 0 (fundo)         : pixels <= t
    classe 1 (primeiro plano): pixels >  t   -> 255 na máscara
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.preprocessing import (
    CHANNELS,
    enhance_contrast,
    gaussian_blur,
    get_channel,
)


def _validate_gray_uint8(img: np.ndarray) -> None:
    """Valida que a entrada é uma imagem 2D, não vazia, de 8 bits."""
    if not isinstance(img, np.ndarray):
        raise TypeError("a imagem deve ser um numpy.ndarray")
    if img.dtype != np.uint8:
        raise TypeError(f"a imagem deve ser uint8, recebeu {img.dtype}")
    if img.ndim != 2:
        raise ValueError(
            f"a imagem deve ter 2 dimensões (um único canal), recebeu {img.ndim}"
        )
    if img.size == 0:
        raise ValueError("a imagem está vazia")


def apply_threshold(img: np.ndarray, t: int, invert: bool = False) -> np.ndarray:
    """Binariza a imagem: 255 onde img > t (ou o contrário, se invert=True)."""
    _validate_gray_uint8(img)
    fg = img > t
    if invert:
        fg = ~fg
    return fg.astype(np.uint8) * 255


def otsu_threshold(img: np.ndarray, invert: bool = False) -> Tuple[int, np.ndarray]:
    """Calcula o limiar de Otsu e devolve (limiar, máscara binária 0/255).

    Princípio: escolher o limiar t que maximiza a variância ENTRE as duas
    classes (fundo e primeiro plano). Para cada t possível:

        sigma_b^2(t) = n0 * n1 * (m0 - m1)^2

    onde n0, n1 são as quantidades de pixels de cada classe e m0, m1 as médias
    de intensidade. (Equivale à forma w0*w1*(m0-m1)^2 com probabilidades, só
    que, trabalhando com contagens, evitamos divisões intermediárias.)

    Implementação vetorizada: o histograma (256 posições) é calculado uma vez
    e os n0, n1, m0, m1 de TODOS os limiares saem de somas acumuladas, sem
    laço sobre os pixels. Custo: O(N) para o histograma + O(256) para o resto.

    Casos especiais (mesmo comportamento do OpenCV):
      * em platôs de empate, vale o PRIMEIRO limiar (np.argmax);
      * imagem constante: variância 0 em todo t -> limiar 0.

    Parâmetros
    ----------
    img    : imagem uint8 de 1 canal (qualquer canal: cinza, L, b, distância...).
    invert : se True, o primeiro plano é o lado escuro (pixels <= t -> 255).
    """
    _validate_gray_uint8(img)

    # Histograma de 256 classes (bincount é muito mais rápido que np.histogram).
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    niveis = np.arange(256, dtype=np.float64)

    total = hist.sum()
    soma_total = (niveis * hist).sum()

    n0 = np.cumsum(hist)                    # nº de pixels com valor <= t
    n1 = total - n0                         # nº de pixels com valor  > t
    s0 = np.cumsum(niveis * hist)           # soma das intensidades da classe 0
    s1 = soma_total - s0

    # Limiares com uma das classes vazia não separam nada: variância 0.
    valido = (n0 > 0) & (n1 > 0)
    m0 = np.divide(s0, n0, out=np.zeros(256), where=valido)
    m1 = np.divide(s1, n1, out=np.zeros(256), where=valido)

    sigma_b2 = np.where(valido, n0 * n1 * (m0 - m1) ** 2, 0.0)

    t = int(np.argmax(sigma_b2))            # primeiro máximo em caso de empate
    return t, apply_threshold(img, t, invert=invert)


# --------------------------------------------------------------------------
# Morfologia matemática (OpenCV)
# --------------------------------------------------------------------------
_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def structuring_element(ksize: int, shape: str = "ellipse") -> np.ndarray:
    """Elemento estruturante ksize x ksize ('rect', 'ellipse' ou 'cross').

    A elipse preserva melhor os contornos arredondados (cabeças de parafuso,
    porcas); o retângulo é útil para testes e para estruturas ortogonais.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"o tamanho do kernel deve ser ímpar e positivo, recebeu {ksize}")
    if shape not in _SHAPES:
        raise ValueError(f"forma desconhecida: '{shape}'. Opções: {tuple(_SHAPES)}")
    return cv2.getStructuringElement(_SHAPES[shape], (ksize, ksize))


def _check_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2:
        raise TypeError("a máscara deve ser um array 2D uint8 (0/255)")


def morph_open(mask: np.ndarray, ksize: int = 3, shape: str = "ellipse") -> np.ndarray:
    """Abertura (erosão seguida de dilatação): remove ruídos isolados menores
    que o elemento estruturante, sem encolher os objetos grandes."""
    _check_mask(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuring_element(ksize, shape))


def morph_close(mask: np.ndarray, ksize: int = 5, shape: str = "ellipse") -> np.ndarray:
    """Fechamento (dilatação seguida de erosão): fecha furos e falhas menores
    que o elemento estruturante (ex.: reflexos especulares dentro da peça)."""
    _check_mask(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structuring_element(ksize, shape))


def clean_mask(
    mask: np.ndarray,
    open_ksize: Optional[int] = 3,
    close_ksize: Optional[int] = 5,
    shape: str = "ellipse",
) -> np.ndarray:
    """Abertura (tira ruído externo) e depois fechamento (tapa furos internos).

    A ordem importa: fechar primeiro poderia unir ruído às peças. Passar
    ksize=None (ou 1) desativa a etapa correspondente.
    """
    _check_mask(mask)
    out = mask
    if open_ksize is not None and open_ksize > 1:
        out = morph_open(out, open_ksize, shape)
    if close_ksize is not None and close_ksize > 1:
        out = morph_close(out, close_ksize, shape)
    return out


# --------------------------------------------------------------------------
# Segmentação completa
# --------------------------------------------------------------------------
# Em qual lado do histograma fica a peça em cada canal. Para a madeira clara,
# as peças costumam ser escuras em cinza/L/b/V; na distância ao fundo, são
# sempre os valores altos. (Hipótese a ser confirmada pelo experimento de
# canais; a porca cromada, mais clara que a madeira, é a exceção em brilho.)
CHANNEL_FOREGROUND = {
    "gray": "dark",
    "L": "dark",
    "b": "dark",
    "S": "dark",
    "V": "dark",
    "dist_lab": "bright",
    "ab_chrom": "dark",
}


def segment_mask(
    img_bgr: np.ndarray,
    channel: str = "b",
    blur_ksize: Optional[int] = 5,
    clahe: bool = False,
    open_ksize: Optional[int] = 3,
    close_ksize: Optional[int] = 5,
    shape: str = "ellipse",
) -> Tuple[np.ndarray, int]:
    """Segmenta peças: suavização -> canal -> (CLAHE) -> Otsu -> morfologia.

    Canal padrão = 'b' (eixo amarelo-azul do Lab). Escolhido por experimento
    (experiments/compare_channels.py, 5 imagens): a madeira é muito amarela
    (b alto) e o metal é neutro (b baixo), então os parafusos escuros e as
    porcas cromadas ficam do MESMO lado do histograma. Ponto fraco conhecido:
    parafusos dourados (latão), de cor próxima à da madeira. Reavaliar no
    conjunto completo antes de afirmar no relatório.

    Devolve (máscara 0/255, limiar de Otsu usado). A suavização é aplicada à
    imagem colorida ANTES de extrair o canal, para que ruído de cor não vaze
    para a distância ao fundo.
    """
    if channel not in CHANNELS:
        raise ValueError(f"canal desconhecido: '{channel}'. Opções: {CHANNELS}")

    img = gaussian_blur(img_bgr, blur_ksize) if blur_ksize else img_bgr
    ch = get_channel(img, channel)
    if clahe:
        ch = enhance_contrast(ch)
    invert = CHANNEL_FOREGROUND[channel] == "dark"
    t, mask = otsu_threshold(ch, invert=invert)
    return clean_mask(mask, open_ksize, close_ksize, shape), t

def split_fused_components(
    mask: np.ndarray,
    labels: np.ndarray,
    features: list,
    rejected_labels: set,
    min_fused_area: int = 4000,
    dist_ratio: float = 0.55,
    min_core_dist: float = 6.5,
) -> np.ndarray:
    """Separa componentes fundidos (peças encostadas) via Distance Transform + Watershed.

    Atua EXCLUSIVAMENTE sobre rótulos presentes em `rejected_labels` com
    `area >= min_fused_area`. Peças já classificadas permanecem intactas.
    """
    out_mask = np.zeros_like(mask)
    for feat in features:
        comp = labels == feat.label
        if feat.label not in rejected_labels or feat.area < min_fused_area:
            out_mask[comp] = 255
            continue

        r0, c0, r1, c1 = feat.bbox
        r0p, c0p = max(0, r0 - 2), max(0, c0 - 2)
        r1p, c1p = min(mask.shape[0], r1 + 2), min(mask.shape[1], c1 + 2)
        roi = np.where(comp[r0p:r1p, c0p:c1p], 255, 0).astype(np.uint8)

        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        max_d = float(dist.max())
        if max_d < min_core_dist * 1.3:
            out_mask[comp] = 255
            continue

        # Limiar adaptativo rompe pontes de contato mais finas que os núcleos
        thresh = max(min_core_dist, dist_ratio * max_d)
        sure_fg = np.where(dist >= thresh, 255, 0).astype(np.uint8)
        sure_fg = cv2.morphologyEx(
            sure_fg, cv2.MORPH_OPEN, structuring_element(3, "ellipse")
        )

        n_seeds, markers = cv2.connectedComponents(sure_fg)
        if n_seeds <= 2:
            # Apenas 1 semente encontrada: não há múltiplas peças separáveis
            out_mask[comp] = 255
            continue

        sure_bg = cv2.dilate(roi, structuring_element(3, "ellipse"), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        markers = markers + 1
        markers[unknown == 255] = 0

        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        relief = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
        ws = cv2.watershed(relief, markers)

        # ws == -1 é a linha de crista (corte) entre peças encostadas
        ridge = (ws == -1).astype(np.uint8) * 255
        ridge = cv2.dilate(ridge, structuring_element(3, "cross"))
        split_roi = np.where((ws >= 2) & (roi > 0) & (ridge == 0), 255, 0).astype(np.uint8)
        out_mask[r0p:r1p, c0p:c1p] = np.maximum(
            out_mask[r0p:r1p, c0p:c1p], split_roi
        )

    return out_mask
