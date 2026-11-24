"""Pré-processamento: carregamento, cores, suavização e realce de contraste.

Contexto do dataset (verificado nas imagens): peças metálicas sobre madeira OSB
clara e texturizada. Isso gera dois problemas que guiam as escolhas deste módulo:
  1. Há peças MAIS ESCURAS que o fundo (parafusos) e peças MAIS CLARAS
     (porcas cromadas): um limiar de brilho de um lado só falha em um dos grupos.
  2. A madeira tem manchas escuras e textura forte, que enganam o canal de
     saturação e o cinza.
Por isso, além dos canais clássicos, oferecemos a "distância ao fundo" no espaço
Lab, que independe da polaridade (escuro/claro).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

# Canais disponíveis para segmentação (todos devolvem uint8 2D).
CHANNELS = ("gray", "L", "b", "S", "V", "dist_lab", "ab_chrom")


def chromatic_wood_contrast(img_bgr: np.ndarray, subsample: int = 4) -> np.ndarray:
    """Combina o canal 'b' com o desvio cromático no canal 'a' (CIE Lab).

    Mantém a supressão dos veios da madeira (como o canal 'b'), mas recupera
    parafusos de latão/dourados cujo valor de 'b' se aproxima da madeira.
    Resultado uint8 onde peças têm valores BAIXOS (mesma polaridade de 'b').
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_med = float(np.median(lab[::subsample, ::subsample, 1]))
    diff_a = np.abs(lab[..., 1] - a_med)
    # Reduz o valor de 'b' onde há desvio cromático em 'a' (latão/cobre)
    score = lab[..., 2] - 1.5 * diff_a
    return np.clip(score, 0, 255).astype(np.uint8)

# --------------------------------------------------------------------------
# Entrada / saída e geometria
# --------------------------------------------------------------------------
def load_image(path: Union[str, Path]) -> np.ndarray:
    """Carrega uma imagem colorida (BGR, uint8).

    FileNotFoundError se o caminho não existe; ValueError se o arquivo existe
    mas não é uma imagem legível (cv2.imread devolve None silenciosamente).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"imagem não encontrada: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"não foi possível decodificar a imagem: {path}")
    return img


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Redimensiona por um fator de escala (>0). scale=1 devolve a própria imagem.

    ATENÇÃO: as anotações estão em pixels da imagem ORIGINAL; ao usar
    scale != 1 é preciso reescalar as caixas antes de avaliar.
    """
    if scale <= 0:
        raise ValueError(f"a escala deve ser positiva, recebeu {scale}")
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


# --------------------------------------------------------------------------
# Canais de cor
# --------------------------------------------------------------------------
def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Converte BGR para tons de cinza (luminância)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def background_distance(img_bgr: np.ndarray, subsample: int = 4) -> np.ndarray:
    """Distância (no espaço Lab) de cada pixel à cor mediana da imagem.

    Ideia: como o fundo (madeira) ocupa a maior parte do quadro, a cor mediana
    estima a cor do fundo. Quanto mais um pixel difere dela, mais provável ser
    peça, seja ela mais escura ou mais clara. Resultado normalizado para 0..255.

    Premissa (limitação): o fundo deve cobrir mais de ~50% da imagem. Em cenas
    com peças ocupando a maioria do quadro a mediana deixaria de ser fundo.
    A mediana é estimada em uma subamostra (1 a cada `subsample` pixels por
    eixo) para ganhar velocidade sem alterar o resultado de forma relevante.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    amostra = lab[::subsample, ::subsample].reshape(-1, 3)
    mediana = np.median(amostra, axis=0)
    dist = np.linalg.norm(lab - mediana, axis=2)
    pico = float(dist.max())
    if pico <= 0.0:
        return np.zeros(dist.shape, np.uint8)
    return np.clip(dist / pico * 255.0, 0, 255).astype(np.uint8)


def get_channel(img_bgr: np.ndarray, name: str) -> np.ndarray:
    """Extrai um canal 2D uint8 pelo nome (ver CHANNELS)."""
    if name == "gray":
        return to_gray(img_bgr)
    if name in ("L", "b"):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        return lab[..., 0] if name == "L" else lab[..., 2]
    if name in ("S", "V"):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return hsv[..., 1] if name == "S" else hsv[..., 2]
    if name == "ab_chrom":
        return chromatic_wood_contrast(img_bgr)
    if name == "dist_lab":
        return background_distance(img_bgr)
    raise ValueError(f"canal desconhecido: '{name}'. Opções: {CHANNELS}")


# --------------------------------------------------------------------------
# Suavização
# --------------------------------------------------------------------------
def _check_odd(ksize: int) -> None:
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"o tamanho do kernel deve ser ímpar e positivo, recebeu {ksize}")


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """Filtro Gaussiano (sigma=0 -> calculado a partir de ksize).

    Reduz ruído de sensor e atenua a textura fina das roscas e da madeira.
    """
    _check_odd(ksize)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Filtro de mediana: remove ruído impulsivo (sal e pimenta) preservando bordas."""
    _check_odd(ksize)
    return cv2.medianBlur(img, ksize)


# --------------------------------------------------------------------------
# Realce de contraste
# --------------------------------------------------------------------------
def enhance_contrast(
    channel: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """CLAHE: equalização adaptativa de histograma com limite de contraste.

    Atua em um canal 2D uint8. O limite de contraste (clip_limit) evita
    amplificar demais o ruído e a textura da madeira.
    """
    if channel.ndim != 2 or channel.dtype != np.uint8:
        raise ValueError("enhance_contrast espera um canal 2D uint8")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(channel)
