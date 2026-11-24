"""Testes da Iteração 2: limiarização de Otsu implementada do zero (NumPy).

Referência: cv2.threshold(..., THRESH_BINARY + THRESH_OTSU).
Convenção (idêntica à do OpenCV): classe 0 = pixels <= t (fundo),
classe 1 = pixels > t (primeiro plano, 255 na máscara).
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.segmentation import apply_threshold, otsu_threshold


def _cv_otsu(img):
    t, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), m


# --------------------------------------------------------------------------
# Casos com resposta conhecida
# --------------------------------------------------------------------------
def test_bimodal_limiar_separa_perfeitamente_os_dois_modos():
    """Entre os dois modos há um 'vale' sem pixels; qualquer t nele é ótimo
    (a variância entre classes é igual ao longo do platô) e o Otsu devolve o
    primeiro. Por isso o critério correto NÃO é 't no meio', e sim: nenhuma
    amostra do modo baixo fica acima de t e nenhuma do alto fica abaixo."""
    rng = np.random.default_rng(0)
    baixo = np.clip(rng.normal(60, 8, 5000), 0, 255).astype(np.uint8)
    alto = np.clip(rng.normal(180, 8, 5000), 0, 255).astype(np.uint8)
    img = np.concatenate([baixo, alto]).reshape(100, 100)
    t, mask = otsu_threshold(img)
    assert baixo.max() <= t < alto.min()
    esperado = np.concatenate([np.zeros(5000), np.full(5000, 255)]).astype(np.uint8)
    assert np.array_equal(mask.ravel(), esperado)


def test_dois_niveis_usa_primeiro_limiar_do_platô_como_o_opencv():
    img = np.where(np.arange(400).reshape(20, 20) % 2 == 0, 50, 200).astype(np.uint8)
    t, mask = otsu_threshold(img)
    t_ref, mask_ref = _cv_otsu(img)
    assert t == t_ref == 50
    assert np.array_equal(mask, mask_ref)


def test_imagem_constante_replica_opencv():
    for valor in (0, 100, 255):
        img = np.full((20, 20), valor, np.uint8)
        t, mask = otsu_threshold(img)
        t_ref, mask_ref = _cv_otsu(img)
        assert t == t_ref
        assert np.array_equal(mask, mask_ref)


def test_mascara_e_binaria_uint8_0_255():
    img = np.random.default_rng(1).integers(0, 256, (50, 50), dtype=np.uint8)
    _, mask = otsu_threshold(img)
    assert mask.dtype == np.uint8 and mask.shape == img.shape
    assert set(np.unique(mask)) <= {0, 255}


def test_polaridade_pixel_maior_que_limiar_e_255():
    img = np.array([[10, 10, 200, 200]], np.uint8)
    t, mask = otsu_threshold(img)
    assert mask.tolist() == [[0, 0, 255, 255]]
    assert apply_threshold(img, t).tolist() == [[0, 0, 255, 255]]


def test_invert_troca_a_polaridade():
    img = np.array([[10, 10, 200, 200]], np.uint8)
    _, mask = otsu_threshold(img, invert=True)
    assert mask.tolist() == [[255, 255, 0, 0]]


def test_apply_threshold_limiar_manual():
    img = np.array([[0, 99, 100, 101, 255]], np.uint8)
    assert apply_threshold(img, 100).tolist() == [[0, 0, 0, 255, 255]]


# --------------------------------------------------------------------------
# Paridade com o OpenCV em muitas distribuições (limiar ±1, aqui esperamos ==)
# --------------------------------------------------------------------------
def test_paridade_com_opencv_em_300_imagens_aleatorias():
    rng = np.random.default_rng(42)
    maior_dif = 0
    for _ in range(300):
        n_modos = rng.integers(2, 4)
        partes = [
            rng.normal(rng.uniform(20, 235), rng.uniform(3, 30), rng.integers(500, 4000))
            for _ in range(n_modos)
        ]
        dados = np.clip(np.concatenate(partes), 0, 255).astype(np.uint8)
        lado = int(np.sqrt(dados.size))
        img = dados[: lado * lado].reshape(lado, lado)
        t, _ = otsu_threshold(img)
        t_ref, _ = _cv_otsu(img)
        maior_dif = max(maior_dif, abs(t - t_ref))
    assert maior_dif <= 1


# --------------------------------------------------------------------------
# Validação de entrada
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.int32])
def test_rejeita_dtype_diferente_de_uint8(dtype):
    with pytest.raises(TypeError):
        otsu_threshold(np.zeros((10, 10), dtype))


def test_rejeita_imagem_colorida():
    with pytest.raises(ValueError, match="2"):
        otsu_threshold(np.zeros((10, 10, 3), np.uint8))


def test_rejeita_imagem_vazia():
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((0, 0), np.uint8))


# --------------------------------------------------------------------------
# Desempenho (limite folgado: serve para pegar regressão grosseira, p.ex. loop em Python)
# --------------------------------------------------------------------------
def test_desempenho_imagem_1920x1440():
    img = np.random.default_rng(3).integers(0, 256, (1440, 1920), dtype=np.uint8)
    t0 = time.perf_counter()
    otsu_threshold(img)
    assert time.perf_counter() - t0 < 1.0


# --------------------------------------------------------------------------
# Imagens reais do dataset (puladas se a pasta não existir)
# --------------------------------------------------------------------------
def _images_dir():
    from config import IMAGES_DIR
    return IMAGES_DIR if Path(IMAGES_DIR).is_dir() else None


real = pytest.mark.skipif(_images_dir() is None, reason="pasta images/ não encontrada")


@real
@pytest.mark.parametrize("nome", ["screws_001.png", "screws_041.png", "screws_042.png",
                                  "screws_043.png", "screws_044.png"])
def test_paridade_em_imagens_reais_canal_cinza(nome):
    caminho = _images_dir() / nome
    if not caminho.is_file():
        pytest.skip(f"{nome} ausente")
    gray = cv2.cvtColor(cv2.imread(str(caminho)), cv2.COLOR_BGR2GRAY)
    t, _ = otsu_threshold(gray)
    t_ref, _ = _cv_otsu(gray)
    assert abs(t - t_ref) <= 1
