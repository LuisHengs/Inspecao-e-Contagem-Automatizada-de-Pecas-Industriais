"""Testes da Iteração 3 (parte 1): pré-processamento.

Cobre carregamento, redimensionamento, canais de cor, distância ao fundo,
suavização e CLAHE.
"""
import cv2
import numpy as np
import pytest

from src.preprocessing import (
    CHANNELS,
    background_distance,
    enhance_contrast,
    gaussian_blur,
    get_channel,
    load_image,
    median_blur,
    resize_image,
    to_gray,
)

# Cor de "madeira" sintética (BGR) usada como fundo nos testes.
WOOD = (110, 170, 215)


def _cena(h=200, w=300, ruido=0.0, seed=0):
    """Fundo de madeira + retângulo ESCURO + retângulo CLARO (ambos < 50% da área)."""
    img = np.full((h, w, 3), WOOD, np.float32)
    if ruido:
        img += np.random.default_rng(seed).normal(0, ruido, img.shape)
    img[40:90, 40:120] = (40, 40, 40)       # peça escura (tipo parafuso)
    img[110:160, 180:260] = (245, 245, 245)  # peça clara (tipo porca)
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------- carregamento --------------------------------
def test_load_image_roundtrip(tmp_path):
    img = _cena()
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), img)
    out = load_image(p)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.array_equal(out, img)  # PNG é sem perdas


def test_load_image_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nao_existe.png")


def test_load_image_arquivo_corrompido(tmp_path):
    p = tmp_path / "lixo.png"
    p.write_text("isto não é uma imagem")
    with pytest.raises(ValueError):
        load_image(p)


# ----------------------------- redimensionamento ---------------------------
def test_resize_escala_metade():
    out = resize_image(_cena(200, 300), 0.5)
    assert out.shape[:2] == (100, 150)


def test_resize_escala_um_devolve_a_mesma_imagem():
    img = _cena()
    assert resize_image(img, 1.0) is img


@pytest.mark.parametrize("escala", [0, -1, -0.5])
def test_resize_escala_invalida(escala):
    with pytest.raises(ValueError):
        resize_image(_cena(), escala)


# ----------------------------- canais --------------------------------------
def test_canais_sao_2d_uint8_mesmo_tamanho():
    img = _cena()
    for nome in CHANNELS:
        ch = get_channel(img, nome)
        assert ch.ndim == 2 and ch.dtype == np.uint8, nome
        assert ch.shape == img.shape[:2], nome


def test_canal_desconhecido():
    with pytest.raises(ValueError, match="canal"):
        get_channel(_cena(), "xyz")


def test_to_gray_cor_constante():
    img = np.full((10, 10, 3), (50, 100, 150), np.uint8)
    g = to_gray(img)
    assert g.ndim == 2 and len(np.unique(g)) == 1


def test_background_distance_imagem_uniforme_e_zero():
    img = np.full((50, 50, 3), WOOD, np.uint8)
    assert background_distance(img).max() == 0


def test_background_distance_destaca_pecas_escura_e_clara_igualmente():
    """Propriedade central: a distância ao fundo NÃO depende da polaridade.
    É exatamente o que um limiar de cinza de um lado só não consegue."""
    d = background_distance(_cena())
    fundo = d[0:30, :]                   # faixa só de madeira
    escura = d[45:85, 45:115]
    clara = d[115:155, 185:255]
    assert fundo.mean() < 5
    assert escura.mean() > 100
    assert clara.mean() > 100


# ----------------------------- suavização ----------------------------------
def test_gaussian_reduz_ruido_e_preserva_constante():
    ruidosa = _cena(ruido=15)
    suave = gaussian_blur(ruidosa, 5)
    assert suave[0:30].std() < ruidosa[0:30].std()
    cte = np.full((20, 20, 3), 77, np.uint8)
    assert np.array_equal(gaussian_blur(cte, 5), cte)


@pytest.mark.parametrize("k", [0, 2, 4, -3])
def test_blur_rejeita_kernel_par_ou_nao_positivo(k):
    with pytest.raises(ValueError, match="ímpar"):
        gaussian_blur(_cena(), k)
    with pytest.raises(ValueError, match="ímpar"):
        median_blur(_cena(), k)


def test_median_remove_ruido_sal_e_pimenta_isolado():
    img = np.full((30, 30), 100, np.uint8)
    img[10, 10] = 255
    img[20, 5] = 0
    assert np.array_equal(median_blur(img, 3), np.full((30, 30), 100, np.uint8))


# ----------------------------- CLAHE ---------------------------------------
def test_clahe_aumenta_contraste_de_imagem_pouco_contrastada():
    rng = np.random.default_rng(0)
    baixa = (120 + rng.integers(0, 12, (128, 128))).astype(np.uint8)
    out = enhance_contrast(baixa)
    assert out.dtype == np.uint8 and out.shape == baixa.shape
    assert out.std() > baixa.std()


def test_clahe_rejeita_entrada_nao_uint8_2d():
    with pytest.raises(ValueError):
        enhance_contrast(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        enhance_contrast(np.zeros((10, 10), np.float32))
