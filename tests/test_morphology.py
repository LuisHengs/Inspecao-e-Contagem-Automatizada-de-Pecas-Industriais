"""Testes da Iteração 3 (parte 2): morfologia e função de segmentação completa.

A morfologia usa OpenCV (decisão do projeto: o "do zero" fica com o Otsu e os
descritores). Os testes verificam PROPRIEDADES da morfologia matemática, não só
exemplos, para garantir que parâmetros e polaridade estão corretos.
"""
import numpy as np
import pytest

from src.segmentation import (
    clean_mask,
    morph_close,
    morph_open,
    segment_mask,
    structuring_element,
)


def _bloco(h=60, w=60, y=15, x=15, lado=30):
    m = np.zeros((h, w), np.uint8)
    m[y:y + lado, x:x + lado] = 255
    return m


# ----------------------------- elemento estruturante -----------------------
def test_structuring_element_formas():
    assert structuring_element(5, "rect").sum() == 25
    el = structuring_element(5, "ellipse")
    assert el.shape == (5, 5) and 0 < el.sum() < 25


@pytest.mark.parametrize("k", [0, 2, 4, -1])
def test_kernel_par_ou_nao_positivo_e_rejeitado(k):
    with pytest.raises(ValueError, match="ímpar"):
        structuring_element(k)


def test_forma_desconhecida():
    with pytest.raises(ValueError, match="forma"):
        structuring_element(3, "triangulo")


def test_mascara_deve_ser_uint8():
    with pytest.raises(TypeError):
        morph_open(np.zeros((10, 10), np.float32), 3)


# ----------------------------- abertura ------------------------------------
def test_abertura_remove_ruido_isolado_e_preserva_bloco_grande():
    m = _bloco()
    ruido = m.copy()
    ruido[2, 2] = 255
    ruido[50, 55] = 255
    out = morph_open(ruido, 3, "rect")
    assert np.array_equal(out, m)


def test_abertura_e_anti_extensiva_e_idempotente():
    rng = np.random.default_rng(0)
    m = ((rng.random((80, 80)) > 0.55) * 255).astype(np.uint8)
    o = morph_open(m, 3)
    assert np.all(o[m == 0] == 0)                       # open(X) ⊆ X
    assert np.array_equal(morph_open(o, 3), o)          # open(open(X)) = open(X)


# ----------------------------- fechamento ----------------------------------
def test_fechamento_preenche_furo_pequeno():
    m = _bloco()
    furado = m.copy()
    furado[28:31, 28:31] = 0                            # furo 3x3 (reflexo/brilho)
    assert np.array_equal(morph_close(furado, 5, "rect"), m)


def test_fechamento_e_extensivo_e_idempotente():
    rng = np.random.default_rng(1)
    m = ((rng.random((80, 80)) > 0.55) * 255).astype(np.uint8)
    c = morph_close(m, 3)
    assert np.all(c[m == 255] == 255)                   # X ⊆ close(X)
    assert np.array_equal(morph_close(c, 3), c)


def test_fechamento_nao_une_objetos_distantes():
    m = np.zeros((40, 100), np.uint8)
    m[10:30, 10:40] = 255
    m[10:30, 50:80] = 255                               # vão de 10 px
    out = morph_close(m, 5, "rect")                     # kernel menor que o vão
    assert out[20, 45] == 0


def test_clean_mask_abre_depois_fecha():
    m = _bloco()
    sujo = m.copy()
    sujo[2, 2] = 255                                    # ruído externo
    sujo[28:31, 28:31] = 0                              # furo interno
    out = clean_mask(sujo, open_ksize=3, close_ksize=5, shape="rect")
    assert np.array_equal(out, m)


def test_clean_mask_kernel_1_ou_none_e_identidade():
    m = _bloco()
    assert np.array_equal(clean_mask(m, open_ksize=1, close_ksize=1), m)
    assert np.array_equal(clean_mask(m, open_ksize=None, close_ksize=None), m)


# ----------------------------- segment_mask (pipeline) ---------------------
WOOD = (110, 170, 215)


def _cena(ruido=6.0):
    img = np.full((200, 300, 3), WOOD, np.float32)
    img += np.random.default_rng(0).normal(0, ruido, img.shape)
    img[40:90, 40:120] = (40, 40, 40)
    img[110:160, 180:260] = (245, 245, 245)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_segment_mask_dist_lab_encontra_peca_escura_e_clara():
    img = _cena()
    mask, t = segment_mask(img, channel="dist_lab")
    assert mask.dtype == np.uint8 and mask.shape == img.shape[:2]
    pm = mask > 0
    gt = np.zeros(mask.shape, bool)
    gt[40:90, 40:120] = True        # peça escura
    gt[110:160, 180:260] = True     # peça clara
    # Desfoque + elemento elíptico consomem ~1 px de borda, então medimos o IoU
    # global e a cobertura do INTERIOR (sem 3 px de borda), não a janela exata.
    iou = (pm & gt).sum() / (pm | gt).sum()
    assert iou > 0.93
    for (y0, y1, x0, x1) in [(43, 87, 43, 117), (113, 157, 183, 257)]:
        assert pm[y0:y1, x0:x1].mean() > 0.99      # interior de cada peça
    assert (pm & ~gt).sum() / (~gt).sum() < 0.01   # quase nenhum falso positivo
    assert 0 <= t <= 255


def test_segment_mask_canal_invalido():
    with pytest.raises(ValueError, match="canal"):
        segment_mask(_cena(), channel="xyz")


def test_segment_mask_sem_morfologia_e_sem_blur_funciona():
    mask, _ = segment_mask(_cena(ruido=0.0), channel="dist_lab",
                           blur_ksize=None, open_ksize=None, close_ksize=None)
    assert (mask[40:90, 40:120] > 0).mean() > 0.99
