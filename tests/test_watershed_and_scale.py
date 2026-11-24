"""Testes unitários das melhorias de segmentação, topologia e escala:
  * Contraste cromático madeira vs. latão (`chromatic_wood_contrast` / `ab_chrom`);
  * Separação seletiva de peças encostadas via Distance Transform + Watershed;
  * Solidez e contagem de furos internos (`num_holes`) em NumPy puro;
  * Regras topológicas no classificador (`classify`);
  * Escalonamento proporcional de kernels e áreas por resolução.
"""
import cv2
import numpy as np
import pytest

import config
from src.classification import ESPURIO, classify, fit_class_ranges
from src.features import component_features, label_components
from src.preprocessing import chromatic_wood_contrast, get_channel
from src.segmentation import split_fused_components


def _canvas(h=200, w=300):
    return np.zeros((h, w), dtype=np.uint8)


def _disk(mask, cy, cx, r, val=255):
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val
    return mask


def _synthetic_ranges():
    rng = np.random.default_rng(42)
    rows = []
    # Classe 7: porca pequena (área ~2700, aspect ~1.02)
    for _ in range(40):
        rows.append({
            "class_id": 7,
            "fused": False,
            "area": float(2700 * np.exp(rng.normal(0, 0.04))),
            "aspect_ratio": float(max(1.0, 1.02 * np.exp(rng.normal(0, 0.02)))),
        })
    # Classe 12: parafuso pequeno (área ~3000, aspect ~5.4)
    for _ in range(40):
        rows.append({
            "class_id": 12,
            "fused": False,
            "area": float(3000 * np.exp(rng.normal(0, 0.04))),
            "aspect_ratio": float(max(1.0, 5.4 * np.exp(rng.normal(0, 0.04)))),
        })
    return fit_class_ranges(rows)


# --------------------------------------------------------------------------
# 1. Canal cromático composto (ab_chrom) para parafusos de latão/dourados
# --------------------------------------------------------------------------
def test_ab_chrom_aumenta_contraste_de_latao_sobre_madeira():
    """Parafuso dourado com 'b' próximo ao da madeira ganha separação via canal 'a'."""
    wood = np.full((120, 200, 3), (70, 165, 210), dtype=np.uint8)
    img = wood.copy()
    img[40:80, 50:150] = (50, 155, 225)  # tom amarelado/avermelhado (latão)

    ch_b = get_channel(img, "b")
    ch_ab = get_channel(img, "ab_chrom")
    assert np.array_equal(ch_ab, chromatic_wood_contrast(img))

    gap_b = float(ch_b[10, 10]) - float(ch_b[60, 100])
    gap_ab = float(ch_ab[10, 10]) - float(ch_ab[60, 100])
    assert gap_ab > gap_b


# --------------------------------------------------------------------------
# 2. Watershed seletivo em componentes fundidos
# --------------------------------------------------------------------------
def test_watershed_separa_dois_objetos_encostados_rejeitados():
    """Dois discos grandes se tocando formam 1 componente conectado; após
    split_fused_components devem voltar a ser 2 componentes separados."""
    m = _canvas(200, 300)
    _disk(m, 100, 110, 42)
    _disk(m, 100, 186, 42)  # centros a 76 px -> sobreposição de 8 px

    labels, n = label_components(m, connectivity=8)
    assert n == 1
    feats = component_features(labels, n)

    split_m = split_fused_components(
        m, labels, feats, rejected_labels={1}, min_fused_area=2000
    )
    _, n_after = label_components(split_m, connectivity=8)
    assert n_after == 2


def test_watershed_nao_altera_componentes_ja_aceitos():
    """Componentes fora de `rejected_labels` permanecem pixel a pixel idênticos."""
    m = _canvas(200, 300)
    _disk(m, 100, 110, 42)
    _disk(m, 100, 186, 42)

    labels, n = label_components(m, connectivity=8)
    feats = component_features(labels, n)

    out = split_fused_components(
        m, labels, feats, rejected_labels=set(), min_fused_area=2000
    )
    assert np.array_equal(out, m)


# --------------------------------------------------------------------------
# 3. Topologia (num_holes) e Solidez (solidity) em NumPy puro
# --------------------------------------------------------------------------
def test_anel_possui_um_furo_e_disco_solido_possui_zero():
    m = _canvas(160, 300)
    _disk(m, 80, 70, 35)             # disco maciço (parafuso/cabeça)
    _disk(m, 80, 210, 35)            # anel externo (porca)
    _disk(m, 80, 210, 14, val=0)     # furo central

    labels, n = label_components(m)
    feats = component_features(labels, n)
    assert n == 2

    solid_disk, nut_ring = feats[0], feats[1]
    assert solid_disk.num_holes == 0
    assert solid_disk.solidity > 0.94

    assert nut_ring.num_holes == 1
    assert 0.75 < nut_ring.solidity < 0.90


def test_forma_concava_irregular_tem_solidez_baixa():
    """Manchas irregulares em U/L têm solidez bem abaixo de 0.5."""
    m = _canvas(80, 80)
    m[10:70, 10:18] = 255
    m[10:70, 62:70] = 255
    m[62:70, 10:70] = 255

    (f,) = component_features(*label_components(m))
    assert f.num_holes == 0
    assert f.solidity < 0.50


def test_classificador_filtra_baixa_solidez_e_prioriza_porca_com_furo():
    ranges = _synthetic_ranges()

    # Solidez muito baixa -> descartado mesmo com área/aspecto de parafuso
    assert classify(3000.0, 5.4, ranges, solidity=0.25, num_holes=0) == ESPURIO

    # Componente com furo interno dentro da faixa da porca (classe 7)
    assert classify(2700.0, 1.02, ranges, solidity=0.83, num_holes=1) == 7

    # Fallback: se tiver furo (ex.: reflexo) mas área/aspecto for exclusivamente
    # de parafuso (classe 12), não descarta falsamente
    assert classify(3000.0, 5.4, ranges, solidity=0.90, num_holes=1) == 12


# --------------------------------------------------------------------------
# 4. Invariância de escala por resolução
# --------------------------------------------------------------------------
def test_helpers_de_escala_por_resolucao():
    assert config.resolution_scale(1440, 1920) == pytest.approx(1.0)
    assert config.resolution_scale(720, 960) == pytest.approx(0.5)
    assert config.resolution_scale(2880, 3840) == pytest.approx(2.0)

    assert config.odd_kernel(5, 1.0) == 5
    assert config.odd_kernel(5, 0.5) == 3
    assert config.odd_kernel(5, 1.4) == 7
    # Garante retorno sempre ímpar e >= 3
    for s in (0.2, 0.75, 1.25, 1.5, 2.3):
        k = config.odd_kernel(5, s)
        assert k >= 3 and k % 2 == 1