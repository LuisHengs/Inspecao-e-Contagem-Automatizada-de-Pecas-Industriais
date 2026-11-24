"""Testes da Iteração 4: componentes conectados e descritores geométricos
(segunda etapa implementada do zero, só NumPy).

Estratégia: formas sintéticas com resposta conhecida (retângulo, disco,
elipse) + paridade com o OpenCV (cv2.connectedComponents e cv2.moments) em
máscaras aleatórias e em um blob irregular.
"""
import cv2
import numpy as np
import pytest

from src.features import component_features, label_components


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------
def _canvas(h=120, w=160):
    return np.zeros((h, w), dtype=np.uint8)


def _rect(mask, r0, c0, h, w):
    mask[r0:r0 + h, c0:c0 + w] = 255
    return mask


def _disk(mask, cy, cx, r):
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 255
    return mask


def _ellipse(mask, cy, cx, a, b):
    """Elipse alinhada aos eixos: semi-eixo a na horizontal, b na vertical."""
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    mask[((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0] = 255
    return mask


def _same_partition(a, b):
    """True se a e b rotulam os mesmos pixels nos mesmos grupos (a menos de
    uma permutação dos rótulos); o fundo (0) deve coincidir."""
    if a.shape != b.shape or not np.array_equal(a == 0, b == 0):
        return False
    fg = a > 0
    pairs = np.unique(np.stack([a[fg], b[fg]], axis=1), axis=0)
    return len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))


# --------------------------------------------------------------------------
# Rotulagem: casos com resposta conhecida
# --------------------------------------------------------------------------
def test_mascara_vazia_nao_tem_componentes():
    labels, n = label_components(_canvas())
    assert n == 0
    assert labels.shape == (120, 160) and not labels.any()


def test_mascara_toda_ligada_e_um_componente():
    labels, n = label_components(np.full((20, 30), 255, np.uint8))
    assert n == 1 and (labels == 1).all()


def test_dois_quadrados_separados():
    m = _rect(_rect(_canvas(), 10, 10, 20, 20), 60, 80, 20, 20)
    labels, n = label_components(m)
    assert n == 2
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
    assert (labels[m == 0] == 0).all()


def test_rotulos_seguem_ordem_de_varredura():
    """O componente cujo primeiro pixel vem antes na varredura linha a linha
    recebe o rótulo menor (resultado determinístico)."""
    m = _rect(_rect(_canvas(), 10, 100, 10, 10), 50, 5, 10, 10)
    labels, _ = label_components(m)
    assert labels[10, 100] == 1 and labels[50, 5] == 2


def test_conectividade_8_vs_4_na_diagonal():
    m = _canvas(10, 10)
    m[2, 2] = m[3, 3] = 255
    assert label_components(m, connectivity=8)[1] == 1
    assert label_components(m, connectivity=4)[1] == 2


def test_forma_em_U_une_ramos_so_na_base():
    """Dois ramos verticais recebem rótulos provisórios distintos e só se
    unem na linha de baixo: força a fusão (union) entre linhas."""
    m = _canvas(30, 30)
    m[2:20, 5:8] = 255
    m[2:20, 15:18] = 255
    m[20:23, 5:18] = 255
    labels, n = label_components(m, connectivity=4)
    assert n == 1


def test_anel_mantem_o_buraco_como_fundo():
    m = _rect(_canvas(40, 40), 5, 5, 30, 30)
    m[12:28, 12:28] = 0
    labels, n = label_components(m)
    assert n == 1
    assert labels[20, 20] == 0


def test_aceita_bool_e_0_255():
    m = _rect(_canvas(), 5, 5, 10, 10)
    a = label_components(m)[0]
    b = label_components(m > 0)[0]
    assert np.array_equal(a, b)


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        label_components(np.zeros((3, 3, 3), np.uint8))
    with pytest.raises(ValueError):
        label_components(_canvas(), connectivity=6)
    with pytest.raises(TypeError):
        label_components([[0, 1], [1, 0]])


# --------------------------------------------------------------------------
# Rotulagem: paridade com o OpenCV
# --------------------------------------------------------------------------
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("density", [0.3, 0.5, 0.65])
def test_paridade_com_cv2_em_mascaras_aleatorias(seed, density, connectivity):
    rng = np.random.default_rng(seed)
    m = ((rng.random((64, 80)) < density) * 255).astype(np.uint8)
    labels, n = label_components(m, connectivity=connectivity)
    n_cv, labels_cv = cv2.connectedComponents(m, connectivity=connectivity)
    assert n == n_cv - 1  # o OpenCV conta o fundo
    assert _same_partition(labels, labels_cv)


def test_rotulos_sao_consecutivos():
    rng = np.random.default_rng(0)
    m = ((rng.random((50, 50)) < 0.4) * 255).astype(np.uint8)
    labels, n = label_components(m)
    assert set(np.unique(labels)) == set(range(n + 1))


# --------------------------------------------------------------------------
# Descritores: retângulo
# --------------------------------------------------------------------------
def test_retangulo_horizontal_40x10():
    m = _rect(_canvas(), 20, 30, 10, 40)  # 10 linhas x 40 colunas
    labels, n = label_components(m)
    (f,) = component_features(labels, n)
    assert f.label == 1
    assert f.area == 400
    assert f.centroid_row == pytest.approx(24.5)
    assert f.centroid_col == pytest.approx(49.5)
    assert f.bbox == (20, 30, 30, 70)  # (r0, c0, r1, c1), fim exclusivo
    assert f.extent == pytest.approx(1.0)
    # razão dos eixos pelos momentos: sqrt(var_x / var_y) = 4 exatamente
    assert f.aspect_ratio == pytest.approx(4.0, rel=1e-9)
    assert f.eccentricity == pytest.approx(np.sqrt(1 - 1 / 16), rel=1e-9)
    assert f.orientation == pytest.approx(0.0, abs=1e-9)


def test_retangulo_vertical_tem_mesma_razao_e_orientacao_90_graus():
    m = _rect(_canvas(), 10, 30, 40, 10)
    labels, n = label_components(m)
    (f,) = component_features(labels, n)
    assert f.aspect_ratio == pytest.approx(4.0, rel=1e-9)
    assert abs(f.orientation) == pytest.approx(np.pi / 2, abs=1e-9)


def test_perimetro_do_retangulo_dentro_de_8_por_cento():
    m = _rect(_canvas(), 20, 30, 10, 40)
    labels, n = label_components(m)
    (f,) = component_features(labels, n)
    assert f.perimeter == pytest.approx(2 * (40 + 10), rel=0.08)


# --------------------------------------------------------------------------
# Descritores: disco e quadrado (circularidade)
# --------------------------------------------------------------------------
def test_disco_e_quase_circular():
    m = _disk(_canvas(200, 200), 100, 100, 40)
    labels, n = label_components(m)
    (f,) = component_features(labels, n)
    assert f.area == pytest.approx(np.pi * 40**2, rel=0.01)
    assert f.perimeter == pytest.approx(2 * np.pi * 40, rel=0.08)
    assert 0.85 <= f.circularity <= 1.0
    assert f.aspect_ratio == pytest.approx(1.0, abs=0.02)
    assert f.eccentricity < 0.15


def _poly(pts, size=400):
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, np.int32)], 255)
    return m


def _hexagon(angle_deg, radius=80, center=200):
    return _poly([
        [center + radius * np.cos(np.radians(angle_deg + 60 * k)),
         center + radius * np.sin(np.radians(angle_deg + 60 * k))]
        for k in range(6)
    ])


def test_quadrado_circularidade_na_faixa_do_estimador():
    """Limitação conhecida do estimador de Crofton em 4 direções: em arestas
    retas longas o perímetro varia com a rotação (medido: circularidade de
    0,76 a 0,88 contra 0,785 ideal). O teste fixa essa faixa e garante que o
    quadrado fique abaixo do disco, que é o que o classificador usa."""
    disco = component_features(*label_components(_disk(_canvas(400, 400), 200, 200, 80)))[0]
    for ang in (0, 15, 30, 45):
        box = cv2.boxPoints(((200, 200), (80, 80), ang))
        (f,) = component_features(*label_components(_poly(box)))
        assert 0.72 <= f.circularity <= 0.92
        assert f.circularity < disco.circularity


@pytest.mark.parametrize("angle", [0, 15, 30])
def test_hexagono_porca_tem_circularidade_estavel_perto_do_ideal(angle):
    """Hexágono regular: circularidade ideal = pi / (2*sqrt(3)) ~ 0,907, e o
    estimador mantém o valor quase igual em qualquer rotação (porca)."""
    (f,) = component_features(*label_components(_hexagon(angle)))
    assert f.circularity == pytest.approx(0.907, abs=0.03)
    assert f.aspect_ratio == pytest.approx(1.0, abs=0.03)


def test_forma_alongada_tem_circularidade_baixa():
    longa = _rect(_canvas(), 20, 10, 8, 120)
    redonda = _disk(_canvas(), 60, 80, 30)
    f_longa = component_features(*label_components(longa))[0]
    f_redonda = component_features(*label_components(redonda))[0]
    assert f_longa.circularity < 0.5 < f_redonda.circularity


def test_circularidade_nunca_passa_de_1():
    for r in (1, 2, 3, 5, 8):
        m = _disk(_canvas(60, 60), 30, 30, r)
        (f,) = component_features(*label_components(m))
        assert f.circularity <= 1.0


# --------------------------------------------------------------------------
# Descritores: elipse, invariâncias e bordas
# --------------------------------------------------------------------------
def test_elipse_comprimentos_dos_eixos():
    m = _ellipse(_canvas(200, 200), 100, 100, a=60, b=20)
    (f,) = component_features(*label_components(m))
    assert f.major_axis_length == pytest.approx(120, rel=0.03)
    assert f.minor_axis_length == pytest.approx(40, rel=0.03)
    assert f.aspect_ratio == pytest.approx(3.0, rel=0.03)


def test_transposta_preserva_razao_e_excentricidade():
    m = _ellipse(_canvas(200, 200), 100, 100, a=60, b=20)
    f1 = component_features(*label_components(m))[0]
    f2 = component_features(*label_components(m.T.copy()))[0]
    assert f1.aspect_ratio == pytest.approx(f2.aspect_ratio, rel=1e-9)
    assert f1.eccentricity == pytest.approx(f2.eccentricity, rel=1e-9)
    assert f1.area == f2.area


def test_translacao_preserva_descritores_exceto_centroide():
    a = _ellipse(_canvas(200, 200), 60, 60, a=40, b=15)
    b = _ellipse(_canvas(200, 200), 120, 130, a=40, b=15)
    fa = component_features(*label_components(a))[0]
    fb = component_features(*label_components(b))[0]
    for campo in ("area", "perimeter", "circularity", "aspect_ratio", "eccentricity"):
        assert getattr(fa, campo) == pytest.approx(getattr(fb, campo), rel=1e-9)
    assert fa.centroid_row != fb.centroid_row


def test_pixel_isolado():
    m = _canvas(10, 10)
    m[4, 5] = 255
    (f,) = component_features(*label_components(m))
    assert f.area == 1
    assert (f.centroid_row, f.centroid_col) == (4.0, 5.0)
    assert f.aspect_ratio == pytest.approx(1.0)
    assert f.eccentricity == pytest.approx(0.0)
    assert np.isfinite(f.circularity)


def test_linha_de_um_pixel_de_largura_tem_razao_finita():
    m = _canvas(10, 30)
    m[5, 3:25] = 255
    (f,) = component_features(*label_components(m))
    assert np.isfinite(f.aspect_ratio) and f.aspect_ratio > 5


def test_anel_inclui_borda_interna_no_perimetro():
    """Convenção documentada: o perímetro soma a borda externa e a dos furos."""
    m = _disk(_canvas(200, 200), 100, 100, 50)
    inner = _canvas(200, 200)
    _disk(inner, 100, 100, 30)
    m[inner > 0] = 0
    (f,) = component_features(*label_components(m))
    assert f.perimeter == pytest.approx(2 * np.pi * (50 + 30), rel=0.08)


# --------------------------------------------------------------------------
# Vários componentes
# --------------------------------------------------------------------------
def test_varios_componentes_consistentes_com_os_rotulos():
    m = _canvas(200, 300)
    _rect(m, 10, 10, 20, 20)
    _disk(m, 100, 150, 25)
    _ellipse(m, 150, 240, 40, 12)
    labels, n = label_components(m)
    feats = component_features(labels, n)
    assert n == 3 and len(feats) == 3
    assert [f.label for f in feats] == [1, 2, 3]
    assert sum(f.area for f in feats) == int((m > 0).sum())
    for f in feats:
        assert f.area == int((labels == f.label).sum())


def test_n_e_opcional():
    m = _rect(_canvas(), 5, 5, 10, 10)
    labels, n = label_components(m)
    assert len(component_features(labels)) == n == 1


def test_sem_componentes_devolve_lista_vazia():
    labels, n = label_components(_canvas())
    assert component_features(labels, n) == []


# --------------------------------------------------------------------------
# Paridade dos momentos com cv2.moments num blob irregular
# --------------------------------------------------------------------------
def test_momentos_batem_com_cv2_moments():
    rng = np.random.default_rng(3)
    m = _canvas(200, 200)
    # blob irregular: união de discos e elipses de tamanhos diferentes
    _disk(m, 90, 80, 30)
    _ellipse(m, 110, 120, 45, 14)
    _disk(m, 85, 125, 18)  # toca a elipse: mantém o blob conexo
    labels, n = label_components(m)
    assert n == 1
    (f,) = component_features(labels, n)

    mo = cv2.moments(m, binaryImage=True)
    assert f.area == mo["m00"]
    assert f.centroid_col == pytest.approx(mo["m10"] / mo["m00"])
    assert f.centroid_row == pytest.approx(mo["m01"] / mo["m00"])
    # variâncias (+1/12 de correção de pixel) -> mesma razão que o OpenCV
    a = mo["mu20"] / mo["m00"] + 1 / 12
    c = mo["mu02"] / mo["m00"] + 1 / 12
    b = mo["mu11"] / mo["m00"]
    tmp = np.sqrt(((a - c) / 2) ** 2 + b**2)
    l1, l2 = (a + c) / 2 + tmp, (a + c) / 2 - tmp
    assert f.aspect_ratio == pytest.approx(np.sqrt(l1 / l2), rel=1e-9)
    assert f.orientation == pytest.approx(0.5 * np.arctan2(2 * b, a - c), abs=1e-9)
