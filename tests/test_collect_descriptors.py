"""Testes do script de coleta de descritores por classe (apoio à Iteração 5).

Cobrem a lógica que decide "que peça é este componente?": casar cada
componente segmentado com as caixas GT por sobreposição de pixels e resumir
os descritores por classe. Imagens sintéticas, sem depender do dataset.
"""
import numpy as np
import pytest

from experiments.collect_descriptors import (
    ESPURIO,
    assign_components,
    summarize,
)
from src.features import label_components


def _canvas(h=200, w=300):
    return np.zeros((h, w), dtype=np.uint8)


def _disk(mask, cy, cx, r):
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 255
    return mask


def _box(h, w, r0, c0, r1, c1):
    g = np.zeros((h, w), dtype=bool)
    g[r0:r1, c0:c1] = True
    return g


def test_componente_dentro_de_uma_caixa_recebe_a_classe_dela():
    m = _disk(_canvas(), 100, 100, 30)
    labels, n = label_components(m)
    gt = [_box(200, 300, 60, 60, 140, 140)]
    out = assign_components(labels, n, gt, [7])
    assert out[1]["class_id"] == 7
    assert out[1]["fused"] is False
    assert out[1]["inside_frac"] == pytest.approx(1.0)


def test_componente_fora_de_qualquer_caixa_e_espurio():
    m = _disk(_canvas(), 100, 100, 30)
    labels, n = label_components(m)
    gt = [_box(200, 300, 0, 200, 50, 280)]
    out = assign_components(labels, n, gt, [7])
    assert out[1]["class_id"] == ESPURIO
    assert out[1]["inside_frac"] == pytest.approx(0.0)


def test_maioria_dentro_conta_minoria_dentro_nao():
    m = _canvas()
    m[50:100, 50:150] = 255                      # 5000 px
    labels, n = label_components(m)
    maioria = [_box(200, 300, 50, 50, 100, 120)]  # cobre 70 %
    minoria = [_box(200, 300, 50, 50, 100, 80)]   # cobre 30 %
    assert assign_components(labels, n, maioria, [3])[1]["class_id"] == 3
    assert assign_components(labels, n, minoria, [3])[1]["class_id"] == ESPURIO


def test_blob_que_cobre_duas_pecas_e_marcado_como_fundido():
    m = _canvas()
    _disk(m, 100, 100, 30)
    _disk(m, 100, 150, 30)                       # discos que se tocam
    labels, n = label_components(m)
    assert n == 1
    gt = [_box(200, 300, 70, 70, 130, 130), _box(200, 300, 70, 120, 130, 180)]
    out = assign_components(labels, n, gt, [9, 11])
    assert out[1]["fused"] is True
    assert out[1]["n_gt"] == 2
    assert out[1]["class_id"] in (9, 11)


def test_componente_pequeno_e_ignorado():
    m = _canvas()
    _disk(m, 50, 50, 5)                          # ~80 px
    _disk(m, 150, 200, 30)
    labels, n = label_components(m)
    out = assign_components(labels, n, [], [], min_area=300)
    assert list(out) == [2]


def test_sem_caixas_tudo_e_espurio():
    labels, n = label_components(_disk(_canvas(), 100, 100, 30))
    out = assign_components(labels, n, [], [])
    assert out[1]["class_id"] == ESPURIO and out[1]["n_gt"] == 0


def test_listas_de_tamanhos_diferentes_falham_alto():
    labels, n = label_components(_disk(_canvas(), 100, 100, 30))
    with pytest.raises(ValueError):
        assign_components(labels, n, [_box(200, 300, 0, 0, 10, 10)], [])


def test_summarize_agrupa_por_classe_com_mediana_e_percentis():
    rows = [
        {"class_id": 7, "fused": False, "area": a, "aspect_ratio": 1.0,
         "circularity": 0.9, "extent": 0.7}
        for a in (100, 200, 300)
    ] + [
        {"class_id": 1, "fused": False, "area": 5000, "aspect_ratio": 12.0,
         "circularity": 0.2, "extent": 0.3}
    ]
    s = summarize(rows, features=("area", "aspect_ratio"))
    assert s[7]["n"] == 3
    assert s[7]["area"]["median"] == pytest.approx(200)
    assert s[7]["area"]["p10"] == pytest.approx(120)
    assert s[7]["area"]["p90"] == pytest.approx(280)
    assert s[1]["aspect_ratio"]["median"] == pytest.approx(12.0)


def test_summarize_exclui_fundidos_por_padrao():
    rows = [
        {"class_id": 7, "fused": False, "area": 100},
        {"class_id": 7, "fused": True, "area": 9999},
    ]
    assert summarize(rows, features=("area",))[7]["n"] == 1
    assert summarize(rows, features=("area",), include_fused=True)[7]["n"] == 2
