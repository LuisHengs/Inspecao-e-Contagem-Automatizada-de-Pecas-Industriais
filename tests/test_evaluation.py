"""Testes da Iteração 1: parser de anotações orientadas e IoU rotacionado.

Escritos ANTES da implementação (TDD). Cada teste fixa uma decisão de design
ou um erro que já cometemos na análise (ex.: bbox[0] é LINHA, não x).

Convenção do dataset (README do MVTec Screws):
    bbox = [row, col, width, height, phi]
    - width: comprimento paralelo à orientação; height: perpendicular.
    - phi em radianos, sentido matemático positivo em relação ao eixo das
      colunas. phi=0 -> aponta p/ direita; phi=pi/2 -> aponta p/ CIMA da imagem.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from config import NUT_CATEGORY_IDS, SCREW_CATEGORY_IDS, group_of
from src.evaluation import (
    AnnotationError,
    OrientedBox,
    box_area,
    box_corners,
    load_annotations,
    rotated_iou,
)


# --------------------------------------------------------------------------
# Geometria: cantos da caixa (em coordenadas x=coluna, y=linha)
# --------------------------------------------------------------------------
def test_corners_phi_zero_aponta_para_direita():
    b = OrientedBox(row=50, col=100, width=40, height=10, phi=0.0)
    c = box_corners(b)
    assert c.shape == (4, 2)
    assert c[:, 0].min() == pytest.approx(80) and c[:, 0].max() == pytest.approx(120)
    assert c[:, 1].min() == pytest.approx(45) and c[:, 1].max() == pytest.approx(55)


def test_corners_phi_pi_sobre_2_aponta_para_cima():
    """phi=pi/2: eixo longo vertical. Eixo y da imagem cresce para BAIXO."""
    b = OrientedBox(row=50, col=100, width=40, height=10, phi=math.pi / 2)
    c = box_corners(b)
    assert c[:, 0].min() == pytest.approx(95) and c[:, 0].max() == pytest.approx(105)
    assert c[:, 1].min() == pytest.approx(30) and c[:, 1].max() == pytest.approx(70)


def test_phi_positivo_inclina_para_cima_na_imagem():
    """Com phi>0 a ponta direita do eixo longo tem y MENOR (mais acima)."""
    b = OrientedBox(row=50, col=100, width=40, height=2, phi=math.pi / 4)
    c = box_corners(b)
    ponta_direita = c[np.argmax(c[:, 0])]
    assert ponta_direita[1] < 50


def test_centro_e_a_media_dos_cantos():
    b = OrientedBox(row=123.4, col=567.8, width=30, height=12, phi=1.1)
    assert box_corners(b).mean(axis=0) == pytest.approx([567.8, 123.4])


def test_area():
    assert box_area(OrientedBox(0, 0, 40, 10, 0.3)) == pytest.approx(400)


# --------------------------------------------------------------------------
# IoU rotacionado: casos com resposta analítica conhecida
# --------------------------------------------------------------------------
def test_iou_identico_e_um():
    b = OrientedBox(100, 100, 50, 20, 0.7)
    assert rotated_iou(b, b) == pytest.approx(1.0)


def test_iou_disjunto_e_zero():
    a = OrientedBox(100, 100, 10, 10, 0.0)
    b = OrientedBox(100, 500, 10, 10, 0.0)
    assert rotated_iou(a, b) == 0.0


def test_iou_deslocado_meia_largura_e_um_terco():
    # interseção 50, união 150 -> 1/3
    a = OrientedBox(100, 100, 10, 10, 0.0)
    b = OrientedBox(100, 105, 10, 10, 0.0)
    assert rotated_iou(a, b) == pytest.approx(1 / 3, rel=1e-4)


def test_iou_quadrados_girados_45_graus():
    # Dois quadrados de lado s, um girado 45°: interseção = octógono
    # A = 2*s^2*(sqrt2-1); IoU = A/(2*s^2 - A) = 1/sqrt(2)... verificado: 0.7071
    a = OrientedBox(100, 100, 20, 20, 0.0)
    b = OrientedBox(100, 100, 20, 20, math.pi / 4)
    assert rotated_iou(a, b) == pytest.approx(1 / math.sqrt(2), rel=1e-3)


def test_iou_e_simetrico():
    a = OrientedBox(100, 100, 50, 20, 0.2)
    b = OrientedBox(110, 120, 40, 25, 1.0)
    assert rotated_iou(a, b) == pytest.approx(rotated_iou(b, a))


def test_iou_invariante_a_phi_mais_pi():
    """Mesma caixa apontando para o lado oposto continua sendo a mesma caixa."""
    a = OrientedBox(100, 100, 50, 20, 0.4)
    b = OrientedBox(100, 100, 50, 20, 0.4 + math.pi)
    assert rotated_iou(a, b) == pytest.approx(1.0)


def test_iou_troca_de_eixos_equivalente():
    """width/height trocados + phi girado 90° descrevem o mesmo retângulo."""
    a = OrientedBox(100, 100, 50, 20, 0.0)
    b = OrientedBox(100, 100, 20, 50, math.pi / 2)
    assert rotated_iou(a, b) == pytest.approx(1.0)


def test_iou_caixa_degenerada_e_zero():
    a = OrientedBox(100, 100, 0, 10, 0.0)
    b = OrientedBox(100, 100, 10, 10, 0.0)
    assert rotated_iou(a, b) == 0.0


# --------------------------------------------------------------------------
# Parser: contrato e tratamento de erros (JSON sintético mínimo)
# --------------------------------------------------------------------------
def _json_minimo(**override):
    data = {
        "categories": [{"id": 7, "name": "type_007"}, {"id": 1, "name": "type_001"}],
        "images": [
            {"id": 1, "file_name": "screws_001.png", "width": 1920, "height": 1440},
            {"id": 2, "file_name": "screws_002.png", "width": 1920, "height": 1440},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7,
             "bbox": [184.5, 876.3, 55.0, 62.6, 0.0], "area": 3440.0},
            {"id": 11, "image_id": 1, "category_id": 1,
             "bbox": [449.1, 953.3, 716.7, 117.8, -0.2], "area": 84402.0},
        ],
    }
    data.update(override)
    return data


def _escreve(tmp_path, data):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_parser_le_campos_corretamente(tmp_path):
    ds = load_annotations(_escreve(tmp_path, _json_minimo()))
    assert set(ds.categories) == {1, 7}
    assert ds.images[1].file_name == "screws_001.png"
    anns = ds.annotations_by_image[1]
    assert len(anns) == 2
    a = next(x for x in anns if x.ann_id == 10)
    # Regressão do erro de análise: bbox[0] é LINHA (y), bbox[1] é COLUNA (x)
    assert a.box.row == pytest.approx(184.5)
    assert a.box.col == pytest.approx(876.3)
    assert a.category_id == 7


def test_parser_imagem_sem_anotacao_retorna_lista_vazia(tmp_path):
    ds = load_annotations(_escreve(tmp_path, _json_minimo()))
    assert ds.annotations_by_image.get(2, []) == []


def test_parser_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_annotations(tmp_path / "nao_existe.json")


def test_parser_json_invalido(tmp_path):
    p = tmp_path / "ruim.json"
    p.write_text("{ isto nao e json", encoding="utf-8")
    with pytest.raises(AnnotationError):
        load_annotations(p)


def test_parser_bbox_com_4_valores_e_rejeitado(tmp_path):
    """Formato COCO padrão (xywh) NÃO é o deste dataset: deve falhar alto."""
    data = _json_minimo()
    data["annotations"][0]["bbox"] = [10, 20, 30, 40]
    with pytest.raises(AnnotationError, match="bbox"):
        load_annotations(_escreve(tmp_path, data))


def test_parser_categoria_desconhecida(tmp_path):
    data = _json_minimo()
    data["annotations"][0]["category_id"] = 99
    with pytest.raises(AnnotationError, match="categor"):
        load_annotations(_escreve(tmp_path, data))


def test_parser_anotacao_de_imagem_inexistente(tmp_path):
    data = _json_minimo()
    data["annotations"][0]["image_id"] = 999
    with pytest.raises(AnnotationError, match="image"):
        load_annotations(_escreve(tmp_path, data))


def test_parser_chave_obrigatoria_ausente(tmp_path):
    data = _json_minimo()
    del data["images"]
    with pytest.raises(AnnotationError, match="images"):
        load_annotations(_escreve(tmp_path, data))


# --------------------------------------------------------------------------
# Mapeamento classe -> grupo (inferido visualmente; ver config.py)
# --------------------------------------------------------------------------
def test_grupos_cobrem_as_13_classes_sem_sobreposicao():
    assert NUT_CATEGORY_IDS | SCREW_CATEGORY_IDS == set(range(1, 14))
    assert NUT_CATEGORY_IDS & SCREW_CATEGORY_IDS == set()


def test_group_of():
    assert group_of(7) == "porca"
    assert group_of(9) == "porca"
    assert group_of(2) == "parafuso"
    with pytest.raises(KeyError):
        group_of(99)


# --------------------------------------------------------------------------
# Dados reais (pulados se o JSON não estiver disponível)
# Defina MVTEC_JSON=/caminho/mvtec_screws.json ou coloque em data/mvtec_screws/
# --------------------------------------------------------------------------
def _real_json():
    env = os.environ.get("MVTEC_JSON")
    if env and Path(env).is_file():
        return Path(env)
    from config import ANNOTATIONS_FILE
    return ANNOTATIONS_FILE if Path(ANNOTATIONS_FILE).is_file() else None


real = pytest.mark.skipif(_real_json() is None, reason="mvtec_screws.json não encontrado")


@real
def test_dados_reais_estrutura():
    ds = load_annotations(_real_json())
    assert len(ds.images) == 384
    assert set(ds.categories) == set(range(1, 14))
    # O README diz 4426 anotações, mas o JSON tem 4427: não fixamos o número,
    # apenas conferimos consistência interna.
    total = sum(len(v) for v in ds.annotations_by_image.values())
    assert total > 4000
    contagens = [len(v) for v in ds.annotations_by_image.values()]
    assert min(contagens) >= 1 and max(contagens) <= 30


@real
def test_dados_reais_primeira_anotacao_regressao_linha_coluna():
    """Na screws_001 a porca minúscula está em x≈876, y≈184 (descoberto ao
    sobrepor as caixas: o JSON guarda [linha, coluna, ...])."""
    ds = load_annotations(_real_json())
    a = next(x for x in ds.annotations_by_image[1] if x.ann_id == 1001)
    assert a.category_id == 7
    assert a.box.row == pytest.approx(184.5, abs=0.1)
    assert a.box.col == pytest.approx(876.3, abs=0.1)


@real
def test_dados_reais_todas_as_caixas_tem_area_positiva_e_iou_proprio_1():
    ds = load_annotations(_real_json())
    for anns in ds.annotations_by_image.values():
        for a in anns:
            assert box_area(a.box) > 0
            assert rotated_iou(a.box, a.box) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Rasterização de caixas (usada nos experimentos de segmentação)
# --------------------------------------------------------------------------
def test_box_mask_area_aproxima_width_vezes_height():
    from src.evaluation import box_mask
    b = OrientedBox(row=200, col=300, width=120, height=40, phi=0.6)
    m = box_mask(b, (400, 600))
    assert m.dtype == bool and m.shape == (400, 600)
    assert m.sum() == pytest.approx(120 * 40, rel=0.03)


def test_box_mask_centro_dentro_e_canto_da_imagem_fora():
    from src.evaluation import box_mask
    m = box_mask(OrientedBox(50, 50, 20, 10, 0.0), (100, 100))
    assert m[50, 50] and not m[0, 0]


def test_box_mask_caixa_parcialmente_fora_da_imagem_nao_quebra():
    from src.evaluation import box_mask
    m = box_mask(OrientedBox(2, 2, 40, 40, 0.3), (100, 100))
    assert 0 < m.sum() < 40 * 40
