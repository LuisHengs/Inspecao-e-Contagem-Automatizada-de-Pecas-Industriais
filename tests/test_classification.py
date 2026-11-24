"""Testes da Iteração 5: classificador por faixas de descritores.

Cada tipo de peça ocupa uma faixa em (log área, log razão de aspecto),
estimada dos dados de treino com cortes por percentil. Um componente recebe a
classe cuja faixa o contém; se nenhuma contém, é espúrio (classe 0).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import config
from src.classification import (
    ESPURIO,
    ClassRanges,
    classify,
    fit_class_ranges,
    group_name,
)


def _rows(class_id, n, area, aspect, spread=0.05, seed=0, fused=False):
    """n linhas sintéticas de uma classe, com variação log-normal pequena."""
    rng = np.random.default_rng(seed + class_id)
    return [
        {
            "class_id": class_id,
            "fused": fused,
            "area": float(area * np.exp(rng.normal(0, spread))),
            "aspect_ratio": float(max(1.0, aspect * np.exp(rng.normal(0, spread)))),
        }
        for _ in range(n)
    ]


def _train():
    return (
        _rows(7, 60, 2700, 1.02)      # porca pequena
        + _rows(9, 60, 30000, 1.02)   # porca grande
        + _rows(1, 60, 31000, 12.0)   # parafuso longo
        + _rows(12, 60, 3100, 5.4)    # parafuso pequeno
    )


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
def test_fit_cria_uma_faixa_por_classe_presente():
    r = fit_class_ranges(_train())
    assert set(r.boxes) == {1, 7, 9, 12}


def test_classifica_o_centro_de_cada_classe_corretamente():
    r = fit_class_ranges(_train())
    assert classify(2700, 1.02, r) == 7
    assert classify(30000, 1.02, r) == 9
    assert classify(31000, 12.0, r) == 1
    assert classify(3100, 5.4, r) == 12


def test_ponto_fora_de_todas_as_faixas_e_espurio():
    r = fit_class_ranges(_train())
    assert classify(8000, 3.0, r) == ESPURIO       # entre as assinaturas
    assert classify(400, 2.0, r) == ESPURIO        # poeira
    assert classify(200000, 1.0, r) == ESPURIO     # blob gigante


def test_linhas_de_classe_zero_e_fundidas_nao_entram_no_ajuste():
    rows = _train() + _rows(0, 50, 9000, 3.0) + _rows(7, 40, 99999, 9.0, fused=True)
    r = fit_class_ranges(rows)
    assert 0 not in r.boxes
    assert classify(99999, 9.0, r) == ESPURIO       # as fundidas não alargaram a faixa 7
    assert classify(9000, 3.0, r) == ESPURIO


def test_corte_por_percentil_ignora_outlier():
    rows = _rows(7, 60, 2700, 1.02) + [
        {"class_id": 7, "fused": False, "area": 400.0, "aspect_ratio": 1.0}  # fragmento
    ]
    r = fit_class_ranges(rows, quantile=0.10, margin=0.0)
    assert classify(400, 1.0, r) == ESPURIO
    # sem corte (quantil 0) o outlier alarga a faixa
    r0 = fit_class_ranges(rows, quantile=0.0, margin=0.0)
    assert classify(400, 1.0, r0) == 7


def test_margem_alarga_a_faixa():
    rows = _rows(7, 60, 2700, 1.02)
    estreita = fit_class_ranges(rows, quantile=0.10, margin=0.0)
    larga = fit_class_ranges(rows, quantile=0.10, margin=0.30)
    ponto = (2700 * np.exp(0.2), 1.02)
    assert classify(*ponto, estreita) == ESPURIO
    assert classify(*ponto, larga) == 7


def test_faixas_sobrepostas_escolhem_o_centro_mais_proximo():
    rows = _rows(5, 80, 12100, 5.6, spread=0.10) + _rows(6, 80, 12000, 5.7, spread=0.10)
    r = fit_class_ranges(rows, quantile=0.05, margin=0.2)
    # ponto no centro de cada uma cai em ambas as faixas; vence a mais central
    assert classify(12100, 5.6, r) in (5, 6)
    a = classify(12100 * 1.15, 5.6, r)
    b = classify(12100, 5.6, r)
    assert a in (5, 6) and b in (5, 6)


def test_razao_de_aspecto_menor_que_1_e_tratada_como_1():
    r = fit_class_ranges(_train())
    assert classify(2700, 0.97, r) == classify(2700, 1.0, r) == 7


def test_fit_sem_dados_utilizaveis_falha_alto():
    with pytest.raises(ValueError):
        fit_class_ranges(_rows(0, 20, 1000, 1.0))
    with pytest.raises(ValueError):
        fit_class_ranges([])


def test_classe_com_poucas_amostras_falha_alto():
    with pytest.raises(ValueError, match="amostras"):
        fit_class_ranges(_train() + _rows(13, 2, 2800, 1.5), min_samples=5)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        fit_class_ranges(_train(), quantile=0.6)
    with pytest.raises(ValueError):
        fit_class_ranges(_train(), margin=-0.1)


# --------------------------------------------------------------------------
# grupo (porca / parafuso / espurio) vem do config
# --------------------------------------------------------------------------
def test_group_name_usa_o_mapeamento_do_config():
    assert group_name(ESPURIO) == "espurio"
    assert group_name(7) == config.GROUP_NUT
    assert group_name(1) == config.GROUP_SCREW
    with pytest.raises(KeyError):
        group_name(99)


# --------------------------------------------------------------------------
# persistência: os limiares precisam ser legíveis e reproduzíveis
# --------------------------------------------------------------------------
def test_json_ida_e_volta_preserva_as_faixas(tmp_path):
    r = fit_class_ranges(_train(), quantile=0.10, margin=0.15)
    p = tmp_path / "ranges.json"
    r.save(p)
    r2 = ClassRanges.load(p)
    assert r2.quantile == 0.10 and r2.margin == 0.15
    for pt in [(2700, 1.02), (31000, 12.0), (8000, 3.0), (3100, 5.4)]:
        assert classify(*pt, r) == classify(*pt, r2)


def test_json_e_legivel_com_unidades_originais(tmp_path):
    """O arquivo traz área em px e razão adimensional (não logs), para poder
    ser citado direto no relatório."""
    r = fit_class_ranges(_train(), quantile=0.10, margin=0.0)
    p = tmp_path / "ranges.json"
    r.save(p)
    d = json.loads(p.read_text())
    c7 = d["classes"]["7"]
    assert c7["group"] == config.GROUP_NUT
    assert 2000 < c7["area_min"] < 2700 < c7["area_max"] < 3500
    assert 1.0 <= c7["aspect_min"] <= c7["aspect_max"] < 1.3


def test_load_de_arquivo_inexistente():
    with pytest.raises(FileNotFoundError):
        ClassRanges.load(Path("/nao/existe/ranges.json"))


# --------------------------------------------------------------------------
# Dados reais (coletados pelo experimento collect_descriptors)
# --------------------------------------------------------------------------
REAL_CSV = config.OUTPUT_DIR / "component_descriptors.csv"
real = pytest.mark.skipif(not REAL_CSV.is_file(), reason="outputs/component_descriptors.csv ausente")


@real
def test_validacao_cruzada_por_imagem_nos_dados_reais():
    """Trava de regressão: com validação cruzada POR IMAGEM (imagens de teste
    nunca vistas no ajuste), o acerto por grupo fica acima de 90 %."""
    from experiments.fit_class_ranges import cross_validate, load_rows

    rows = load_rows(REAL_CSV)
    res = cross_validate(rows, k=5, seed=0)
    assert res["group_accuracy"] >= 0.90
    assert res["n"] > 1500
    # nenhuma porca virar parafuso (e vice-versa) em mais de 2 % dos casos
    assert res["confusion"]["porca"].get("parafuso", 0) <= 0.02 * res["support"]["porca"]
    assert res["confusion"]["parafuso"].get("porca", 0) <= 0.02 * res["support"]["parafuso"]
