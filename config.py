"""Configuração central do projeto (caminhos e constantes do dataset).

Manter tudo aqui evita "números mágicos" espalhados pelos módulos e deixa o
relatório rastreável: cada decisão do dataset tem um único lugar de origem.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Pasta do dataset. Pode ser sobrescrita pela variável de ambiente
# MVTEC_DATA_DIR sem alterar o código.
DATA_DIR = Path(os.environ.get("MVTEC_DATA_DIR", PROJECT_ROOT / "data" / "mvtec_screws"))
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "mvtec_screws.json"
ANNOTATIONS_TEST_FILE = DATA_DIR / "mvtec_screws_test.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------
# Mapeamento classe -> grupo.
# ATENÇÃO: o README do dataset NÃO descreve as classes (apenas "13 tipos de
# parafusos e porcas"). Este mapeamento foi INFERIDO por inspeção visual das
# caixas sobre as imagens 001 e 041-044:
#   - 7, 9, 10, 11 -> porcas sextavadas (4 tamanhos)
#   - demais       -> parafusos (longos: 1,2,3; médios/curtos: 4,5,6,8,12,13)
# Não existem arruelas neste dataset.
# --------------------------------------------------------------------------
NUT_CATEGORY_IDS = frozenset({7, 9, 10, 11})
SCREW_CATEGORY_IDS = frozenset({1, 2, 3, 4, 5, 6, 8, 12, 13})

GROUP_NUT = "porca"
GROUP_SCREW = "parafuso"


def group_of(category_id: int) -> str:
    """Retorna o grupo ('porca' ou 'parafuso') de uma classe original.

    Levanta KeyError para classes fora de 1..13 (falhar alto é preferível a
    classificar silenciosamente errado).
    """
    if category_id in NUT_CATEGORY_IDS:
        return GROUP_NUT
    if category_id in SCREW_CATEGORY_IDS:
        return GROUP_SCREW
    raise KeyError(f"categoria desconhecida: {category_id}")

REFERENCE_HEIGHT = 1440
REFERENCE_WIDTH = 1920


def resolution_scale(height: int, width: int) -> float:
    """Fator linear de escala relativo à resolução de calibração (1920x1440)."""
    return ((height * width) / (REFERENCE_HEIGHT * REFERENCE_WIDTH)) ** 0.5


def odd_kernel(base_ksize: int, scale: float) -> int:
    """Ajusta um tamanho de kernel para o inteiro ímpar >= 3 mais próximo."""
    k = max(3, int(round(base_ksize * scale)))
    return k if k % 2 == 1 else k + 1


# --------------------------------------------------------------------------
# Classificador por faixas (Iteração 5)
# Valores escolhidos por validação cruzada POR IMAGEM sobre as imagens que não
# são do conjunto de teste (ver experiments/fit_class_ranges.py --sweep):
# cada classe ocupa a faixa entre os percentis [Q, 1-Q] de (log área, log
# razão de aspecto), alargada por MARGIN (em unidades de log: 0,15 ~ 16 %).
# --------------------------------------------------------------------------
CLASSIFIER_QUANTILE = 0.10
CLASSIFIER_MARGIN = 0.15
CLASSIFIER_MIN_SAMPLES = 5
CLASS_RANGES_FILE = OUTPUT_DIR / "class_ranges.json"
