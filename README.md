# Inspeção e Contagem Automatizada de Peças Industriais (MVTec Screws)

Pipeline clássico de Visão Computacional (sem redes neurais) para segmentação, rotulagem, extração de descritores geométricos/topológicos, classificação fina e contagem automatizada de parafusos e porcas sobre bancada de madeira no dataset **MVTec Screws** ($1920 \times 1440$ px, 13 classes).

## Visão Geral da Arquitetura

O sistema opera em **duas passadas** sobre imagens RGB e foi projetado para rodar integralmente em CPU comum com alta interpretabilidade:

1. **Pré-processamento e Espaço de Cor (`src/preprocessing.py`):** Suavização Gaussiana proporcional à resolução da imagem seguida pela extração do canal **`b` (eixo amarelo-azul do espaço CIE Lab)**. Como a bancada de madeira é fortemente amarelada ($b$ alto) e os fixadores metálicos são cromática ou relativamente neutros ($b$ baixo), peças escuras e porcas cromadas ficam do mesmo lado do histograma, suprimindo os veios da madeira.
2. **Segmentação de Otsu do Zero e Morfologia (`src/segmentation.py`):** Implementação vetorizada da limiarização de Otsu em **NumPy puro** (histograma + somas acumuladas em $O(N + 256)$), seguida de abertura e fechamento morfológicos com elemento estruturante elíptico.
3. **Rotulagem e Descritores do Zero (`src/features.py`):**
   * **Componentes conectados** extraídos em **NumPy puro** via compressão de trechos horizontais (*run-length*) e fusão por *union-find* com compressão de caminho (conectividade 4 ou 8).
   * **Descritores geométricos e topológicos** vetorizados: área, centroide, caixa delimitadora, *extent*, perímetro de Cauchy-Crofton em 4 direções, eixos da elipse equivalente via momentos centrais de 2ª ordem, razão de aspecto, excentricidade, orientação, **solidez** (via envoltória convexa *Andrew's monotone chain*) e **contagem de furos internos** (`num_holes`).
4. **Segunda Passada Seletiva com Watershed (`src/segmentation.py`):** Componentes grandes rejeitados como espúrios na primeira passada (típicos de duas ou mais peças encostadas) passam por Transformada de Distância Euclidiana + *Watershed* controlado por marcadores para separação individual antes da contagem final.
5. **Classificação em Espaço Logarítmico (`src/classification.py`):** Classificação nas 13 classes originais (e nos grupos macro `porca` e `parafuso`) por faixas calibradas em $(\log \text{área}, \log \text{razão de aspecto})$, combinadas com filtro de solidez mínima e priorização topológica de porcas quando `num_holes >= 1`.

---

## Estrutura do Projeto

```text
screws_project/
├── config.py                     # Caminhos, escalonamento por resolução e mapeamento das 13 classes
├── main.py                       # Ponto de entrada para inferência, contagem e geração de painéis
├── pytest.ini                    # Configuração da suíte de testes
├── requirements.txt              # Dependências Python
├── README.md                     # Documentação e resultados experimentais
├── src/
│   ├── preprocessing.py          # Leitura, extração de canais (Lab, HSV, dist_lab) e suavização
│   ├── segmentation.py           # Limiarização de Otsu (NumPy puro), morfologia e Watershed seletivo
│   ├── features.py               # Rotulagem (runs + union-find), momentos, solidez e furos (NumPy)
│   ├── classification.py           # Calibração por percentis e classificador em espaço logarítmico
│   ├── evaluation.py             # Parser COCO/MVTec, IoU de caixas orientadas (OBB) e métricas
│   └── visualization.py          # Geração de etapas intermediárias e comparação com Ground Truth
├── experiments/
│   ├── compare_channels.py       # Benchmark comparativo entre canais de cor e CLAHE
│   ├── collect_descriptors.py    # Extração de descritores no conjunto de treino
│   ├── fit_class_ranges.py       # Ajuste de faixas e validação cruzada por imagem (5 dobras)
│   └── evaluate_test.py          # Avaliação oficial na partição de teste isolada (60 imagens)
├── tests/                        # 190 testes unitários, de propriedade e de integração
└── outputs/
    └── class_ranges.json         # Faixas calibradas para execução imediata
```

---

## Instalação e Configuração

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Organize o dataset **MVTec Screws** em `data/mvtec_screws/` (ou defina a variável de ambiente `MVTEC_DATA_DIR`):

```text
data/mvtec_screws/
├── images/
├── mvtec_screws.json
└── mvtec_screws_test.json
```

---

## Como Executar

### 1. Suíte de Testes Automatizados (190 testes)

Verifica paridade numérica com o OpenCV (Otsu, rotulagem e momentos), invariâncias geométricas, escalonamento de resolução, separação via *Watershed* e ausência de vazamento entre treino e teste:

```bash
pytest -v
```

### 2. Inferência e Visualização em Imagem Única ou Lote

```bash
# Processa uma imagem e salva as 7 etapas intermediárias em outputs/visualizations/
python main.py --image data/mvtec_screws/images/screws_001.png

# Processa lote do conjunto de teste e gera painéis comparativos com Ground Truth
python main.py --limit 60 --save-vis 8
```

### 3. Reprodução Completa dos Experimentos

```bash
# 1. Comparação de canais de cor (gray, L, b, S, V, dist_lab)
python -m experiments.compare_channels --limit 100

# 2. Coleta de descritores apenas no treino (exclui automaticamente mvtec_screws_test.json)
python -m experiments.collect_descriptors --channel b --limit 100

# 3. Calibração das faixas e validação cruzada por imagem (5 dobras)
python -m experiments.fit_class_ranges

# 4. Avaliação final na partição de teste isolada (60 imagens)
python -m experiments.evaluate_test
```

---

## Resultados Experimentais (Partição de Teste Oficial — 60 imagens)

Os parâmetros do classificador (`Q = 0,10`, `margem = 0,15`) foram calibrados exclusivamente sobre imagens de treino, alcançando **94,6%** de acurácia por grupo e **90,0%** nas 13 classes em validação cruzada de 5 dobras por imagem. Na partição de teste isolada (`mvtec_screws_test.json`, **661 peças reais**), os resultados foram:

> **Ressalva metodológica.** Os 94,6% / 90,0% da validação cruzada referem-se ao classificador por faixas em (log área, log razão de aspecto). As regras adicionais (rejeição por solidez < 0,35, prioridade de porca com furo interno), o Watershed seletivo e o escalonamento por resolução **não foram calibrados separadamente no treino**. Por isso o F1 da partição de teste abaixo deve ser lido como estimativa possivelmente otimista, e não como acurácia de generalização garantida. Para medir o efeito de cada componente, rode a ablação: `python -m experiments.evaluate_test --no-split-fused` (sem Watershed) e compare com o resultado padrão.

### Detecção e Classificação ($\text{IoU} \ge 0{,}50$)

| Métrica / Grupo | TP | FP | FN | Precisão | Recall | F1-Score | IoU Médio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Localização pura** (qualquer classe) | — | — | — | 81,1% | 77,0% | **79,0%** | 0,659 |
| **Porcas** (Classes 7, 9, 10, 11) | 179 | 9 | 23 | 95,2% | 88,6% | **91,8%** | 0,789 |
| **Parafusos** (Classes 1–6, 8, 12, 13) | 329 | 111 | 130 | 74,8% | 71,7% | **73,2%** | 0,588 |
| **Micro (geral)** | 508 | 120 | 153 | 80,9% | 76,9% | **78,8%** | 0,659 |
| **Macro (média entre grupos)** | — | — | — | 85,0% | 80,1% | **82,5%** | 0,689 |

* **Acurácia na classificação fina (13 classes exatas entre os TPs):** **91,3%**

### Contagem Industrial por Imagem

| Categoria | Acurácia Exata | MAE (peças/img) | Total Previsto | Total Real | Previsto / Real |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total de peças** | 56,7% | **1,12** | 628 | 661 | 95,0% |
| **Porcas** | 66,7% | **0,43** | 188 | 202 | 93,1% |
| **Parafusos** | 58,3% | **0,92** | 440 | 459 | 95,9% |

> **Como ler:** "Previsto / Real" compara apenas os totais e **não** mede acerto de contagem: falsos positivos e falsos negativos se cancelam na soma (viés de −0,55 peça/imagem). A medida que importa é a **acurácia exata por imagem (56,7%)** e o MAE (1,12 peça/imagem). Metade das imagens tem a contagem exata; o erro se concentra em poucas imagens com pilhas de peças (ex.: imagem 26 do teste, com 2 peças previstas para 16 reais).

### Desempenho Computacional (CPU, $1920 \times 1440$ px)

* **Tempo médio total:** **283,9 ms/imagem** (~3,52 FPS)
  * Segmentação (Gaussiano + canal `b` + Otsu + morfologia + Watershed seletivo): `138,4 ms`
  * Rotulagem (`run-length` + *union-find*) e descritores (NumPy): `145,2 ms`
  * Classificação em espaço logarítmico: `0,2 ms`

---

## Limitações Conhecidas

* **Pilhas de peças encostadas (principal causa de erro).** Peças que se tocam formam um único componente conectado e o descritor deixa de representar uma peça. O Watershed seletivo não resolve pilhas densas: no teste, a imagem 26 teve 2 peças previstas de 16 (F1 0,11) e as imagens 1 e 10 tiveram F1 de 0,24 e 0,44. Esse é o principal motivo do recall de 71,7% nos parafusos.
* **Classes 5 e 6 indistinguíveis.** Têm o mesmo tamanho (área mediana 12.100 e 12.094 px) e a mesma razão de aspecto; os descritores geométricos não as separam. O acerto por grupo (parafuso/porca) não é afetado, mas o acerto nas 13 classes é limitado por essa troca.
* **Parafusos pequenos vs. manchas da madeira.** As classes 12 e 13 têm área e razão de aspecto parecidas com as de manchas alongadas (na validação cruzada, ~6% dos espúrios viram parafuso e ~6% dos parafusos viram espúrio). Os 111 falsos positivos de parafuso no teste vêm em parte daqui.
* **Circularidade descartada.** Em máscaras reais a circularidade ficou inutilizável (porcas entre 0,33 e 0,49, quando o ideal é ~0,9), provavelmente porque o perímetro inclui furos internos e bordas serrilhadas (hipótese não verificada). Por isso a classificação usa só área e razão de aspecto.
* **Escolha do canal `b` feita sobre amostra que pode incluir imagens de teste.** A comparação de canais (`compare_channels --limit 100`) não exclui as imagens do conjunto de teste. A margem do canal `b` sobre o segundo colocado é grande (precisão de pixels 0,91 contra 0,78), então a conclusão é robusta, mas o procedimento não é estritamente isolado.
* **Sensibilidade de IoU em parafusos extralongos:** Nas classes de razão de aspecto elevada ($10$ a $19$), pequenas perdas na ponta da rosca ou leve desvio angular nos momentos de 2ª ordem podem reduzir o $\text{IoU}$ da caixa orientada abaixo de $0{,}50$, penalizando a métrica de detecção mesmo quando a contagem por imagem está exata.
* **Parafusos de latão/dourados:** Peças com tonalidade próxima à da bancada amarelada apresentam menor contraste cromático no canal `b`, podendo sofrer fragmentação ou omissão pontual.