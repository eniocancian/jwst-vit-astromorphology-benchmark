# 🔭 JWST Sagittarius C: Astromorphology & ViT-Tiny Explorer (Benchmark & Web App)

Prova de conceito avançada de visão computacional e inteligência artificial generativa sobre um mosaico de Nível 3 (`_i2d.fits`) do **James Webb Space Telescope (JWST / NIRCam)** na região **Sagittarius C** (*Proposal ID 2221*). 

O projeto combina um benchmark rigoroso comparando **ViT-Base**, **ViT-Tiny**, **MobileNetV2**, **EfficientNet-B0** e **ResNet-50** (sob validação cruzada espacial por blocos e testes estatísticos pareados) com uma interface interativa em **Streamlit** integrada ao **Google Gemini** para análises astrofísicas em tempo real.

---

## 🚀 1. Pipeline Metodológico do Benchmark

- **Pré-processamento:** `ZScaleInterval` + `AsinhStretch` aplicados sobre a extensão `SCI` do mosaico original do JWST.
- **Tiling:** Divisão em 121 patches de $224 \times 224$ pixels com *stride* de 168 (25% de sobreposição), descartando bordas sem sinal significativo.
- **Extração de Embeddings:** Backbones do `timm` com `num_classes=0`. Avaliação do `ViT-Tiny` com token `[CLS]` puro versus variante com pooling espacial (*CLS + média + máximo + desvio*).
- **Controle de Vazamento Espacial:** Uso de `StratifiedGroupKFold` para evitar que patches vizinhos com sobreposição fiquem simultaneamente no treino e no teste.

---

## 📊 2. Resultados da Validação Cruzada Espacial (5 folds x 5 repetições)

Desempenho supervisionado médio obtido por classificadores lineares e não-lineares ajustados estritamente sobre os folds de treino:

| Modelo | Acurácia LogReg | Macro F1 LogReg | Acurácia SVM-RBF | Macro F1 SVM-RBF |
| :--- | :---: | :---: | :---: | :---: |
| **ViT-Tiny (CLS)** | **73,7 ± 11,2%** | **0,716 ± 0,115** | **74,4 ± 10,8%** | **0,723 ± 0,111** |
| **ViT-Tiny (pooling espacial)** | 71,9 ± 11,8% | 0,698 ± 0,121 | 73,1 ± 11,4% | 0,707 ± 0,118 |
| **ViT-Base (CLS)** | 61,2 ± 13,5% | 0,582 ± 0,138 | 61,0 ± 12,9% | 0,589 ± 0,132 |
| **EfficientNet-B0** | 56,2 ± 14,1% | 0,545 ± 0,144 | 54,6 ± 13,8% | 0,545 ± 0,139 |
| **MobileNetV2** | 35,5 ± 12,4% | 0,332 ± 0,128 | 36,4 ± 13,1% | 0,229 ± 0,105 |
| **ResNet-50** | 38,2 ± 13,0% | 0,361 ± 0,134 | 29,8 ± 11,5% | 0,298 ± 0,112 |
| **Descritores físicos (oráculo)** | 84,3 ± 8,5% | 0,829 ± 0,089 | 82,8 ± 9,1% | 0,814 ± 0,094 |

![CV espacial](assets/cv_espacial_benchmark.png)

---

## 🔬 3. Comparação Estatística Pareada Fold a Fold (Wilcoxon, p < 0.05)

Avaliando a consistência da superioridade do **ViT-Tiny (CLS)** frente às demais arquiteturas (métricas em Macro F1 com SVM-RBF):

- **ViT-Tiny (CLS) vs. ViT-Base (CLS):** Diferença média de $+0,1338$ ($p = 0.0000$ — Significativo)
- **ViT-Tiny (CLS) vs. EfficientNet-B0:** Diferença média de $+0,1777$ ($p = 0.0000$ — Significativo)
- **ViT-Tiny (CLS) vs. MobileNetV2:** Diferença média de $+0,4942$ ($p = 0.0000$ — Significativo)
- **ViT-Tiny (CLS) vs. ResNet-50:** Diferença média de $+0,4245$ ($p = 0.0000$ — Significativo)
- **ViT-Tiny (CLS) vs. ViT-Tiny (pooling espacial):** Diferença média de $+0,0158$ ($p = 0.1128$ — Não significativo, justificando manter a versão `CLS` mais enxuta).

---

## 📈 4. Estrutura Não Supervisionada (K-Means, K=3) e Eficiência

### Agrupamento Não Supervisionado
| Modelo | Silhouette (↑) | Davies-Bouldin (↓) | ARI vs. rótulos (↑) |
| :--- | :---: | :---: | :---: |
| **ViT-Tiny (CLS)** | **0,2428** | **1,5189** | **0,3145** |
| **EfficientNet-B0** | 0,1367 | 2,1720 | 0,0210 |
| **ResNet-50** | 0,1311 | 2,1779 | 0,0150 |
| **MobileNetV2** | 0,1104 | 2,2140 | 0,0090 |
| **ViT-Base** | 0,0983 | 2,3597 | -0,0120 |

### Eficiência Computacional
| Modelo | Parâmetros | Latência / Patch | Throughput |
| :--- | :---: | :---: | :---: |
| **MobileNetV2** | 2,2 M | 26,00 ms | 38,5 FPS |
| **EfficientNet-B0** | 4,0 M | 29,87 ms | 33,5 FPS |
| **ResNet-50** | 23,5 M | 87,90 ms | 11,4 FPS |
| **ViT-Tiny (CLS)** | **5,5 M** | **31,25 ms** | **32,0 FPS** |
| **ViT-Base** | 85,8 M | 180,36 ms | 5,5 FPS |

---

## 🖥️ 5. Aplicação Interativa (Streamlit + Gemini)

O repositório também inclui uma interface web completa (`app_total.py`) que coloca o modelo vencedor em produção local:
- **Extração em Tempo Real:** Carrega o `ViT-Tiny` otimizado via `timm` para computar vetores latentes de 192 dimensões e métricas estatísticas de intensidade ($\mu$ e $\sigma$) de patches celestes.
- **Análise Astrofísica Inteligente:** Integra o SDK do Google GenAI (`google-genai`) utilizando o modelo **`gemini-3.8-flash`** para gerar relatórios científicos textuais automáticos sobre berçários estelares, nebulosidade e densidade em Sagittarius C.

---

## ⚙️ 6. Como Executar o Projeto

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/eniocancian/jwst-vit-astromorphology-benchmark.git](https://github.com/eniocancian/jwst-vit-astromorphology-benchmark.git)
   cd jwst-vit-astromorphology-benchmark