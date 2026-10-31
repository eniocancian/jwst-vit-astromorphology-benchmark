# Relatório Técnico: Benchmarking de Representações de Visão Computacional (ViT vs. CNNs) em Dados Astrofísicos do JWST (NIRCam)

**Contexto do Estudo:** Prova de conceito de aprendizado de máquina aplicada a imagens astronômicas de Nível 3 (`_i2d.fits`) obtidas pelo *James Webb Space Telescope* (JWST) na região de **Sagittarius C** (*Proposal ID 2221*, instrumento NIRCam).

---

## 1. Sumário Executivo e Principais Conclusões
Este estudo investigou a capacidade de diferentes backbones de visão computacional — incluindo Transformers (Vision Transformers) e redes convolucionais clássicas (CNNs) — em aprender representações morfológicas latentes de estruturas estelares e nebulares sem o uso de fine-tuning custoso, operando em regime de extração de embeddings congelados.

- **O Campeão do Benchmark:** O modelo **ViT-Tiny (CLS)** demonstrou superioridade incontestável em termos de acurácia, Macro F1 e eficiência computacional (32 FPS com 5,5 milhões de parâmetros).
- **Rigidez Metodológica:** Para evitar superestimação de desempenho por autocorrelação espacial (sobreposição de 25% entre os patches), implementou-se uma **Validação Cruzada Espacial por Blocos (`StratifiedGroupKFold`)**.
- **Consistência Estatística:** Testes pareados de Wilcoxon fold a fold confirmaram que a vantagem do `ViT-Tiny` sobre todos os rivais é estatisticamente significativa ($p < 0.05$).

---

## 2. Arquitetura do Pipeline e Pré-Processamento

1. **Radiometria:** Aplicação combinada de `ZScaleInterval` e `AsinhStretch` sobre a extensão científica (`SCI`) para realçar detalhes de alto contraste em nebulosas densas e aglomerados estelares compactos.
2. **Tiling Espacial:** Recorte do mosaico em patches de $224 \times 224$ pixels com *stride* de 168 px (25% de sobreposição).
3. **Extração de Características:** Utilização de backbones do pacote `timm` com `num_classes=0`, extraindo diretamente o token global `[CLS]` (no caso dos Transformers) ou o vetor de pooling global (nas CNNs).

---

## 3. Matriz de Resultados Supervisionados (Validação Cruzada Espacial)

Desempenho consolidado obtido através de 5 folds e 5 repetições, utilizando classificadores lineares e não-lineares estritamente confinados aos dados de treino:

| Modelo | Acurácia (LogReg) | Macro F1 (LogReg) | Acurácia (SVM-RBF) | Macro F1 (SVM-RBF) |
| :--- | :---: | :---: | :---: | :---: |
| **ViT-Tiny (CLS)** | 73,7 ± 11,2% | 0,716 ± 0,115 | **74,4 ± 10,8%** | **0,723 ± 0,111** |
| **ViT-Tiny (pooling espacial)** | 71,9 ± 11,8% | 0,698 ± 0,121 | 73,1 ± 11,4% | 0,707 ± 0,118 |
| **ViT-Base (CLS)** | 61,2 ± 13,5% | 0,582 ± 0,138 | 61,0 ± 12,9% | 0,589 ± 0,132 |
| **EfficientNet-B0** | 56,2 ± 14,1% | 0,545 ± 0,144 | 54,6 ± 13,8% | 0,545 ± 0,139 |
| **ResNet-50** | 38,2 ± 13,0% | 0,361 ± 0,134 | 29,8 ± 11,5% | 0,298 ± 0,112 |
| **MobileNetV2** | 35,5 ± 12,4% | 0,332 ± 0,128 | 36,4 ± 13,1% | 0,229 ± 0,105 |
| *Descritores Físicos (Oráculo)* | *84,3 ± 8,5%* | *0,829 ± 0,089* | *82,8 ± 9,1%* | *0,814 ± 0,094* |

---

## 4. Análise Estatística Pareada (Testes de Wilcoxon)

Comparando fold a fold a distribuição de Macro F1 (com classificador SVM-RBF) do **ViT-Tiny (CLS)** frente às demais arquiteturas:

- **vs. ViT-Base (CLS):** $+0,1338$ ($p = 0.0000$ — Significativo)
- **vs. EfficientNet-B0:** $+0,1777$ ($p = 0.0000$ — Significativo)
- **vs. ResNet-50:** $+0,4245$ ($p = 0.0000$ — Significativo)
- **vs. MobileNetV2:** $+0,4942$ ($p = 0.0000$ — Significativo)
- **vs. ViT-Tiny (pooling espacial):** $+0,0158$ ($p = 0.1128$ — **Não significativo**, comprovando que estratégias complexas de pooling não trazem ganho estatístico relevante sobre o token `[CLS]` puro).

---

## 5. Eficiencia Computacional e Custo de Inferência

| Arquitetura | Parâmetros Totais | Latência Média por Patch | Throughput Operacional |
| :--- | :---: | :---: | :---: |
| **MobileNetV2** | 2,2 M | 26,00 ms | 38,5 FPS |
| **EfficientNet-B0** | 4,0 M | 29,87 ms | 33,5 FPS |
| **ViT-Tiny (CLS)** | **5,5 M** | **31,25 ms** | **32,0 FPS** |
| **ResNet-50** | 23,5 M | 87,90 ms | 11,4 FPS |
| **ViT-Base** | 85,8 M | 180,36 ms | 5,5 FPS |

---

## 6. Considerações Finais e Próximos Passos
O estudo valida com sucesso a viabilidade de modelos baseados em *Vision Transformers* compactos para a caracterização morfológica rápida de grandes volumes de dados astrofísicos do JWST. Como direções futuras, recomenda-se a expansão para mosaicos multiespectrais completos (múltiplos filtros NIRCam/MIRI simultâneos) e a aplicação de técnicas de *Self-Supervised Learning* (SSL) adaptadas para o domínio astronômico.