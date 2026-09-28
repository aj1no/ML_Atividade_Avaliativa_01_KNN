# Atividade Avaliativa I: K-Nearest Neighbors (K-NN) — Aprendizado de Máquina I

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/ML_Atividade_Avaliativa_01_KNN/blob/main/Atividade_Pratica_KNN.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-388e3c?style=flat-square)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a resolução integral da **Atividade Avaliativa I** da disciplina de **Aprendizado de Máquina I**, focada na investigação empírica e conceitual aprofundada do algoritmo de classificação **K-Nearest Neighbors (K-NN)**.

---

## Informações Acadêmicas

* **Curso:** Ciência de Dados
* **Disciplina:** Aprendizado de Máquina I
* **Professor:** Prof. Me. Mateus Guilherme Fuini
* **Integrantes:**
  * Caio Roberto Farias Saraiva
  * Rodolfo Vinicius Cima Takemoto

---

## Questão Central da Atividade

> *"Qual configuração de K-NN apresenta a melhor capacidade de generalização para o conjunto de dados recebido e quais evidências sustentam essa decisão?"*

---

## Estrutura do Notebook e Roteiro Experimental

O notebook [`Atividade_Pratica_KNN.ipynb`](Atividade_Pratica_KNN.ipynb) desenvolve um pipeline experimental rigoroso estruturado nas seguintes etapas:

1. **Conhecimento Inicial do Conjunto de Dados (EDA):**
   - Carga do dataset com 752 observações, 12 atributos previsores contínuos e target binário (`tipo_X`: 59,04%, `tipo_Y`: 40,96%).
   - Diagnóstico de ausência de valores nulos e mapeamento de atributos com escalas e desvios-padrão amplamente discrepantes.
2. **Separação Treino e Teste com Estratificação:**
   - Divisão em 75% treino (564 amostras) e 25% teste (188 amostras) com `stratify=y` e `random_state=42`.
   - Garantia de isolamento estrito para evitar vazamento de dados (*Data Leakage*).
3. **Modelo de Linha de Base (Baseline):**
   - Avaliação do K-NN padrão ($k=5$, métrica Euclidiana, pesos uniformes) sem pré-processamento/escalonamento.
   - Desempenho no teste: **Acurácia: 60,11%** | **F1-macro: 54,64%** (severamente prejudicado pelas variáveis de alta amplitude).
4. **Comparação de Escalonadores (`MinMaxScaler` vs `StandardScaler`):**
   - Comparação direta contra os dados originais.
   - Salto expressivo com normalização Z-score: **Acurácia: 88,30% (+28,19 p.p.)** e **F1-macro: 87,85% (+33,21 p.p.)**.
5. **Investigação do Hiperparâmetro $k$ e Curva de Complexidade:**
   - Varredura de $k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31\}$.
   - Diagnóstico do dilema Viés × Variância:
     - $k=1$: Overfitting severo (Treino F1 = 100%, Teste F1 = 86,75%, Gap = 13,25%).
     - $k=5$: Ponto ótimo de generalização (Teste F1 = 87,85%, Gap = 4,38%).
     - $k \ge 15$: Início de Underfitting com fronteiras excessivamente suavizadas.
6. **Métrica de Distância (Euclidiana vs Manhattan):**
   - Comparação entre norma $L_2$ ($p=2$) e norma $L_1$ ($p=1$).
   - A métrica Euclidiana demonstrou melhor aderência à topologia esférica/elipsoidal dos clusters normalizados.
7. **Ponderação dos Vizinhos (`uniform` vs `distance`):**
   - Análise de robustez frente a ruídos. A votação uniforme (`uniform`) provou maior estabilidade e menor risco de superajuste em relação à ponderação por inverso da distância (`distance`).
8. **Efeito da Dimensionalidade e Inserção de Ruído:**
   - Inclusão de 10 atributos de ruído uniforme sintético $\mathcal{U}(0, 1)$ para simular a Maldição da Dimensionalidade.
   - Queda de desempenho constatada: redução do F1-score macro de **87,85% para 79,93% (-7,92 p.p.)**, comprovando que a proximidade geométrica perde significado em hipervolumes esparsos.
9. **Configuração Final e Avaliação Conclusiva:**
   - Validação detalhada do modelo ótimo com Matriz de Confusão e `classification_report`.
10. **Discussão Conceitual Obrigatória (Pergunta Final):**
    - Análise aprofundada comparando modelo de treino quase perfeito (alto overfit / alta variância) versus modelo com gap mínimo (alta generalização / baixa variância).

---

## Tabela Geral Comparativa de Resultados

| Seção / Experimento | Configuração Avaliada | Acc Treino | Acc Teste | F1 Treino | F1 Teste | Gap Treino-Teste | Diagnóstico / Conclusão |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **3. Baseline** | $k=5$, Sem Escala | 0.7429 | 0.6011 | 0.7188 | 0.5464 | 0.1724 | Viesado por atributos de grande variância |
| **4. Escalonador** | $k=5$, MinMaxScaler | 0.9220 | 0.8670 | 0.9184 | 0.8624 | 0.0560 | Ganho expressivo de separabilidade |
| **4. Escalonador** | **$k=5$, StandardScaler** | **0.9273** | **0.8830** | **0.9223** | **0.8785** | **0.0438** | **Melhor pré-processamento (+33,2 p.p. F1)** |
| **5. Hiperparâmetro** | $k=1$ (StandardScaler) | 1.0000 | 0.8723 | 1.0000 | 0.8675 | 0.1325 | **Overfitting / Alta Variância** |
| **5. Hiperparâmetro** | $k=3$ (StandardScaler) | 0.9415 | 0.8670 | 0.9380 | 0.8624 | 0.0756 | Bom desempenho, gap ligeiramente maior |
| **5. Hiperparâmetro** | **$k=5$ (StandardScaler)** | **0.9273** | **0.8830** | **0.9223** | **0.8785** | **0.0438** | **Ponto Ótimo de Generalização** |
| **5. Hiperparâmetro** | $k=7$ (StandardScaler) | 0.9167 | 0.8670 | 0.9110 | 0.8617 | 0.0493 | Fronteira de decisão mais suave |
| **5. Hiperparâmetro** | $k=21$ (StandardScaler) | 0.8883 | 0.8457 | 0.8804 | 0.8385 | 0.0419 | Tendência a **Underfitting** |
| **6. Distância** | Manhattan ($p=1, k=5$) | 0.9220 | 0.8723 | 0.9168 | 0.8679 | 0.0489 | Desempenho sólido |
| **6. Distância** | **Euclidiana ($p=2, k=5$)** | **0.9273** | **0.8830** | **0.9223** | **0.8785** | **0.0438** | **Melhor métrica de distância** |
| **7. Pesos** | Distance ($k=5, p=2$) | 1.0000 | 0.8777 | 1.0000 | 0.8727 | 0.1273 | Treino decorado (gap elevado) |
| **7. Pesos** | **Uniform ($k=5, p=2$)** | **0.9273** | **0.8830** | **0.9223** | **0.8785** | **0.0438** | **Mais estável e robusto** |
| **8. Dimensionalidade**| K-NN com 10 ruídos | 0.8777 | 0.8085 | 0.8690 | 0.7993 | 0.0697 | **Degradação por Maldição da Dimensionalidade** |
| **9. Modelo Final** | **StandardScaler + $k=5$ + $L_2$ + Uniform** | **0.9273** | **0.8830** | **0.9223** | **0.8785** | **0.0438** | **Configuração Final Selecionada** |

---

## Configuração Final Selecionada

* **Pré-processamento:** `StandardScaler` (Z-score aplicado estritamente com `fit` no treino e `transform` no teste);
* **Número de Vizinhos ($k$):** `5`;
* **Métrica de Distância:** Minkowski com $p=2$ (Distância Euclidiana);
* **Ponderação dos Vizinhos:** `weights='uniform'`.

### Matriz de Confusão no Conjunto de Teste:
```text
                  Predito: tipo_X   Predito: tipo_Y
Real: tipo_X            98                13
Real: tipo_Y             9                68
```
- **Taxa de Acerto tipo_X:** 88,29% (98 / 111)
- **Taxa de Acerto tipo_Y:** 88,31% (68 / 77)

---

## Discussão Conceitual (Pergunta Final)

**Cenário:**
> *Considere dois modelos: o primeiro apresenta desempenho quase perfeito nos dados de treinamento, mas sofre uma redução perceptível no teste; o segundo apresenta desempenho ligeiramente inferior no treino, porém mantém resultados muito semelhantes no teste. Qual modelo é mais adequado para novas observações?*

### Resposta Técnica Fundamentada:
O **Segundo Modelo** é inquestionavelmente o mais adequado para emprego prático e produção.

1. **Hiperparâmetro $k$ e Fronteiras de Decisão:** O primeiro modelo espelha um K-NN com $k=1$ (fronteiras ultracomplexas e recortadas em torno de ruídos). O segundo modelo representa um $k$ equilibrado (ex: $k=5$), com fronteiras regulares que capturam a distribuição real.
2. **Dilema Viés × Variância:** O primeiro modelo sofre de **alta variância** (instável ante novas amostras). O segundo opera no ponto ótimo do *bias-variance tradeoff*, com **baixa variância** e excelente estabilidade.
3. **Overfitting vs. Generalização:** O objetivo de um modelo supervisionado é a **generalização estatística** para dados não vistos, e não a memorização do conjunto de treinamento.
4. **Confiabilidade:** Um gap estreito entre treino e teste assegura previsibilidade e menor risco operacional ao lidar com novos registros.

---

## Ferramentas Utilizadas

Em conformidade com a transparência acadêmica:

* **Ambiente de Desenvolvimento:** Google Colaboratory / Antigravity IDE (Python 3.14 / Jupyter Kernel).
* **Bibliotecas Principais:** `scikit-learn` (modelagem K-NN, normalizadores e métricas), `pandas` & `numpy` (estruturação e manipulação vetorial), `matplotlib` & `seaborn` (geração de gráficos e mapas de calor).
* **Assistência de IA (Antigravity / Gemini 3.7 Flash High):** Apoio na formulação dos gráficos comparativos, revisão estatística do trade-off viés-variância e documentação Markdown.

---

## Como Executar

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/aj1no/ML_Atividade_Avaliativa_01_KNN.git
   cd ML_Atividade_Avaliativa_01_KNN
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   .\venv\Scripts\activate  # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn jupyter
   ```

4. **Inicie o Jupyter Notebook ou abra no Google Colab:**
   ```bash
   jupyter notebook Atividade_Pratica_KNN.ipynb
   ```

---

## Licença

Este projeto está sob a licença [MIT](LICENSE).
