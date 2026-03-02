# 📊 Fundamentos de Machine Learning em Python

Este projeto é uma implementação didática dos principais conceitos fundamentais de Machine Learning, desenvolvida inteiramente em Python puro, sem o uso de bibliotecas especializadas como scikit-learn.

O objetivo principal é demonstrar, de forma clara e estruturada, como os componentes matemáticos e conceituais de Machine Learning se conectam na prática.

---

## 🎯 Objetivo do Projeto

Demonstrar, de forma educacional:

* Como um modelo linear realiza previsões
* Como calcular funções de perda (loss functions)
* O que é risco empírico
* Como avaliar modelos de regressão e classificação
* Como funciona o Gradient Descent
* O que é regularização
* O que significa generalização do modelo

O código foi estruturado para funcionar como material de estudo e revisão teórica.

---

## 🧠 Estrutura do Código

### 🔹 Regressão

Inclui:

* Modelo de Regressão Linear
* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* Huber Loss
* RMSE (Root Mean Squared Error)
* R² (Coeficiente de Determinação)

Essas métricas são utilizadas para avaliar modelos que preveem valores numéricos contínuos.

---

### 🔹 Classificação

Inclui:

* 0-1 Loss
* Log Loss (Cross-Entropy)

Essas funções são utilizadas para medir o erro em problemas de classificação.

---

### 🔹 Avaliação de Classificação

Métricas implementadas:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de Confusão

Essas métricas são amplamente utilizadas na avaliação de classificadores.

---

### 🔹 Otimização

Implementação manual de:

* Gradient Descent

Mostra como os pesos do modelo são ajustados para minimizar a função de perda.

---

### 🔹 Regularização

Inclui:

* L1 (Lasso)
* L2 (Ridge)

Demonstra como reduzir overfitting penalizando a complexidade do modelo.

---

### 🔹 Generalização

Demonstra conceitualmente:

* Erro de treino
* Erro de teste

Mostra a diferença entre overfitting e underfitting.

---

## ▶️ Como Executar

1. Salve o arquivo Python.
2. Execute no terminal com:

python nome_do_arquivo.py

---

## 📚 Conceitos Abordados

Este projeto cobre os pilares fundamentais de Machine Learning:

* Modelagem matemática
* Funções de perda
* Avaliação de desempenho
* Otimização
* Regularização
* Generalização

---

## ⚠️ Observação

Este projeto é educacional.
Em aplicações reais, bibliotecas como scikit-learn, numpy e pandas são normalmente utilizadas.

Aqui, todos os cálculos são feitos manualmente para facilitar a compreensão dos fundamentos.

---

## 👤 Autor

Projeto desenvolvido para estudo de fundamentos de Machine Learning e construção de portfólio.

Sinta-se livre para utilizar, adaptar e expandir o código.
