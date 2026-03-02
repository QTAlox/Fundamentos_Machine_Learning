# 📊 Fundamentos de Machine Learning em Python

Este repositório contém um **script educacional único** que implementa, do zero (sem bibliotecas de ML), os principais conceitos teóricos de Machine Learning.
O objetivo não é performance, mas **entendimento conceitual** — cada parte do código corresponde diretamente a um tópico clássico estudado em cursos universitários.

O arquivo principal demonstra como um modelo:

1. faz previsões
2. mede erro
3. é avaliado
4. aprende (otimização)
5. evita overfitting

---

## 🎯 Objetivo

Mostrar de forma prática a conexão entre:

* Modelo matemático
* Função de perda
* Risco empírico
* Métricas de avaliação
* Otimização (Gradient Descent)
* Regularização
* Generalização

Tudo foi implementado manualmente para facilitar o aprendizado.

---

## 📁 Estrutura Conceitual do Script

### 1️⃣ Regressão

Implementação de um modelo linear:

[
\hat{y} = w_1x_1 + w_2x_2 + ... + b
]

O script calcula:

* Previsão do modelo
* **MSE (Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **Huber Loss**
* **RMSE (Root Mean Squared Error)**
* **R² (Coeficiente de Determinação)**

Isso demonstra como avaliar modelos com saída numérica contínua.

---

### 2️⃣ Classificação

Para problemas categóricos (ex.: 0 ou 1), o código implementa:

* 0-1 Loss (taxa de erro)
* Log Loss (Cross-Entropy)

---

### 3️⃣ Avaliação de Classificação

Métricas clássicas usadas em ML:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de Confusão

Essas métricas são padrão para avaliar classificadores.

---

### 4️⃣ Otimização (Treinamento do Modelo)

Implementação manual do **Gradient Descent**:

[
w = w - \eta \frac{\partial L}{\partial w}
]

Mostra como o modelo aprende ajustando seus parâmetros para minimizar o erro.

---

### 5️⃣ Regularização

Para reduzir overfitting:

* L1 (Lasso)
* L2 (Ridge)

---

### 6️⃣ Generalização

Demonstração conceitual:

* erro de treino
* erro de teste

Explica a diferença entre **underfitting** e **overfitting**.

---

## ▶️ Como executar

No terminal, dentro da pasta do projeto:

```bash
python main.py
```

*(ou o nome do arquivo que você salvou)*

---

## 📚 O que você aprende com este código

Este projeto cobre praticamente o ciclo completo de ML básico:

* Modelagem matemática
* Funções de perda
* Avaliação
* Treinamento
* Regularização
* Generalização

Ele foi feito como material de estudo e revisão teórica.

---

## ⚠️ Observação

Este projeto é **educacional**.
Em aplicações reais usamos bibliotecas como `scikit-learn`, `numpy` e `pandas`, mas aqui os cálculos são feitos manualmente para deixar claro o funcionamento interno dos algoritmos.

---

## 👤 Autor

Projeto desenvolvido para estudo de fundamentos de Machine Learning e construção de portfólio.

Sinta-se livre para usar, modificar e estudar o código.
