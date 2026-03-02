# Regressao

    # Forma Geral de um modelo linear

        # Exemplo com 3 variáveis

w1 = 0.5
w2 = 1.2
w3 = -0.3
b = 2

x1 = 10
x2 = 5
x3 = 8

y_hat = w1*x1 + w2*x2 + w3*x3 + b

print("\nPrevisão (Exemplo Modelo Linear) =", y_hat)

    # Risco Empirico

        # MSE (Mean Squared Error)
            # Valores reais
y = [10, 5, 8]

            # Previsões do modelo
y_hat = [8, 6, 9]

            # Número de observações
n = len(y)


            # Função de perda (erro quadrático)
def loss(y_real, y_pred):
    return (y_real - y_pred) ** 2


            # Cálculo do risco empírico (média dos erros quadráticos)
risco_empirico = sum(loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (MSE) =", risco_empirico)

        # MAE (Mean Absolut Error)

            # Valores reais
y = [10, 5, 8]

            # Previsões do modelo
y_hat = [8, 6, 9]

            # Número de observações
n = len(y)

            # Função de perda (erro absoluto)
def loss(y_real, y_pred):
    return abs(y_real - y_pred)

            # Cálculo do risco empírico (média dos erros absolutos)
risco_empirico = sum(loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (MAE) =", risco_empirico)

        # Huber Loss

            # Valores reais
y = [10, 5, 8]

            # Previsões do modelo
y_hat = [8, 6, 9]

            # Número de observações
n = len(y)

            # Parâmetro delta (define a transição entre MSE e MAE)
delta = 1


            # Função de perda Huber
def loss(y_real, y_pred):
    error = y_real - y_pred

    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)


            # Cálculo do risco empírico
risco_empirico = sum(loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (Huber) =", risco_empirico)

# Classificacao

    # 0-1 Loss

        # Valores reais (classes)
y = [1, 0, 1, 1]

        # Previsões do modelo
y_hat = [1, 1, 1, 0]

        # Número de observações
n = len(y)

        # Função de perda 0-1
def loss(y_real, y_pred):
    if y_real != y_pred:
        return 1  # erro
    else:
        return 0  # acerto

        # Cálculo do risco empírico (taxa de erro)
risco_empirico = sum(loss(y[i], y_hat[i]) for i in range(n)) / n

print("\nRisco Empírico (0-1 Loss) =", risco_empirico)

    # Log Loss

import math

        # Valores reais (0 ou 1)
y = [1, 0, 1]

        # Probabilidades previstas pelo modelo
p = [0.9, 0.2, 0.8]

        # Número de observações
n = len(y)

        # Função de perda Log Loss
def loss(y_real, prob):
    return -(y_real * math.log(prob) + (1 - y_real) * math.log(1 - prob))

        # Cálculo do risco empírico
risco_empirico = sum(loss(y[i], p[i]) for i in range(n)) / n

print("\nRisco Empírico (Log Loss) =", risco_empirico)

# Gradiant Descent

        # Dados
x = 2
y = 10

        # Peso inicial
w = 1

        # Learning rate
eta = 0.01

        # Previsão
y_hat = w * x

        # Derivada da loss em relação a w
gradiente = -2 * x * (y - y_hat)

        # Atualização do peso
w = w - eta * gradiente

print("\nNovo peso (Gradient Descent) :", w)

#Exemplo Geral
print("\n--- Exemplo ---")
# ==============================
# 1️⃣ DADOS
# ==============================

# x = variável de entrada
x = [1, 2, 3]

# y = valores reais
y = [2, 4, 6]   # relação perfeita y = 2x


# ==============================
# 2️⃣ MODELO (Regressão Linear)
#    y_hat = w*x + b
# ==============================

w = 0.0   # peso inicial
b = 0.0   # bias inicial

eta = 0.01   # learning rate
n = len(x)


# ==============================
# 3️⃣ TREINAMENTO (Gradient Descent)
# ==============================

for epoch in range(100):

    # ---- Previsões do modelo ----
    y_hat = [w * x[i] + b for i in range(n)]

    # ---- LOSS (MSE) ----
    losses = [(y[i] - y_hat[i])**2 for i in range(n)]

    # ---- RISCO EMPÍRICO (média da loss) ----
    risco_empirico = sum(losses) / n

    # ---- Cálculo dos Gradientes ----
    dw = sum(-2 * x[i] * (y[i] - y_hat[i]) for i in range(n)) / n
    db = sum(-2 * (y[i] - y_hat[i]) for i in range(n)) / n

    # ---- GRADIENT DESCENT (atualização dos parâmetros) ----
    w = w - eta * dw
    b = b - eta * db


# ==============================
# 4️⃣ RESULTADO FINAL
# ==============================

print("Peso final (w):", round(w, 4))
print("Bias final (b):", round(b, 4))
print("Risco Empírico final:", round(risco_empirico, 6))