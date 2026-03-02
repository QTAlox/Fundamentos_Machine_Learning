import math

# =========================================================
# ---------------------- REGRESSAO -------------------------
# =========================================================

print("\n--- Regressao ---")

# Modelo Linear
w1, w2, w3, b = 0.5, 1.2, -0.3, 2
x1, x2, x3 = 10, 5, 8

y_pred = w1*x1 + w2*x2 + w3*x3 + b
print("\nPrevisão (Exemplo Modelo Regressao Linear) =", y_pred)


# ----------------- MSE -----------------
y = [10, 5, 8]
y_hat = [8, 6, 9]
n = len(y)

def mse_loss(y_real, y_pred):
    return (y_real - y_pred)**2

mse = sum(mse_loss(y[i], y_hat[i]) for i in range(n)) / n
print("\nRisco Empírico (MSE) =", mse)


# ----------------- R² -----------------
y = [3, 5, 7, 9]
y_hat = [2.5, 5.5, 6.5, 10]
n = len(y)

y_mean = sum(y) / n
ss_res = sum((y[i] - y_hat[i])**2 for i in range(n))
ss_tot = sum((y[i] - y_mean)**2 for i in range(n))

r2 = 1 - (ss_res / ss_tot)
print("\nCoeficiente de Determinação (R²) =", r2)


# ----------------- MAE -----------------
y = [10, 5, 8]
y_hat = [8, 6, 9]
n = len(y)

def mae_loss(y_real, y_pred):
    return abs(y_real - y_pred)

mae = sum(mae_loss(y[i], y_hat[i]) for i in range(n)) / n
print("\nRisco Empírico (MAE) =", mae)


# ----------------- Huber -----------------
delta = 1

def huber_loss(y_real, y_pred):
    error = y_real - y_pred
    if abs(error) <= delta:
        return 0.5 * error**2
    else:
        return delta * (abs(error) - 0.5 * delta)

huber = sum(huber_loss(y[i], y_hat[i]) for i in range(n)) / n
print("\nRisco Empírico (Huber) =", huber)


# ----------------- RMSE -----------------
rmse = math.sqrt(mse)
print("\nRMSE =", rmse)


# =========================================================
# -------------------- CLASSIFICACAO -----------------------
# =========================================================

print("\n--- Classificacao ---")

# 0-1 Loss
y = [1, 0, 1, 1]
y_hat = [1, 1, 1, 0]
n = len(y)

def zero_one_loss(y_real, y_pred):
    return 1 if y_real != y_pred else 0

error_rate = sum(zero_one_loss(y[i], y_hat[i]) for i in range(n)) / n
print("\nRisco Empírico (0-1 Loss) =", error_rate)


# Log Loss
y = [1, 0, 1]
p = [0.9, 0.2, 0.8]
n = len(y)

def log_loss(y_real, prob):
    return -(y_real * math.log(prob) + (1-y_real)*math.log(1-prob))

logloss = sum(log_loss(y[i], p[i]) for i in range(n)) / n
print("\nFunção de Perda (Log Loss) =", logloss)


# =========================================================
# ------------- AVALIACAO DE CLASSIFICACAO -----------------
# =========================================================

print("\n--- Avaliacao de Classificacao ---")

y = [1, 0, 1, 1, 0]
y_hat = [1, 1, 1, 0, 0]
n = len(y)

# Accuracy
accuracy = sum(1 for i in range(n) if y[i] == y_hat[i]) / n
print("\nAccuracy =", accuracy)

# Precision, Recall, F1
tp = sum(1 for i in range(n) if y[i]==1 and y_hat[i]==1)
tn = sum(1 for i in range(n) if y[i]==0 and y_hat[i]==0)
fp = sum(1 for i in range(n) if y[i]==0 and y_hat[i]==1)
fn = sum(1 for i in range(n) if y[i]==1 and y_hat[i]==0)

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)

print("Precision =", precision)
print("Recall =", recall)
print("F1-Score =", f1)

print("\nMatriz de Confusão")
print("TP =", tp)
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)


# =========================================================
# ---------------- OTIMIZACAO (GRADIENT DESCENT) ----------
# =========================================================

print("\n--- Otimizacao / Treinamento ---")

x, y = 2, 10
w = 1
eta = 0.01

y_hat = w*x
grad = -2*x*(y - y_hat)
w = w - eta*grad

print("\nPeso do modelo após otimização (Gradient Descent) =", w)


# =========================================================
# -------------------- REGULARIZACAO -----------------------
# =========================================================

print("\n--- Regularizacao ---")

weights = [0.5, -1.2, 0.3]
lambda_ = 0.1

l1 = lambda_ * sum(abs(w) for w in weights)
l2 = lambda_ * sum(w**2 for w in weights)

print("\nPenalização L1 (Lasso) =", l1)
print("Penalização L2 (Ridge) =", l2)


# =========================================================
# ------------------- GENERALIZACAO ------------------------
# =========================================================

print("\n--- Generalizacao do Modelo ---")

erro_treino = 0.05
erro_teste = 0.25

print("\nErro de Treino =", erro_treino)
print("Erro de Teste =", erro_teste)

      
