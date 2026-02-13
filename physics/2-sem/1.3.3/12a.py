
# 1.1) Заголовок графиков d3 (измени при необходимости)
PLOT_TITLE_D3 = "d3"
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) ВХОДНЫЕ ДАННЫЕ
# =========================

# Коэффициенты микроманометра
K = 0.2       # твой наклон
n_corr = 0.9953  # твоя поправка по температуре

# Координаты точек (в метрах). Если у тебя другие — замени.
x = np.array([0.000, 0.115, 0.415, 0.815, 1.315], dtype=float)

# Твои измерения N: считаем, что ΔP_ij = P_i - P_j = 9.8067 * N * K * n
measurements = [
    {"i": 0, "j": 1, "N": 42},
    {"i": 0, "j": 2, "N": 67},
    {"i": 0, "j": 3, "N": 95},
    {"i": 0, "j": 4, "N": 129},
    {"i": 1, "j": 2, "N": 27},
    {"i": 1, "j": 3, "N": 54},
    {"i": 1, "j": 4, "N": 87},
    {"i": 2, "j": 3, "N": 29},
    {"i": 2, "j": 4, "N": 62},
    {"i": 3, "j": 4, "N": 34},
]

# =========================
# 2) N -> ΔP (Па) и восстановление P_i по МНК
# =========================

G = 9.8067
n_points = len(x)

# Составляем систему A P = b из уравнений P_i - P_j = ΔP_ij
A = []
b = []
for m in measurements:
    i, j = int(m["i"]), int(m["j"])
    N = float(m["N"])
    dP = G * N * K * n_corr  # Па
    row = np.zeros(n_points)
    row[i] = 1.0
    row[j] = -1.0
    A.append(row)
    b.append(dP)

# Фиксируем уровень отсчёта: P0 = 0
row0 = np.zeros(n_points)
row0[0] = 1.0
A.append(row0)
b.append(0.0)

A = np.vstack(A)
b = np.array(b, dtype=float)

# Решение МНК
P, *_ = np.linalg.lstsq(A, b, rcond=None)

# Для удобства: падение давления от точки 0
P_drop = -P  # т.к. P0=0

# =========================
# 3) Аппроксимация P(x) методом МНК (гладкая линия)
# =========================

# ВЫБОР УЧАСТКА ДЛЯ АППРОКСИМАЦИИ:
# - если хочешь аппроксимацию по всем точкам: fit_from = 0
# - если хочешь исключить входной участок (обычно точка 0 или 0..1): fit_from = 1 или 2
fit_from = 1
xf = x[fit_from:]
Pf = P[fit_from:]

# Линейная аппроксимация: P(x) = a*x + c
a, c = np.polyfit(xf, Pf, deg=1)

# Гладкая линия
x_line = np.linspace(x.min(), x.max(), 300)
P_line = a * x_line + c

# (опционально) оценка качества аппроксимации на выбранном участке
P_pred = a * xf + c
ss_res = np.sum((Pf - P_pred) ** 2)
ss_tot = np.sum((Pf - np.mean(Pf)) ** 2)
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

# =========================
# 4) Графики
# =========================

plt.figure()
plt.plot(x, P, "o", label="Восстановленные точки P_i (МНК по измерениям)")
plt.plot(x_line, P_line, "-", label=fr"Аппроксимация МНК: $P(x)=ax+c$,  $R^2={R2:.3f}$")
plt.xlabel("x, м")
plt.ylabel("P относительно P0, Па (P0=0)")
plt.title(f"{PLOT_TITLE_D3}: P(x)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(x, P_drop, "o", label=r"Точки $\Delta P_{\mathrm{drop}}(x)=P_0-P(x)$")
plt.plot(x_line, -P_line, "-", label=r"Аппроксимация МНК для $\Delta P_{\mathrm{drop}}(x)$")
plt.xlabel("x, м")
plt.ylabel("Падение давления от точки 0, Па")
plt.title(f"{PLOT_TITLE_D3}: ΔP_drop(x)")
plt.grid(True)
plt.legend()
plt.show()

# =========================
# 5) Печать таблицы + диагностика согласованности измерений
# =========================

print("Точки и восстановленные давления (P0 = 0):")
for idx, (xi, Pi) in enumerate(zip(x, P)):
    print(f"  {idx}: x={xi:.6f} м,  P={Pi:.3f} Па")

# Среднеквадратичная невязка по уравнениям (насколько измерения между собой согласованы)
residuals = A @ P - b
rmse = np.sqrt(np.mean(residuals**2))
print(f"\nRMSE по уравнениям P_i - P_j = ΔP_ij: {rmse:.3f} Па")
print(f"Линейная аппроксимация на точках {fit_from}..{n_points-1}: a={a:.3f} Па/м, c={c:.3f} Па, R^2={R2:.3f}")
