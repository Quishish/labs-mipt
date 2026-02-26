import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Линейная модель ---
def linear(x, a, b):
    return a * x + b

# --- Функция для подгонки ---
def fit_and_predict(x, y, sigma):
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    y_fit = linear(x, a, b)
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    ndof = len(x) - 2
    chi2_red = chi2 / ndof
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = linear(x_smooth, a, b)
    return a, b, da, db, chi2, chi2_red, x_smooth, y_smooth

# --- Ваши данные (5 наборов) ---
x1 = np.array([1, 2, 3, 4, 5])
y1 = np.array([1314, 1530, 1745, 1961, 2178])
sigma1 = np.ones_like(x1) * 0.5

x2 = np.array([1, 2, 3, 4, 5])
y2 = np.array([1326, 1542, 1760, 1976, 2194])
sigma2 = np.ones_like(x2) * 1.0

x3 = np.array([1, 2, 3, 4, 5])
y3 = np.array([1333, 1555, 1773, 1992, 2211])
sigma3 = np.ones_like(x3) * 0.8

x4 = np.array([1, 2, 3, 4, 5])
y4 = np.array([1361, 1578, 1800, 2023, 2246])
sigma4 = np.ones_like(x4) * 0.3

# Новый пятый набор
x5 = np.array([1, 2, 3, 4, 5])
y5 = np.array([1150, 1379, 1603, 1828, 2054])
sigma5 = np.ones_like(x5) * 0.6

datasets = [
    (x1, y1, sigma1, 'Набор 1', 'blue'),
    (x2, y2, sigma2, 'Набор 2', 'red'),
    (x3, y3, sigma3, 'Набор 3', 'green'),
    (x4, y4, sigma4, 'Набор 4', 'purple'),
    (x5, y5, sigma5, 'Набор 5', 'orange')   # добавлен оранжевый цвет
]

# --- Построение общего графика ---
plt.figure(figsize=(10, 8))

for x, y, sigma, label, color in datasets:
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)

    # Точки с погрешностями (увеличенные кресты)
    plt.errorbar(x, y, yerr=sigma, fmt='o', color=color, capsize=4,
                 markersize=6)

    # Линия регрессии
    plt.plot(xs, ys, '-', color=color, linewidth=2,
             label=f'{label}: y = ({a:.2f}±{da:.2f})x')

# Оформление
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Пять линейных аппроксимаций на одном графике', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()
