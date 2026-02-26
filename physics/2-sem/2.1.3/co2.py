import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Линейная модель ---
def linear(x, a, b):
    return a * x + b

# --- Функция для подгонки и возврата результатов ---
def fit_and_predict(x, y, sigma):
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    # Вычисление χ²
    y_fit = linear(x, a, b)
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    ndof = len(x) - 2
    chi2_red = chi2 / ndof
    # Гладкая линия для графика
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = linear(x_smooth, a, b)
    return a, b, da, db, chi2, chi2_red, x_smooth, y_smooth

# --- Генерация четырёх тестовых наборов данных (замените своими) ---
np.random.seed(42)

# Набор 1
x1 = np.array([1, 2, 3, 4])
y1 = np.array([31, 100, 165, 230])
sigma1 = np.ones_like(y1) * 2

# Набор 2
x2 = np.array([1,2,3,4])
y2 = np.array([52, 107, 159, 212])
sigma2 = np.ones_like(y2) * 2

# Набор 3
x3 = np.array([1,2,3,4,5])
y3 = np.array([23,71,121,168,216])
sigma3 = np.ones_like(y3) * 2 

# Набор 4
x4 = np.array([1,2,3,4,5])
y4 = np.array([20,64,104,145,187])
sigma4 = np.ones_like(y4) * 2

# Собираем всё в список
datasets = [
    (x1, y1, sigma1, 'частота 1', 'blue'),
    (x2, y2, sigma2, 'частота 2', 'red'),
    (x3, y3, sigma3, 'частота 3', 'green'),
    (x4, y4, sigma4, 'частота 4', 'purple')
]

# --- Построение общего графика ---
plt.figure(figsize=(10, 8))

for x, y, sigma, label, color in datasets:
    # Подгонка
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)

    # Точки с погрешностями
    plt.errorbar(x, y, yerr=sigma, fmt='o', color=color, capsize=4,
                 markersize=6)

    # Линия регрессии
    plt.plot(xs, ys, '-', color=color, linewidth=2,
             label=f'{label}: y = ({a:.2f}±{da:.2f})x')

    # Можно добавить текст с χ² рядом с линией (но чтобы не загромождать, вынесем в легенду)
    # Для компактности вся информация о параметрах уже в легенде

# Оформление
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('график для углекислого газа', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()
