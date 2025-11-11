import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Генерируем случайные данные
np.random.seed(42)  # для воспроизводимости результатов
x = [0.00275, 0.0055, 0.00825, 0.011, 0.01375, 0.0165, 0.022, 0.0275]
y = [1, 2, 3, 4, 5, 6, 7]

x1 = x[::-1]
y1 = []

# Вычисляем коэффициенты линейной регрессии с оценкой погрешностей
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

# Альтернативный способ через polyfit с ковариационной матрицей
coefficients, covariance = np.polyfit(x, y, 1, cov=True)  # 1 - степень полинома (линейная)
a = coefficients[0]  # угловой коэффициент
a_error = np.sqrt(covariance[0, 0])  # стандартная ошибка углового коэффициента

# Альтернативный способ через polyfit с ковариационной матрицей
coefficients1, covariance1 = np.polyfit(x1, y1, 1, cov=True)  # 1 - степень полинома (линейная)
a1 = coefficients1[0]  # угловой коэффициент
a_error1 = np.sqrt(covariance1[0, 0])  # стандартная ошибка углового коэффициента

# Создаем линию аппроксимации
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit

x_fit1 = np.linspace(min(x1), max(x1), 100)
y_fit1 = a1 * x_fit1

# Доверительный интервал для линии регрессии
# Стандартная ошибка предсказания
y_err = std_err * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

y_err1 = std_err1 * np.sqrt(1/len(x1) + (x_fit1 - np.mean(x1))**2 / np.sum((x1 - np.mean(x1))**2))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit, y_fit, color='red', linewidth=2, 
         label=f'Линейная аппроксимация: k1 = {a:.4f} ± {a_error:.4f}')

plt.scatter(x1, y1, color='green', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit1, y_fit1, color='orange', linewidth=2, 
         label=f'Линейная аппроксимация: k2 = {a:.4f} ± {a_error:.4f}')


plt.xlabel('Момент силы тяжести, Н · м')
plt.ylabel('Угол смещения, °')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('зависимость угла смещения от момента силы тяжести')
plt.show()