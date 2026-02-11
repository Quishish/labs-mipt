import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Генерируем случайные данные
np.random.seed(42)  # для воспроизводимости результатов
#x = [0.0275, 0.055, 0.0825, 0.11, 0.1375, 0.165, 0.22, 0.275]
x = [0.02695, 0.0539, 0.08085, 0.1078, 0.13475, 0.1617, 0.2156, 0.2695]
y = [0.0, 0.011929, 0.02662, 0.041191, 0.052775, 0.067418, 0.09703, 0.124383]

x1 = x[::-1]
y1 = [0.124383, 0.100833, 0.07242, 0.059992, 0.044271, 0.028957, 0.015654, 0.000236]

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
y_fit = a * x_fit - 0.015

x_fit1 = np.linspace(min(x1), max(x1), 100)
y_fit1 = a1 * x_fit1 - 0.012

# Доверительный интервал для линии регрессии
# Стандартная ошибка предсказания
y_err = std_err * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

y_err1 = std_err1 * np.sqrt(1/len(x1) + (x_fit1 - np.mean(x1))**2 / np.sum((x1 - np.mean(x1))**2))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit, y_fit, color='red', linewidth=2, 
         label=f'Линейная аппроксимация: k1 = {a:.3f} ± 0.030')

plt.scatter(x1, y1, color='green', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit1, y_fit1, color='orange', linewidth=2, 
         label=f'Линейная аппроксимация: k2 = {a:.3f} ± 0.030')


plt.xlabel('Момент силы тяжести, Н · м')
plt.ylabel('Угол смещения, рад')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('зависимость угла смещения от момента силы тяжести')
plt.show()