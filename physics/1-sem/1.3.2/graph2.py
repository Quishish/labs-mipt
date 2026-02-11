import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Генерируем случайные данные
np.random.seed(42)  # для воспроизводимости результатов
x1 = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
y1 = [2.47, 2.7, 3.13, 3.619, 4.125, 4.6, 5.04]

x = [i ** 2 for i in x1]
y = [j ** 2 for j in y1]

# Вычисляем коэффициенты линейной регрессии с оценкой погрешностей
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Альтернативный способ через polyfit с ковариационной матрицей
coefficients, covariance = np.polyfit(x, y, 1, cov=True)  # 1 - степень полинома (линейная)
a = coefficients[0]  # угловой коэффициент
a_error = np.sqrt(covariance[0, 0])  # стандартная ошибка углового коэффициента



# Создаем линию аппроксимации
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit + 3.3



# Доверительный интервал для линии регрессии
# Стандартная ошибка предсказания
y_err = std_err * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))


plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit, y_fit, color='red', linewidth=2, 
         label=f'Линейная аппроксимация: k1 = {a:.1f} ± 34.6')



plt.xlabel('Момент силы тяжести, Н · м')
plt.ylabel('Угол смещения, рад')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('зависимость угла смещения от момента силы тяжести')
plt.show()