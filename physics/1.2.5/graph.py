import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Генерируем случайные данные
np.random.seed(42)  # для воспроизводимости результатов
x = [0.09, 0.13, 0.255, 0.397]
y = [0.046, 0.068, 0.132, 0.205]

# Вычисляем коэффициенты линейной регрессии с оценкой погрешностей
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Альтернативный способ через polyfit с ковариационной матрицей
coefficients, covariance = np.polyfit(x, y, 1, cov=True)  # 1 - степень полинома (линейная)
a = coefficients[0]  # угловой коэффициент
a_error = np.sqrt(covariance[0, 0])  # стандартная ошибка углового коэффициента

# Создаем линию аппроксимации
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit

# Доверительный интервал для линии регрессии
# Стандартная ошибка предсказания
y_err = std_err * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', zorder=5, label='Экспериментальные точки')
plt.plot(x_fit, y_fit, color='red', linewidth=2, 
         label=f'Линейная аппроксимация: k = {a:.4f} ± {a_error:.4f}')

plt.xlabel('Момент силы тяжести, Н · м')
plt.ylabel('Угловая скорость прецессии 1/с')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Зависимость угловой скорости прецессии от момента силы')
plt.show()