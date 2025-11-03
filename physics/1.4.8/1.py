import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Функция для аппроксимации
def linear_func(x, a):
    return a * x

x_data1 = [1, 2, 3, 4, 5, 6]
x_data2 = [1, 2, 3, 4, 5, 6]
x_data3 = [1, 2, 3, 4, 5, 6]
y_data1 = [3.248, 6.488, 9.735, 12.993, 16.213, 19.450]
y_data2 = [4.125, 8.247, 12.381, 16.500, 20.633, 24.758]
y_data3 = [4.235, 8.456, 12.700, 17.000, 21.180, 25.400]

# Создание фигуры и оси
fig, ax = plt.subplots(figsize=(10, 6))

# Цвета для разных наборов данных
colors = ['red', 'blue', 'green']
labels = []

# Список для хранения угловых коэффициентов
slopes = []

# Обработка каждого набора данных
for i, (x, y) in enumerate(zip([x_data1, x_data2, x_data3], [y_data1, y_data2, y_data3],)):
    # Аппроксимация данных
    params, covariance = curve_fit(linear_func, x, y)
    slope = params[0]  # Угловой коэффициент
    slopes.append(slope)
    
    x_fit = np.linspace(0, max(x) + 1, 100)
    y_fit = linear_func(x_fit, *params)
    
    # Построение графика с погрешностями по обеим осям
    ax.errorbar(x, y, fmt='o', color=colors[i], 
                label=labels[i], capsize=3, capthick=1, elinewidth=1)
    ax.plot(x_fit, y_fit, color=colors[i], )

# Настройка внешнего вида
ax.set_xlabel('I амперметра, мА')
ax.set_ylabel('V вольтметра, мВ')

plt.minorticks_on()

ax.spines['right'].set_visible(False)  # Прячем правую ось
ax.spines['top'].set_visible(False)  # Прячем верхнюю ось

ax.xaxis.set_ticks_position('bottom')
plt.grid(True, alpha=0.3)
ax.yaxis.set_ticks_position('left')
ax.legend()

plt.tight_layout()
plt.show()