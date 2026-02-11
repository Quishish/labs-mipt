import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Функция для аппроксимации
def linear_func(x, a):
    return a * x

# наборы данных
np.random.seed(42)
x_data1 = [29.73, 34.43, 38.78, 53.42, 68.85, 83.38, 98.81, 112.74, 128.25, 142.12]
x_data2 = [19.00, 24.67, 49.28, 73.30, 99.42, 124.12, 147.48, 172.10, 198.52, 233.00]
x_data3 = [22.88, 48.75, 70.8, 97.21, 121.95, 158.94, 194.80, 230.62, 274.55, 318.63]
y_data1 = [150, 175, 200, 275, 350, 425, 505, 575, 655, 730]
y_data2 = [60, 75, 150, 225, 300, 375, 450, 525, 605, 700]
y_data3 = [50, 100, 150, 200, 250, 325, 395, 470, 560, 645]

# Погрешности измерений по оси Y
y_err1 = np.ones_like(y_data1)  # постоянная погрешность
y_err2 = np.ones_like(y_data2)  # постоянная погрешность 
y_err3 = np.ones_like(y_data3)  # постоянная погрешность 

# Погрешности измерений по оси X
x_err1 = np.ones_like(x_data1)  # постоянная погрешность 
x_err2 = np.ones_like(x_data2)  # постоянная погрешность 
x_err3 = np.ones_like(x_data3)  # постоянная погрешность 

# Создание фигуры и оси
fig, ax = plt.subplots(figsize=(10, 6))

# Цвета для разных наборов данных
colors = ['red', 'blue', 'green']
labels = ['l = 50 см, R = 5,1356±0,017 Ом', 'l = 30 см, R = 3,0323±0,008 Ом', 'l = 20 см, R = 2,0361±0,007 Ом']

# Список для хранения угловых коэффициентов
slopes = []

# Обработка каждого набора данных
for i, (x, y, x_err, y_err) in enumerate(zip([x_data1, x_data2, x_data3], 
                                           [y_data1, y_data2, y_data3],
                                           [x_err1, x_err2, x_err3],
                                           [y_err1, y_err2, y_err3])):
    # Аппроксимация данных
    params, covariance = curve_fit(linear_func, x, y, sigma=y_err)
    slope = params[0]  # Угловой коэффициент
    slopes.append(slope)
    
    x_fit = np.linspace(0, max(x) + 1, 100)
    y_fit = linear_func(x_fit, *params)
    
    # Построение графика с погрешностями по обеим осям
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color=colors[i], 
                label=labels[i], capsize=3, capthick=1, elinewidth=1)
    ax.plot(x_fit, y_fit, color=colors[i], linestyle='--', )

# Настройка внешнего вида
ax.set_xlabel('I амперметра, мА')
ax.set_ylabel('V вольтметра, мВ')
ax.set_ylim(0, 800)
ax.set_xlim(0, 350)
plt.minorticks_on()

ax.spines['right'].set_visible(False)  # Прячем правую ось
ax.spines['top'].set_visible(False)  # Прячем верхнюю ось

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.legend()

plt.tight_layout()
plt.show()