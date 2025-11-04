import numpy as np
import matplotlib.pyplot as plt

# Данные
x = [1, 2, 3, 4, 5, 6]
y1 = [3248, 6488, 9735, 12993, 16213, 19450]
y2 = [4125, 8247, 12381, 16500, 20633, 24758]
y3 = [4235, 8456, 12700, 17000, 21180, 25400]



# Создаем линии аппроксимации для каждого набора
x_fit = np.linspace(min(x), max(x), 100)

plt.figure(figsize=(10, 6))

# Для каждого набора данных
for i, (y, color, marker) in enumerate(zip(
    [y1, y2, y3], 
    ['blue', 'green', 'red'],
    ['o', 's', '^']
)):
    coefficients = np.polyfit(x, y, 1)
    a = coefficients[0]
    y_fit = a * x_fit

    labels = [f'Медь: k{i+1} = {a:.2f}', f'Сталь: k{i+1} = {a:.2f}', f'Дюраль: k{i+1} = {a:.2f}']
    
    plt.scatter(x, y, color=color, marker=marker, s=60)
    plt.plot(x_fit, y_fit, color=color, alpha=0.7,
             label=labels[i])

plt.xlabel('Номер резонансного пика, N')
plt.ylabel('Резонансная частота f, Гц')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Зависимость резонансной частоты стержня от номера пика')
plt.show()
