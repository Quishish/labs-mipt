import matplotlib.pyplot as plt

# Данные для оси X
x_values = [6.908, 8.517, 9.21, 9.616, 10.127, 10.82, 11.225, 11.513, 11.918, 12.429]

# Три различных сценария времени выполнения (в секундах)
# Кривая 1: пузырек
y1 = [-5.142, -1.843, -0.453, 0.356, 1.395, 2.779, 3.588, 4.183, 4.986, 6.004]

# Кривая 2: вставка  
y2 = [-5.967, -2.74, -1.361, -0.552, 0.47, 1.87, 2.664, 3.26, 4.082, 5.096]


# Кривая 3: выбор
y3 = [-2.56, -1.697, -0.315, 0.506, 1.532, 2.915, 3.733, 4.308, 5.126, 6.145]


# Создаем график
plt.figure(figsize=(10, 6))

# Рисуем три кривые с разными стилями
plt.plot(x_values, y1, 'blue', linewidth=2, marker='o', markersize=6, label='selection sort')
plt.plot(x_values, y2, 'red', linewidth=2, marker='s', markersize=6, linestyle='--', label='insertion sort')
plt.plot(x_values, y3, 'green', linewidth=2, marker='^', markersize=6, linestyle='-.', label='bubble sort')

# Настраиваем оси и заголовок
plt.xlabel('Количество элементов массива', fontsize=12)
plt.ylabel('Время (секунды)', fontsize=12)
plt.title('Время выполнения алгоритмов в зависимости от размера данных (log оси)', fontsize=14)

# Устанавливаем пределы оси Y
plt.ylim(-7, 7)

# Настраиваем сетку
plt.grid(True, linestyle='--', alpha=0.7)

# Добавляем легенду
plt.legend(loc='upper left', fontsize=10)

# Настраиваем метки на оси X

plt.tight_layout()
plt.show()