import matplotlib.pyplot as plt

# Данные для оси X
x_values = [1000, 5000, 10000, 50000, 100000]

# Три различных сценария времени выполнения (в секундах)
# Кривая 1: Быстрый алгоритм
y1 = [3, 12, 20, 35, 50]

# Кривая 2: Средний алгоритм  
y2 = [7, 25, 40, 68, 85]

# Кривая 3: Медленный алгоритм
y3 = [7088600/10**9, 185889600/10**9, 750150900/10**8, 19338211400/10**9, 87252395700/10**9]

# Создаем график
plt.figure(figsize=(10, 6))

# Рисуем три кривые с разными стилями
plt.plot(x_values, y1, 'blue', linewidth=2, marker='o', markersize=6, label='Быстрый алгоритм')
plt.plot(x_values, y2, 'red', linewidth=2, marker='s', markersize=6, linestyle='--', label='Средний алгоритм')
plt.plot(x_values, y3, 'green', linewidth=2, marker='^', markersize=6, linestyle='-.', label='Медленный алгоритм')

# Настраиваем оси и заголовок
plt.xlabel('Количество элементов', fontsize=12)
plt.ylabel('Время (секунды)', fontsize=12)
plt.title('Время выполнения алгоритмов в зависимости от размера данных', fontsize=14)

# Устанавливаем пределы оси Y
plt.ylim(0, 90)

# Настраиваем сетку
plt.grid(True, linestyle='--', alpha=0.7)

# Добавляем легенду
plt.legend(loc='upper left', fontsize=10)

# Настраиваем метки на оси X
plt.xscale('log')
plt.xticks(x_values, [f'{x:,}' for x in x_values])

# Добавляем подписи значений на кривых (опционально)
for i, (x, y) in enumerate(zip(x_values, y1)):
    plt.annotate(f'{y}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

for i, (x, y) in enumerate(zip(x_values, y2)):
    plt.annotate(f'{y}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

for i, (x, y) in enumerate(zip(x_values, y3)):
    plt.annotate(f'{y}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.show()