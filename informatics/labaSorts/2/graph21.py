import matplotlib.pyplot as plt
import numpy as np

# Данные для оси X
x_values = [1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000]

# Данные для оси Y (в секундах)
y1 = [157400/10**9, 1000700/10**9, 2314900/10**9, 3601500/10**9, 6400900/10**9, 
      14623700/10**9, 21557100/10**9, 28263200/10**9, 71236900/10**9, 126433400/10**9]

y2 = [1299800/10**9, 59990300/10**9, 103380500/10**9, 132330900/10**9, 415126200/10**9, 
      1775273700/10**9, 7450945500/10**9, 32186253600/10**9, 119704493100/10**9, 250351795400/10**9]

y3 = [181300/10**9, 1169900/10**9, 2462000/10**9, 3885300/10**9, 6925900/10**9, 
      15198100/10**9, 22997200/10**9, 49358000/10**9, 77338300/10**9, 90475400/10**9]

# Создаем график
plt.figure(figsize=(12, 8))

# Рисуем три кривые
plt.plot(x_values, y1, 'blue', linewidth=2, marker='o', markersize=6, label='comb sort')
plt.plot(x_values, y2, 'red', linewidth=2, marker='s', markersize=6, linestyle='--', label='quick sort')
plt.plot(x_values, y3, 'green', linewidth=2, marker='^', markersize=6, linestyle='-.', label='heap sort')

# Логарифмическая шкала по Y
plt.yscale('log')

# Настраиваем оси и заголовок
plt.xlabel('Количество элементов массива', fontsize=12)
plt.ylabel('Время (секунды, логарифмическая шкала)', fontsize=12)
plt.title('Сравнение времени выполнения алгоритмов сортировки', fontsize=14)

# Сетка
plt.grid(True, linestyle='--', alpha=0.7, which='both')

# Легенда
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()