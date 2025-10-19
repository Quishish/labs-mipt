import matplotlib.pyplot as plt
import numpy as np

x = [10, 100, 500, 1000, 10000, 20000, 100000, 500000, 1000000, 10000000]
y = [63.125, 70.794, 71.492, 71.579, 71.658, 71.662, 71.666, 71.666, 71.667, 71.667]

plt.figure(figsize=(10, 6))

plt.xlabel('Количество прямоугольников', fontsize=12)
plt.ylabel('Площадь', fontsize=12)
plt.title('Зависимость значения площади от количества прямоугольников', fontsize=14)

plt.xscale('log')
plt.grid(True, linestyle=':', alpha=0.5)

plt.plot(x, y, color='red', linewidth=2, marker='.', markersize=10,
         label='Вычисленная площадь')

plt.legend()

# Красивые подписи для оси X
plt.xticks(x, [f'$10^{int(np.log10(val))}$' if val in [10, 100, 1000, 10000, 100000, 1000000, 10000000] 
               else f'{val:,}' for val in x], rotation = 45)

plt.tight_layout()
plt.show()