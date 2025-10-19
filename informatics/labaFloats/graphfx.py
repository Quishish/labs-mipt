import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 400)
y1 = x*x
y2 = (x**3)/3
y3 = 2*x

plt.figure(figsize = (8, 6))

plt.xlabel('Ось абсцисс', fontsize=12)
plt.ylabel('Ось ординат', fontsize=12)
plt.title('Графики функции, её производной и первообразной', fontsize=14)

plt.grid(True, linestyle='-', alpha=0.7, which='both')

plt.plot(x,y1, color = 'blue', label = 'исходная функция')
plt.plot(x,y2, color = 'red', label = 'первообразная')
plt.plot(x,y3, color = 'green', label = 'производная')
plt.legend()

plt.show()