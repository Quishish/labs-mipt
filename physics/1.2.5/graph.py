import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные данные
np.random.seed(42)  # для воспроизводимости результатов
x = [0.09, 0.13, 0.255, 0.397]
y = [0.046, 0.068, 0.132, 0.205]

# Вычисляем коэффициенты линейной регрессии
coefficients = np.polyfit(x, y, 1)  # 1 - степень полинома (линейная)
a = coefficients[0]  # угловой коэффициент
b = coefficients[1]  # свободный член

# Создаем линию аппроксимации
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit + b

# Строим график
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue')
plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Угловой коэффициент: k = {a:.2f}x')
plt.xlabel('Момент силы тяжести, Н · м')
plt.ylabel('Угловая скорость прецессии 1/с')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Выводим коэффициенты
print(f"Уравнение прямой: y = {a:.4f}x + {b:.4f}")