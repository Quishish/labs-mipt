import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Данные
x = [1, 2, 3, 4, 5, 6]
y1 = [3248, 6488, 9735, 12993, 16213, 19450]
y2 = [4125, 8247, 12381, 16500, 20633, 24758]
y3 = [4235, 8456, 12700, 17000, 21180, 25400]

# Создаем линии аппроксимации для каждого набора
x_fit = np.linspace(min(x), max(x), 100)

plt.figure(figsize=(12, 8))

# Для каждого набора данных
materials = ['Медь', 'Сталь', 'Дюраль']
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

print("РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ АППРОКСИМАЦИИ:")
print("="*50)

for i, (y, material, color, marker) in enumerate(zip(
    [y1, y2, y3], materials, colors, markers
)):
    # Линейная регрессия с оценкой погрешностей
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Альтернативный способ с ковариационной матрицей
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    a = coefficients[0]
    a_error = np.sqrt(covariance[0, 0])
    
    y_fit = a * x_fit
    
    # Доверительный интервал для линии регрессии
    y_err = std_err * np.sqrt(1/len(x) + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    
    # Вывод статистической информации
    print(f"{material}:")
    print(f"  Угловой коэффициент k = {a:.2f} ± {a_error:.2f} Гц")
    print(f"  Относительная погрешность: {a_error/a*100:.2f}%")
    print(f"  Коэффициент детерминации R² = {r_value**2:.6f}")
    print(f"  Стандартная ошибка: {std_err:.2f} Гц")
    print(f"  Свободный член: {intercept:.2f} Гц")
    print("-" * 30)
    
    # Построение графика
    plt.scatter(x, y, color=color, marker=marker, s=80, label=f'{material} (эксперимент)')
    plt.plot(x_fit, y_fit, color=color, linewidth=2, 
             label=f'{material}: k = {a:.0f} ± {a_error:.0f} Гц')
    plt.fill_between(x_fit, y_fit - 2*y_err, y_fit + 2*y_err, 
                    color=color, alpha=0.2, label=f'{material} (доверительный интервал)')

plt.xlabel('Номер резонансного пика, N', fontsize=12)
plt.ylabel('Резонансная частота f, Гц', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Зависимость резонансной частоты стержня от номера пика', fontsize=14)

# Добавляем информацию о погрешностях на график
plt.text(0.02, 0.98, f'Погрешности угловых коэффициентов:\n'
                     f'Медь: ±{a_error1:.0f} Гц ({a_error1/a1*100:.1f}%)\n'
                     f'Сталь: ±{a_error2:.0f} Гц ({a_error2/a2*100:.1f}%)\n'
                     f'Дюраль: ±{a_error3:.0f} Гц ({a_error3/a3*100:.1f}%)',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=9)

plt.tight_layout()
plt.show()

# Дополнительный анализ
print("\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
print("="*50)

# Сохраняем результаты для каждого материала
results = []
for i, (y, material) in enumerate(zip([y1, y2, y3], materials)):
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    a = coefficients[0]
    a_error = np.sqrt(covariance[0, 0])
    results.append((material, a, a_error))

# Сравнение угловых коэффициентов
print("Сравнение скоростей звука (по угловым коэффициентам):")
for material, a, a_error in results:
    # Скорость звука u = 2L * (df/dN), где df/dN = a
    # Для L = 0.4 м (примерно)
    L = 0.4  # м
    u = 2 * L * a / 1000  # переводим в м/с
    u_error = 2 * L * a_error / 1000
    print(f"{material}: u = {u:.0f} ± {u_error:.0f} м/с")

print(f"\nКачество линейной аппроксимации (R²):")
for i, (y, material) in enumerate(zip([y1, y2, y3], materials)):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{material}: R² = {r_value**2:.6f}")