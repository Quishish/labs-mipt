import matplotlib.pyplot as plt

# Данные
x_values = [1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000]
y1 = [2.2785983817190614, 2.349835156982282, 2.513370740394594, 2.4969307095206954, 2.5283433095795327, 2.70314226698538, 2.560550270041974, 2.4549103601855973, 3.9847046776551633, 4.068909833014243]
y3 = [2.624586318968652, 2.7471491457515453, 2.673082536114515, 2.6936900973763036, 2.7357173097247083, 2.8093181949759987, 2.731605209894155, 4.28718140755614, 4.325992368728823, 2.911700901074375]

plt.figure(figsize=(12, 8))

plt.plot(x_values, y1, 'blue', linewidth=2, marker='o', markersize=8, label='comb sort')
plt.plot(x_values, y3, 'green', linewidth=2, marker='^', markersize=8, linestyle='-.', label='heap sort')

plt.xlabel('Количество элементов массива', fontsize=12)
plt.ylabel('Время (секунды)', fontsize=12)
plt.title('Сравнение Comb sort и Heap sort', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Автоматическое масштабирование
plt.ylim(0, 5)

plt.tight_layout()
plt.show()