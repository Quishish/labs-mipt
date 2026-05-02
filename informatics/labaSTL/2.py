# plot_lab2.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Загружаем данные
df_std = pd.read_csv("results/test2_erase-vector.csv")
df_custom = pd.read_csv("results/test2_erase-subvector.csv")

plt.figure(figsize=(10, 6))

# Рисуем графики
plt.loglog(df_std['size'], df_std['avg_time_ms'], 'o-', label='std::vector', color='blue', linewidth=2)
plt.loglog(df_custom['size'], df_custom['avg_time_ms'], 's--', label='subvector', color='red', linewidth=2)

# Референсная линия O(n) — масштабируем через первую точку std::vector
k = df_std['avg_time_ms'].iloc[0] / df_std['size'].iloc[0]
plt.loglog(df_std['size'], k * df_std['size'], 'k:', label='O(n) референс', alpha=0.5)

# Оценка асимптотики: линейная регрессия в log-log
log_x = np.log(df_std['size'])
log_y = np.log(df_std['avg_time_ms'])
slope, _ = np.polyfit(log_x, log_y, 1)
print(f"📈 std::vector: наклон ≈ {slope:.3f} → асимптотика O(N^{slope:.2f})")

# Оформление
plt.xlabel('Размер контейнера (N)')
plt.ylabel('Среднее время удаления, мс')
plt.title('Пункт 2: Удаление из произвольного места вектора')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.3)
plt.tight_layout()

# Сохраняем и выводим
plt.savefig("results/plot_lab2_erase.png", dpi=300)
print("✅ График сохранён: results/plot_lab2_erase.png")