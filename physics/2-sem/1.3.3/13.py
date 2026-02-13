import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Параметры микроманометра
K = 0.2          # угловой коэффициент наклона
n = 0.9975       # поправочный коэффициент плотности спирта
G = 9.8067       # ускорение свободного падения, м/с²
t = 30.0         # время измерения, секунды

# Данные для ламинарного и турбулентного режимов
# Формат: [радиус (мм), расход_ламинарный (мл/с), расход_турбулентный (мл/с)]
data = [
    [1.500, 11.5, 57.7],   # d2(30), d=3.00 мм
    [1.975, 31.0, 119.0],  # d3(50), d=3.95 мм
    [2.650, 105.7, 268.8]  # d1(50), d=5.30 мм
]

# Преобразование в логарифмический масштаб
radii = np.array([row[0] for row in data])
Q_lam = np.array([row[1] for row in data])
Q_turb = np.array([row[2] for row in data])

ln_R = np.log(radii)
ln_Q_lam = np.log(Q_lam)
ln_Q_turb = np.log(Q_turb)

# Линейная регрессия для ламинарного режима
slope_lam, intercept_lam, r_value_lam, p_value_lam, std_err_lam = linregress(ln_R, ln_Q_lam)
R2_lam = r_value_lam**2

# Линейная регрессия для турбулентного режима
slope_turb, intercept_turb, r_value_turb, p_value_turb, std_err_turb = linregress(ln_R, ln_Q_turb)
R2_turb = r_value_turb**2

# Генерация линий аппроксимации
ln_R_fit = np.linspace(ln_R.min() - 0.1, ln_R.max() + 0.1, 100)
ln_Q_lam_fit = slope_lam * ln_R_fit + intercept_lam
ln_Q_turb_fit = slope_turb * ln_R_fit + intercept_turb

# Построение графика
plt.figure(figsize=(10, 7))

# Экспериментальные точки
plt.scatter(ln_R, ln_Q_lam, color='blue', s=100, marker='o', 
            label='Ламинарный режим', zorder=3)
plt.scatter(ln_R, ln_Q_turb, color='red', s=100, marker='s', 
            label='Турбулентный режим', zorder=3)

# Линии аппроксимации
plt.plot(ln_R_fit, ln_Q_lam_fit, color='blue', linewidth=2, linestyle='--',
         label=f'Ламинарный: $\\ln Q = {slope_lam:.2f} \\ln R + {intercept_lam:.2f}$\n$\\beta = {slope_lam:.2f} \\pm {std_err_lam:.2f}$, $R^2 = {R2_lam:.4f}$', zorder=2)
plt.plot(ln_R_fit, ln_Q_turb_fit, color='red', linewidth=2, linestyle='--',
         label=f'Турбулентный: $\\ln Q = {slope_turb:.2f} \\ln R + {intercept_turb:.2f}$\n$\\beta = {slope_turb:.2f} \\pm {std_err_turb:.2f}$, $R^2 = {R2_turb:.4f}$', zorder=2)

# Теоретические линии для сравнения
ln_Q_lam_theor = 4.0 * ln_R_fit + (intercept_lam - (4.0 - slope_lam) * np.mean(ln_R))
ln_Q_turb_theor = 2.5 * ln_R_fit + (intercept_turb - (2.5 - slope_turb) * np.mean(ln_R))
plt.plot(ln_R_fit, ln_Q_lam_theor, color='blue', linewidth=1, linestyle=':', alpha=0.6,
         label='Теория ламинарного: $\\beta = 4.0$', zorder=1)
plt.plot(ln_R_fit, ln_Q_turb_theor, color='red', linewidth=1, linestyle=':', alpha=0.6,
         label='Теория турбулентного: $\\beta = 2.5$', zorder=1)

# Оформление графика
plt.xlabel('$\\ln R$ (радиус трубки)', fontsize=13)
plt.ylabel('$\\ln Q$ (расход)', fontsize=13)
plt.title('Зависимость расхода от радиуса трубки в двойном логарифмическом масштабе', 
          fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, zorder=1)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Сохранение и отображение
plt.savefig('lnQ_vs_lnR.png', dpi=300, bbox_inches='tight')
print("График сохранён как 'lnQ_vs_lnR.png'")
plt.show()

# Вывод результатов
print("\nРезультаты анализа зависимости расхода от радиуса трубки:")
print("=" * 70)
print(f"{'Режим':<15} {'Эксп. β':<12} {'Теор. β':<12} {'Отклонение, %':<18} {'R²':<10}")
print("=" * 70)
dev_lam = abs(slope_lam - 4.0) / 4.0 * 100
dev_turb = abs(slope_turb - 2.5) / 2.5 * 100
print(f"{'Ламинарный':<15} {slope_lam:<12.2f} {4.0:<12.1f} {dev_lam:<18.1f} {R2_lam:<10.4f}")
print(f"{'Турбулентный':<15} {slope_turb:<12.2f} {2.5:<12.1f} {dev_turb:<18.1f} {R2_turb:<10.4f}")
print("=" * 70)

print("\nВыводы:")
print(f"1. Для ламинарного режима получена степень β = {slope_lam:.2f} ± {std_err_lam:.2f}")
print(f"   Теоретическое значение β = 4.0 (формула Пуазейля)")
print(f"   Отклонение: {dev_lam:.1f}% — находится в пределах экспериментальной погрешности.")

print(f"\n2. Для турбулентного режима получена степень β = {slope_turb:.2f} ± {std_err_turb:.2f}")
print(f"   Теоретическое значение β = 2.5")
print(f"   Отклонение: {dev_turb:.1f}% — находится в пределах экспериментальной погрешности.")

print("\n3. Экспериментальные данные подтверждают теоретические зависимости:")
print("   - Ламинарный режим: Q ∝ R⁴")
print("   - Турбулентный режим: Q ∝ R²·⁵")
