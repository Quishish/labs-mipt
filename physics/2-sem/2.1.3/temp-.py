import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# ЛИНЕЙНАЯ МОДЕЛЬ
# ============================================================================
def linear(x, a, b):
    return a * x + b

# ============================================================================
# ФУНКЦИЯ ДЛЯ ПОДГОНКИ
# ============================================================================
def fit_and_predict(x, y, sigma):
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    y_fit = linear(x, a, b)
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    ndof = len(x) - 2
    chi2_red = chi2 / ndof
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = linear(x_smooth, a, b)
    return a, b, da, db, chi2, chi2_red, x_smooth, y_smooth

# ============================================================================
# ВАШИ ДАННЫЕ (5 НАБОРОВ)
# ============================================================================
x1 = np.array([1, 2, 3, 4, 5])
y1 = np.array([1314, 1530, 1745, 1961, 2178])
sigma1 = np.ones_like(x1) * 0.5

x2 = np.array([1, 2, 3, 4, 5])
y2 = np.array([1326, 1542, 1760, 1976, 2194])
sigma2 = np.ones_like(x2) * 1.0

x3 = np.array([1, 2, 3, 4, 5])
y3 = np.array([1333, 1555, 1773, 1992, 2211])
sigma3 = np.ones_like(x3) * 0.8

x4 = np.array([1, 2, 3, 4, 5])
y4 = np.array([1361, 1578, 1800, 2023, 2246])
sigma4 = np.ones_like(x4) * 0.3

x5 = np.array([1, 2, 3, 4, 5])
y5 = np.array([1150, 1379, 1603, 1828, 2054])
sigma5 = np.ones_like(x5) * 0.6

datasets = [
    (x1, y1, sigma1, '25 °C', 'blue'),
    (x2, y2, sigma2, '30 °C', 'red'),
    (x3, y3, sigma3, '35 °C', 'green'),
    (x4, y4, sigma4, '45 °C', 'purple'),
    (x5, y5, sigma5, '55 °C', 'orange')
]

# ============================================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ И ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ============================================================================
R = 8.314  # Универсальная газовая постоянная, Дж/(моль·К)
MU = 0.029  # Молярная масса воздуха, кг/моль
GAMMA_THEORY = 1.40  # Теоретическое значение для двухатомного газа

# ТЕМПЕРАТУРЫ (в градусах Цельсия, в порядке возрастания номеров наборов)
TEMPERATURES_C = np.array([25, 30, 35, 45, 55])  # °C
TEMPERATURES_K = TEMPERATURES_C + 273.15  # Перевод в Кельвины

# ДЛИНА ТРУБЫ (фиксированная, измеренная отдельно)
# ЗАМЕНИТЕ НА ВАШЕ ЗНАЧЕНИЕ!
L_TUBE = 0.800  # м (примерное значение, укажите точное)
L_ERROR = 0.001  # м (погрешность измерения длины)

# ПОГРЕШНОСТИ
FREQ_ERROR = 5  # Погрешность частоты, Гц
TEMP_ERROR = 0.5  # Погрешность температуры, К

# ============================================================================
# РАСЧЁТ ИСКОМЫХ ВЕЛИЧИН И ПОГРЕШНОСТЕЙ
# ============================================================================
print("=" * 80)
print("ОБРАБОТКА ДАННЫХ: ЗАВИСИМОСТЬ ОТ ТЕМПЕРАТУРЫ (ФИКСИРОВАННАЯ ТРУБА)")
print("=" * 80)
print(f"Длина трубы: {L_TUBE:.3f} ± {L_ERROR:.3f} м")
print(f"Молярная масса: {MU} кг/моль")
print(f"Теоретическое γ: {GAMMA_THEORY}")
print("=" * 80)

all_gamma = []
all_dgamma = []
all_c = []
all_dc = []
all_slope = []
all_dslope = []

print("\n--- РЕЗУЛЬТАТЫ ДЛЯ КАЖДОЙ ТЕМПЕРАТУРЫ ---\n")

for i, (x, y, sigma, label, color) in enumerate(datasets):
    # Подгонка линейной зависимости f(n)
    # Для фиксированной трубы: f = n·c/(2L), значит угловой коэффициент a = c/(2L)
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)
    
    # Температура для текущего набора
    T_C = TEMPERATURES_C[i]
    T_K = TEMPERATURES_K[i]
    
    # Расчёт скорости звука: c = 2L·a (где a - угловой коэффициент в Гц)
    c = 2 * L_TUBE * a
    dc = c * np.sqrt((L_ERROR/L_TUBE)**2 + (da/a)**2)
    
    # Расчёт показателя адиабаты: γ = μ·c²/(R·T)
    gamma = MU * c**2 / (R * T_K)
    
    # Погрешность γ (независимые погрешности)
    dgamma = gamma * np.sqrt(
        (2*dc/c)**2 + 
        (TEMP_ERROR/T_K)**2 +
        (L_ERROR/L_TUBE)**2
    )
    
    # Относительная погрешность
    epsilon = abs(gamma - GAMMA_THEORY) / GAMMA_THEORY * 100
    
    # Сохраняем результаты
    all_gamma.append(gamma)
    all_dgamma.append(dgamma)
    all_c.append(c)
    all_dc.append(dc)
    all_slope.append(a)
    all_dslope.append(da)
    
    # Вывод результатов для каждого набора
    print(f"{label} (T = {T_C}°C = {T_K} К)")
    print(f"  Угловой коэффициент (c/2L): {a:.2f} ± {da:.2f} Гц")
    print(f"  Скорость звука c: {c:.1f} ± {dc:.1f} м/с")
    print(f"  Показатель адиабаты γ: {gamma:.4f} ± {dgamma:.4f}")
    print(f"  Отклонение от теории: {epsilon:.2f}%")
    print()

# ============================================================================
# УСРЕДНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("=" * 80)
print("УСРЕДНЁННЫЕ РЕЗУЛЬТАТЫ")
print("=" * 80)

# Среднее значение γ
gamma_mean = np.mean(all_gamma)
gamma_std = np.std(all_gamma, ddof=1)  # стандартное отклонение
gamma_error = np.sqrt(np.sum(np.array(all_dgamma)**2) / len(all_dgamma)**2)  # погрешность среднего

# Среднее значение скорости звука
c_mean = np.mean(all_c)
c_std = np.std(all_c, ddof=1)

print(f"\nСредняя скорость звука: c = {c_mean:.1f} ± {c_std:.1f} м/с")
print(f"Среднее значение γ: {gamma_mean:.4f} ± {gamma_error:.4f}")
print(f"Стандартное отклонение γ: {gamma_std:.4f}")
print(f"Теоретическое значение: {GAMMA_THEORY}")
print(f"Относительное отклонение: {abs(gamma_mean - GAMMA_THEORY)/GAMMA_THEORY*100:.2f}%")

# ============================================================================
# ПРОВЕРКА ЗАВИСИМОСТИ СКОРОСТИ ЗВУКА ОТ ТЕМПЕРАТУРЫ
# ============================================================================
print("\n" + "=" * 80)
print("ПРОВЕРКА ЗАВИСИМОСТИ c(T)")
print("=" * 80)

# Теоретическая зависимость: c ∝ √T
# Строим график c² от T
T_array = np.array(TEMPERATURES_K)
c_array = np.array(all_c)
c2_array = c_array ** 2

# Линейная подгонка c² = k·T
k_fit = np.mean(c2_array / T_array)
k_error = np.std(c2_array / T_array) / np.sqrt(len(T_array))

print(f"\nПроверка зависимости c² ∝ T:")
print(f"  Угловой коэффициент k = c²/T: {k_fit:.2f} ± {k_error:.2f} м²/(с²·К)")
print(f"  Теоретическое значение (γR/μ): {GAMMA_THEORY * R / MU:.2f} м²/(с²·К)")
print(f"  Отклонение: {abs(k_fit - GAMMA_THEORY * R / MU)/(GAMMA_THEORY * R / MU)*100:.2f}%")

# ============================================================================
# ПОСТРОЕНИЕ ОБЩЕГО ГРАФИКА
# ============================================================================
plt.figure(figsize=(10, 8))
for x, y, sigma, label, color in datasets:
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)
    # Точки с погрешностями (увеличенные кресты)
    plt.errorbar(x, y, yerr=sigma, fmt='o', color=color, capsize=4,
                 markersize=6)

    # Линия регрессии
    plt.plot(xs, ys, '-', color=color, linewidth=2,
             label=f'{label}: y = ({a:.2f}±{da:.2f}) с^-1')

# Оформление
plt.xlabel('Номер резонанса n', fontsize=12)
plt.ylabel('Частота f, Гц', fontsize=12)
plt.title('Зависимость резонансной частоты от номера резонанса (разные температуры)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: c² от T
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(T_array, c2_array, 'o', markersize=8, label='Экспериментальные данные')

# Теоретическая прямая
T_theory = np.linspace(min(T_array)*0.95, max(T_array)*1.05, 100)
c2_theory = (GAMMA_THEORY * R / MU) * T_theory
plt.plot(T_theory, c2_theory, '--', linewidth=2, label=f'Теория: c² = {GAMMA_THEORY * R / MU:.1f}·T')

# Экспериментальная прямая
c2_fit = k_fit * T_theory
plt.plot(T_theory, c2_fit, '-', linewidth=2, label=f'Подгонка: c² = {k_fit:.1f}·T')

plt.xlabel('Температура T, К', fontsize=12)
plt.ylabel('c², м²/с²', fontsize=12)
plt.title('Проверка зависимости c² ∝ T', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()
