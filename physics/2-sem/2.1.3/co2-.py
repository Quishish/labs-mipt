import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# ЛИНЕЙНАЯ МОДЕЛЬ
# ============================================================================
def linear(x, a, b):
    return a * x + b

# ============================================================================
# ФУНКЦИЯ ДЛЯ ПОДГОНКИ И ВОЗВРАТА РЕЗУЛЬТАТОВ
# ============================================================================
def fit_and_predict(x, y, sigma):
    popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
    a, b = popt
    da, db = np.sqrt(np.diag(pcov))
    # Вычисление χ²
    y_fit = linear(x, a, b)
    chi2 = np.sum(((y - y_fit) / sigma) ** 2)
    ndof = len(x) - 2
    chi2_red = chi2 / ndof
    # Гладкая линия для графика
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = linear(x_smooth, a, b)
    return a, b, da, db, chi2, chi2_red, x_smooth, y_smooth

# ============================================================================
# ГЕНЕРАЦИЯ ЧЕТЫРЁХ ТЕСТОВЫХ НАБОРОВ ДАННЫХ (замените своими)
# ============================================================================
np.random.seed(42)
# Набор 1
x1 = np.array([1, 2, 3, 4])
y1 = np.array([31, 100, 165, 230])
sigma1 = np.ones_like(y1) * 2

# Набор 2
x2 = np.array([1,2,3,4])
y2 = np.array([52, 107, 159, 212])
sigma2 = np.ones_like(y2) * 2

# Набор 3
x3 = np.array([1,2,3,4,5])
y3 = np.array([23,71,121,168,216])
sigma3 = np.ones_like(y3) * 2 

# Набор 4
x4 = np.array([1,2,3,4,5])
y4 = np.array([20,64,104,145,187])
sigma4 = np.ones_like(y4) * 2

# Собираем всё в список
datasets = [
    (x1, y1, sigma1, '2 кГц', 'blue'),
    (x2, y2, sigma2, '2.5 кГц', 'red'),
    (x3, y3, sigma3, '2.75 кГц', 'green'),
    (x4, y4, sigma4, '3 кГц', 'purple')
]

# ============================================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ И ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ============================================================================
R = 8.314  # Универсальная газовая постоянная, Дж/(моль·К)
MU_CO2 = 0.044  # Молярная масса CO2, кг/моль
GAMMA_THEORY_CO2 = 1.30  # Теоретическое значение для CO2 (с учётом колебательных степеней)
TEMPERATURE = 293  # Температура газа, К (20°C)
TEMP_ERROR = 1  # Погрешность температуры, К

# ЧАСТОТЫ ДЛЯ CO2 (в порядке возрастания номеров наборов): 2, 2.5, 2.75, 3 кГц
FREQUENCIES_CO2 = np.array([2000, 2500, 2750, 3000])  # Гц
FREQ_ERROR = 5  # Погрешность частоты, Гц

# ============================================================================
# РАСЧЁТ ИСКОМЫХ ВЕЛИЧИН И ПОГРЕШНОСТЕЙ
# ============================================================================
print("=" * 80)
print("ОБРАБОТКА ДАННЫХ: УГЛЕКИСЛЫЙ ГАЗ (CO₂)")
print("=" * 80)
print(f"Температура: {TEMPERATURE} ± {TEMP_ERROR} К")
print(f"Молярная масса: {MU_CO2} кг/моль")
print(f"Теоретическое γ: {GAMMA_THEORY_CO2}")
print("=" * 80)

all_gamma = []
all_dgamma = []
all_c = []
all_dc = []

for i, (x, y, sigma, label, color) in enumerate(datasets):
    # Подгонка линейной зависимости L(n)
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)
    
    # Частота для текущего набора
    f = FREQUENCIES_CO2[i]
    
    # Расчёт длины волны: λ = 2·(λ/2) = 2·a (где a - угловой коэффициент, λ/2 в мм)
    lambda_wave = 2 * a / 1000  # переводим из мм в м
    dlambda = 2 * da / 1000  # погрешность
    
    # Расчёт скорости звука: c = λ·f
    c = lambda_wave * f
    dc = c * np.sqrt((dlambda/lambda_wave)**2 + (FREQ_ERROR/f)**2)
    
    # Расчёт показателя адиабаты: γ = μ·c²/(R·T)
    gamma = MU_CO2 * c**2 / (R * TEMPERATURE)
    
    # Погрешность γ (независимые погрешности)
    dgamma = gamma * np.sqrt(
        (2*dc/c)**2 + 
        (TEMP_ERROR/TEMPERATURE)**2
    )
    
    # Относительная погрешность
    epsilon = abs(gamma - GAMMA_THEORY_CO2) / GAMMA_THEORY_CO2 * 100
    
    # Сохраняем результаты
    all_gamma.append(gamma)
    all_dgamma.append(dgamma)
    all_c.append(c)
    all_dc.append(dc)
    
    # Вывод результатов для каждого набора
    print(f"\n--- {label} (f = {f/1000} кГц) ---")
    print(f"Угловой коэффициент (λ/2): {a:.3f} ± {da:.3f} мм")
    print(f"Длина волны λ: {lambda_wave*1000:.3f} ± {dlambda*1000:.3f} мм")
    print(f"Скорость звука c: {c:.1f} ± {dc:.1f} м/с")
    print(f"Показатель адиабаты γ: {gamma:.4f} ± {dgamma:.4f}")
    print(f"Отклонение от теории: {epsilon:.2f}%")


# ============================================================================
# УСРЕДНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 80)
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
print(f"Теоретическое значение: {GAMMA_THEORY_CO2}")
print(f"Относительное отклонение: {abs(gamma_mean - GAMMA_THEORY_CO2)/GAMMA_THEORY_CO2*100:.2f}%")

# ============================================================================
# ПОСТРОЕНИЕ ОБЩЕГО ГРАФИКА
# ============================================================================
plt.figure(figsize=(10, 8))
for x, y, sigma, label, color in datasets:
    # Подгонка
    a, b, da, db, chi2, chi2_red, xs, ys = fit_and_predict(x, y, sigma)
    # Точки с погрешностями
    plt.errorbar(x, y, yerr=sigma, fmt='o', color=color, capsize=4,
                 markersize=6)
    # Линия регрессии
    plt.plot(xs, ys, '-', color=color, linewidth=2,
             label=f'{label}: y = ({a:.2f}±{da:.2f}) мм')

# Оформление
plt.xlabel('Номер резонанса n', fontsize=12)
plt.ylabel('Удлиннение ΔL, мм', fontsize=12)
plt.title('Зависимость удлиннения трубы от номера резонанса (CO₂)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()
