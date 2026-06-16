import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ========================
# Параметры установки
# ========================
V1 = 220.0          # cm³
V2 = 220.0          # cm³
d = 0.95            # cm (9.5 мм)
L = 1.10            # cm (11.0 мм)
S = np.pi * (d/2)**2   # см²

# Погрешности геометрических параметров
sigma_V = 20        # cm³
sigma_d = 0.01      # cm
sigma_L = 0.01      # cm

# Погрешность измерения напряжения (мВ) – фиксированная (как в твоём примере)
sigma_U = 0.01     # мВ

# Данные по давлениям (торр) и имена файлов
data_list = [
    (11.5, "10(11.5).csv"),
    (22.0, "20_2(22).csv"),
    (33.5, "30_2(33.5).csv"),
    (44.0, "40_2(44).csv"),
    (55.0, "50_2(55).csv")
]

# ========================
# Модели
# ========================
def exp_decay(t, U0, tau):
    return U0 * np.exp(-t / tau)

def calc_chi2_exp(t, U, U0_fit, tau_fit, sigma_U):
    """Расчёт хи-квадрат для экспоненциальной модели (явный цикл, как в примере)"""
    chi2 = 0.0
    for i in range(len(t)):
        U_pred = U0_fit * np.exp(-t[i] / tau_fit)
        chi2 += (U[i] - U_pred)**2 / (sigma_U**2)
    return chi2

# ========================
# Функция обработки одного файла
# ========================
def process_file(idx, P, fname, sigma_U):
    df = pd.read_csv(fname, names=['t', 'U'], skiprows=1)
    t = df['t'].values
    U = df['U'].values

    # Начальные приближения
    U0_guess = U[0]
    if U[-1] > 0 and U[0] > U[-1]:
        tau_guess = (t[-1] - t[0]) / np.log(U0_guess / U[-1])
    else:
        tau_guess = (t[-1] - t[0]) / 3.0

    # Экспоненциальная подгонка
    try:
        popt, pcov = curve_fit(exp_decay, t, U, p0=[U0_guess, tau_guess])
        U0_fit, tau_fit = popt
        tau_err = np.sqrt(pcov[1, 1]) if (len(pcov) > 1 and pcov[1, 1] > 0) else 0.0
    except Exception:
        logU = np.log(U)
        slope, intercept, r_val, p_val, std_err = stats.linregress(t, logU)
        tau_fit = -1.0 / slope
        tau_err = std_err / (slope * slope) if slope != 0 else 0.0
        U0_fit = np.exp(intercept)

    # R²
    U_pred = exp_decay(t, U0_fit, tau_fit)
    ss_res = np.sum((U - U_pred)**2)
    ss_tot = np.sum((U - np.mean(U))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Хи-квадрат (явный цикл, как в примере)
    chi2 = calc_chi2_exp(t, U, U0_fit, tau_fit, sigma_U)
    dof = len(t) - 2
    chi2_red = chi2 / dof if dof > 0 else np.inf

    # Расчёт Dn
    V_eff = (V1 * V2) / (V1 + V2)
    Dn = V_eff * (L / (S * tau_fit))

    # Погрешность Dn
    rel_err_tau = tau_err / tau_fit if tau_fit != 0 else 0
    rel_err_V = np.sqrt(2) * sigma_V / V1
    rel_err_S = 2 * sigma_d / d
    rel_err_L = sigma_L / L
    rel_err_Dn = np.sqrt(rel_err_tau**2 + rel_err_V**2 + rel_err_S**2 + rel_err_L**2)
    Dn_err = Dn * rel_err_Dn

    # Построение графиков (как раньше)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t, U, 'bo', markersize=3, label='Эксперимент')
    t_smooth = np.linspace(t[0], t[-1], 500)
    ax1.plot(t_smooth, exp_decay(t_smooth, U0_fit, tau_fit), 'r-',
             label=f'τ = {tau_fit:.1f} с, R² = {R2:.4f}, χ²_red = {chi2_red:.2f}')
    ax1.set_xlabel('Время t, с')
    ax1.set_ylabel('Напряжение U, мВ')
    ax1.set_title(f'P = {P} торр, файл: {fname}')
    ax1.grid(True)
    ax1.legend()

    logU = np.log(U)
    ax2.plot(t, logU, 'bo', markersize=3, label='Эксперимент')
    slope_lin, intercept_lin, r_val, p_val, std_err = stats.linregress(t, logU)
    ax2.plot(t, intercept_lin + slope_lin * t, 'r-',
             label=f'slope = {slope_lin:.4f} c⁻¹, R² = {r_val**2:.4f}')
    ax2.set_xlabel('Время t, с')
    ax2.set_ylabel('ln(U)')
    ax2.set_title(f'Полулогарифмический график, P = {P} торр')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    base_name = os.path.splitext(fname)[0]
    plt.savefig(f'fit_{idx+1:02d}_{base_name}.png', dpi=150)
    plt.close()

    print(f"Файл {fname}: P={P} торр, τ={tau_fit:.1f}±{tau_err:.1f} с, "
          f"R²={R2:.4f}, χ²_red={chi2_red:.2f}, Dn={Dn:.5f}±{Dn_err:.5f} см²/с")
    return P, tau_fit, tau_err, Dn, Dn_err, R2, chi2_red

# ========================
# Обработка файлов
# ========================
print("=== Обработка экспериментальных данных ===")
results = []
for idx, (P, fname) in enumerate(data_list):
    if not os.path.exists(fname):
        print(f"Файл {fname} не найден, пропускаем.")
        continue
    res = process_file(idx, P, fname, sigma_U)
    results.append(res)

if not results:
    print("Нет данных.")
    exit()

# Сводная таблица
print("\n=== Сводная таблица результатов ===")
print(f"{'P, торр':<10} {'τ, с':<15} {'Dn, см²/с':<20} {'1/Dn, с/см²':<15} {'R²':<8} {'χ²_red':<8}")
print("-" * 85)
for P, tau, tau_err, Dn, Dn_err, R2, chi2_red in results:
    print(f"{P:<10.1f} {tau:>6.1f}±{tau_err:<4.1f}   {Dn:.5f}±{Dn_err:.5f}   {1/Dn:<8.2f}   {R2:<8.4f} {chi2_red:<8.2f}")

# ========================
# График 1/Dn(P) и линейная регрессия
# ========================
P_vals = np.array([r[0] for r in results])
invDn_vals = 1.0 / np.array([r[3] for r in results])
invDn_err = invDn_vals * (np.array([r[4] for r in results]) / np.array([r[3] for r in results]))

# Линейная регрессия (обычная, без весов)
slope, intercept, r_value, p_value, std_err = stats.linregress(P_vals, invDn_vals)
print(f"\n=== Линейная регрессия 1/Dn(P) ===")
print(f"1/Dn = {slope:.4f} * P + {intercept:.2f}")
print(f"R² = {r_value**2:.4f}")

# Расчёт хи-квадрат для линейной модели (с использованием погрешностей invDn_err, как в твоём стиле – но с переменной погрешностью)
# Если хочешь использовать константную погрешность – замени invDn_err[i] на какую-то константу, например 5.0
chi2_lin = 0.0
for i in range(len(P_vals)):
    chi2_lin += (invDn_vals[i] - (slope * P_vals[i] + intercept))**2 / (invDn_err[i]**2)
dof_lin = len(P_vals) - 2
chi2_red_lin = chi2_lin / dof_lin if dof_lin > 0 else np.inf
print(f"χ²_red (с учётом погрешностей invDn_err) = {chi2_red_lin:.2f}")

# Альтернативный вариант: с фиксированной погрешностью (как в твоём примере, где sigma=0.03)
# Здесь можно взять, например, среднюю погрешность или какую-то характерную, например 5 с/см²
sigma_const = 5.0  # пример (можно подобрать)
chi2_lin_const = 0.0
for i in range(len(P_vals)):
    chi2_lin_const += (invDn_vals[i] - (slope * P_vals[i] + intercept))**2 / (sigma_const**2)
chi2_red_lin_const = chi2_lin_const / dof_lin if dof_lin > 0 else np.inf
print(f"χ²_red (с константной погрешностью {sigma_const}) = {chi2_red_lin_const:.2f}")

# Построение графика
plt.figure(figsize=(8, 6))
plt.errorbar(P_vals, invDn_vals, yerr=invDn_err, fmt='o', capsize=3, label='Эксперимент')
P_fit = np.linspace(0, max(P_vals) * 1.1, 100)
plt.plot(P_fit, slope * P_fit + intercept, 'r-',
         label=f'Линейная аппроксимация: 1/D = {slope:.4f}·P + {intercept:.2f}\n'
               f'R² = {r_value**2:.4f}, χ²_red = {chi2_red_lin:.2f}')
plt.xlabel('Давление P, торр')
plt.ylabel('1/D_n, с/см²')
plt.title('Зависимость 1/D_n от давления')
plt.grid(True)
plt.legend()
plt.savefig('invDn_vs_P.png', dpi=150)
plt.show()

# ========================
# Расчёт параметров пористой среды
# ========================
D0 = 0.30          # см²/с при 760 торр
a = slope
b = intercept
delta_xi2 = 1.0 / (a * D0 * 760)
D_Kn = 1.0 / b
T = 293
R_gas = 8.314
mu_He = 0.004
v = np.sqrt(8 * R_gas * T / (np.pi * mu_He)) * 100
d_pore = 3 * D_Kn / (delta_xi2 * v)

print("\n=== Параметры пористой среды ===")
print(f"δξ² = {delta_xi2:.4f}")
print(f"D_Kn = {D_Kn:.4f} см²/с")
print(f"Средняя скорость He: v = {v:.0f} см/с")
print(f"Диаметр пор: d_пор = {d_pore:.2e} см = {d_pore*1e4:.2f} мкм")