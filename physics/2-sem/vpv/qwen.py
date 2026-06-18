import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import os

# Параметры установки
V1 = 220.0   # cm^3
V2 = 220.0   # cm^3
d = 0.95     # cm
L = 1.10     # cm
S = np.pi * (d/2)**2  # cm^2
sigma_V = 20   # cm^3
sigma_d = 0.01 # cm
sigma_L = 0.01 # cm

# Данные по давлениям (торр) и имена файлов (выбраны вторые измерения)
data_list = [
    (11.0,  "10_2(11).csv"),
    (22.0,  "20_2(22).csv"),
    (33.5,  "30_2(33.5).csv"),
    (44.0,  "40_2(44).csv"),
    (55.0,  "50_2(55).csv")
]

def exp_decay(t, U0, tau):
    """Экспоненциальная модель затухания"""
    return U0 * np.exp(-t / tau)

def process_and_plot(idx, P, fname):
    """Обрабатывает один файл и строит два графика, возвращает tau, Dn и R²"""
    df = pd.read_csv(fname, names=['t', 'U'], skiprows=1)
    t = df['t'].values
    U = df['U'].values
    
    # Начальное приближение
    U0_guess = U[0]
    tau_guess = (t[-1] - t[0]) / np.log(U0_guess / U[-1]) if U[-1] > 0 else (t[-1]-t[0])/3

    # Экспоненциальная подгонка
    try:
        popt, pcov = curve_fit(exp_decay, t, U, p0=[U0_guess, tau_guess], maxfev=10000)
        U0_fit, tau_fit = popt
        tau_err = np.sqrt(pcov[1,1]) if pcov.shape[0] > 1 else 0
    except Exception:
        # Если не сходится, используем логарифмическую регрессию
        logU = np.log(U)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, logU)
        tau_fit = -1.0 / slope
        tau_err = std_err / slope**2
        U0_fit = np.exp(intercept)

    # Вычисление R^2 для экспоненциальной модели
    U_pred = exp_decay(t, U0_fit, tau_fit)
    ss_res = np.sum((U - U_pred)**2)
    ss_tot = np.sum((U - np.mean(U))**2)
    r2 = 1 - ss_res/ss_tot

    # Расчёт Dn
    Dn = (V1 * V2 / (V1 + V2)) * (L / (S * tau_fit))
    
    # Погрешности Dn
    rel_err_tau = tau_err / tau_fit if tau_fit != 0 else 0
    rel_err_V = np.sqrt(2) * sigma_V / V1
    rel_err_S = 2 * sigma_d / d
    rel_err_L = sigma_L / L
    rel_err_Dn = np.sqrt(rel_err_tau**2 + rel_err_V**2 + rel_err_S**2 + rel_err_L**2)
    Dn_err = Dn * rel_err_Dn

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График U(t)
    ax1.plot(t, U, 'bo', markersize=3, label='Эксперимент')
    t_smooth = np.linspace(t[0], t[-1], 500)
    ax1.plot(t_smooth, exp_decay(t_smooth, U0_fit, tau_fit), 'r-', 
             label=f'Аппроксимация: τ = {tau_fit:.1f} с, R² = {r2:.4f}')
    ax1.set_xlabel('Время t, с')
    ax1.set_ylabel('Напряжение U, мВ')
    ax1.set_title(f'Зависимость U(t) при P = {P} торр\nФайл: {fname}')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9)
    # ax1.tight_layout()  <-- УБРАНО (ошибка была здесь)

    # Полулогарифмический график lnU vs t
    logU = np.log(U)
    ax2.plot(t, logU, 'bo', markersize=3, label='Эксперимент')
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, logU)
    ax2.plot(t, intercept + slope*t, 'r-', 
             label=f'Линейная аппроксимация: slope = {slope:.4f} c⁻¹, R² = {r_value**2:.4f}')
    ax2.set_xlabel('Время t, с')
    ax2.set_ylabel('ln(U)')
    ax2.set_title(f'Полулогарифмический график ln U(t) при P = {P} торр')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)
    # ax2.tight_layout()  <-- УБРАНО

    # Применяем tight_layout один раз ко всему Figure
    fig.tight_layout()
    
    base_name = os.path.splitext(fname)[0]
    plt.savefig(f'fit_{idx+1:02d}_{base_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Файл {fname} (№{idx+1}): P={P} торр, τ={tau_fit:.1f}±{tau_err:.1f} с, R²={r2:.4f}, Dn={Dn:.5f}±{Dn_err:.5f} см²/с")
    return P, tau_fit, tau_err, Dn, Dn_err, r2
# ---- Обработка всех файлов с построением графиков ----
results = []
for idx, (P, fname) in enumerate(data_list):
    if not os.path.exists(fname):
        print(f"Файл {fname} не найден, пропускаем.")
        continue
    res = process_and_plot(idx, P, fname)
    results.append(res)

if not results:
    print("Нет данных для обработки.")
    exit()

# Печать сводной таблицы
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*80)
print(f"{'P (torr)':>10} {'τ (s)':>12} {'Dn (cm²/s)':>15} {'1/Dn (s/cm²)':>15} {'R²':>8}")
print("-"*80)
for P, tau, tau_err, Dn, Dn_err, r2 in results:
    print(f"{P:10.1f} {tau:8.1f}±{tau_err:4.1f} {Dn:12.5f}±{Dn_err:7.5f} {1/Dn:15.2f} {r2:8.4f}")
print("="*80)

# Построение графика 1/Dn vs P
P_vals = np.array([r[0] for r in results])
invDn_vals = 1.0 / np.array([r[3] for r in results])
invDn_err = invDn_vals * (np.array([r[4] for r in results]) / np.array([r[3] for r in results]))

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = stats.linregress(P_vals, invDn_vals)
print(f"\nЛинейная регрессия: 1/Dn = {slope:.4f}·P + {intercept:.2f}")
print(f"R² = {r_value**2:.4f}, погрешность наклона: ±{std_err:.4f}")

# Построение графика с R² в легенде
plt.figure(figsize=(8, 6))
plt.errorbar(P_vals, invDn_vals, yerr=invDn_err, fmt='o', capsize=4, label='Эксперимент', markersize=6)
P_fit = np.linspace(0, max(P_vals)*1.1, 100)
plt.plot(P_fit, slope*P_fit + intercept, 'r-', linewidth=1.5,
         label=f'Линейная аппроксимация:\n1/D = {slope:.4f}·P + {intercept:.2f}\nR² = {r_value**2:.4f}')
plt.xlabel('Давление P, торр', fontsize=12)
plt.ylabel('1/Dₙ, с/см²', fontsize=12)
plt.title('Зависимость 1/Dₙ от давления', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('invDn_vs_P.png', dpi=150, bbox_inches='tight')
plt.show()

# Вычисление параметров пористой среды
D0 = 0.30  # см²/с при 760 торр (табличное значение для воздуха)
a = slope
b = intercept
delta_xi2 = 1.0 / (a * D0 * 760)
D_Kn = 1.0 / b
T = 293  # K
R = 8.314  # Дж/(моль·К)
mu_He = 0.004  # кг/моль (для воздуха ~0.029, но в работе используется He)
v = np.sqrt(8 * R * T / (np.pi * mu_He)) * 100  # см/с
d_pore = 3 * D_Kn / (delta_xi2 * v)

print(f"\n" + "="*80)
print("ПАРАМЕТРЫ ПОРИСТОЙ СРЕДЫ")
print("="*80)
print(f"δξ² = {delta_xi2:.4f} ± {delta_xi2*std_err/abs(slope):.4f}")
print(f"D_Kn = {D_Kn:.4f} см²/с")
print(f"Средняя скорость молекул: v = {v:.0f} см/с")
print(f"Диаметр пор: d_пор = {d_pore:.2e} см = {d_pore*1e4:.2f} мкм")
print("="*80)

# Оценка погрешности по χ²
chi2 = np.sum(((invDn_vals - (slope*P_vals + intercept)) / invDn_err)**2)
dof = len(results) - 2  # степени свободы
reduced_chi2 = chi2 / dof
print(f"\nКритерий согласия χ²: χ² = {chi2:.2f}, χ²/ν = {reduced_chi2:.2f} (ν = {dof})")
if 0.5 < reduced_chi2 < 2.0:
    print("→ Модель хорошо описывает данные")
elif reduced_chi2 < 0.5:
    print("→ Возможна завышенная оценка погрешностей")
else:
    print("→ Возможна неучтённая систематическая погрешность")