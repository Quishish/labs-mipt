import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import os

# ------------------------------------------------------------
# Параметры установки
# ------------------------------------------------------------
V1 = 220.0   # cm^3
V2 = 220.0   # cm^3
d = 0.95     # cm
L = 1.10     # cm
S = np.pi * (d/2)**2  # cm^2

sigma_V = 20   # cm^3
sigma_d = 0.01 # cm
sigma_L = 0.01 # cm

# ------------------------------------------------------------
# Погрешности измерений (по последнему разряду)
# ------------------------------------------------------------
delta_U = 0.001   # мВ (абсолютная погрешность напряжения)
delta_t = 0.001    # с   (абсолютная погрешность времени)

# ------------------------------------------------------------
# Данные по давлениям (торр) и имена файлов (выбраны лучшие измерения)
# ------------------------------------------------------------
data_list = [
    (11.5, "10(11.5).csv"),
    #(11.0, "10_2(11).csv"),
    #(22.0, "20(22).csv"),
    (22.0, "20_2(22).csv"),
    #(33.5, "30(33.5).csv"),
    (33.5, "30_2(33.5).csv"),
    #(44.0, "40(44).csv"),
    (44.0, "40_2(44).csv"),
    #(55.0, "50(55).csv"),
    (55.0, "50_2(55).csv")
]

def exp_decay(t, U0, tau):
    return U0 * np.exp(-t / tau)

def weighted_linregress(x, y, yerr):
    """Взвешенная линейная регрессия y = a + b*x с весами w = 1/yerr^2.
       Возвращает (b, a, r_value, chi2, chi2_red, std_err_b, std_err_a)"""
    w = 1.0 / (yerr ** 2)
    w_sum = np.sum(w)
    x_w = np.sum(w * x) / w_sum
    y_w = np.sum(w * y) / w_sum
    xx = np.sum(w * (x - x_w)**2)
    xy = np.sum(w * (x - x_w) * (y - y_w))
    b = xy / xx
    a = y_w - b * x_w
    y_pred = a + b * x
    residuals = y - y_pred
    chi2 = np.sum(w * residuals**2)
    dof = len(x) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan
    # Оценка дисперсий
    sigma_sq = chi2 / dof if dof > 0 else 1.0
    var_b = sigma_sq / xx
    var_a = sigma_sq * (1.0 / w_sum + x_w**2 / xx)
    # Коэффициент корреляции
    r_num = np.sum(w * (x - x_w) * (y - y_w))
    r_den = np.sqrt(np.sum(w * (x - x_w)**2) * np.sum(w * (y - y_w)**2))
    r_value = r_num / r_den if r_den != 0 else 0
    return b, a, r_value, chi2, chi2_red, np.sqrt(var_b), np.sqrt(var_a)

def process_and_plot(idx, P, fname):
    """Обрабатывает один файл и строит два графика, возвращает tau, Dn, R² и chi2"""
    df = pd.read_csv(fname, names=['t', 'U'], skiprows=1)
    t = df['t'].values
    U = df['U'].values
    
    # Начальное приближение
    U0_guess = U[0]
    tau_guess = (t[-1] - t[0]) / np.log(U0_guess / U[-1]) if U[-1] > 0 else (t[-1]-t[0])/3
    
    # --------------------------------------------------------
    # Экспоненциальная подгонка с весами (погрешность напряжения)
    # --------------------------------------------------------
    sigma_U = np.full_like(U, delta_U)   # постоянная погрешность напряжения
    chi2_exp = None
    chi2_red_exp = None
    try:
        popt, pcov = curve_fit(exp_decay, t, U, p0=[U0_guess, tau_guess], sigma=sigma_U, absolute_sigma=True)
        U0_fit, tau_fit = popt
        tau_err = np.sqrt(pcov[1,1]) if len(pcov) > 1 else 0
        # Вычисление взвешенного R^2 и χ²
        U_pred = exp_decay(t, U0_fit, tau_fit)
        residuals = U - U_pred
        chi2_exp = np.sum((residuals / sigma_U)**2)
        dof = len(t) - 2
        chi2_red_exp = chi2_exp / dof if dof > 0 else np.nan
        # Взвешенный R²
        ss_res = np.sum(((U - U_pred) / sigma_U)**2)
        ss_tot = np.sum(((U - np.mean(U)) / sigma_U)**2)
        r2 = 1 - ss_res / ss_tot
    except Exception as e:
        print(f"Ошибка экспоненциальной подгонки для {fname}: {e}")
        # Запасной вариант: линейная регрессия на логарифме
        logU = np.log(U)
        sigma_logU = delta_U / U   # погрешность логарифма
        b, a, r_val, chi2_log, chi2_red_log, err_b, err_a = weighted_linregress(t, logU, sigma_logU)
        tau_fit = -1.0 / b
        tau_err = err_b / b**2 if b != 0 else 0
        U0_fit = np.exp(a)
        r2 = r_val**2
        chi2_exp = chi2_log
        chi2_red_exp = chi2_red_log
    
    # --------------------------------------------------------
    # Расчёт Dn и его погрешности
    # --------------------------------------------------------
    Dn = (V1 * V2 / (V1 + V2)) * (L / (S * tau_fit))
    rel_err_tau = tau_err / tau_fit if tau_fit != 0 else 0
    rel_err_V = np.sqrt(2) * sigma_V / V1
    rel_err_S = 2 * sigma_d / d
    rel_err_L = sigma_L / L
    rel_err_Dn = np.sqrt(rel_err_tau**2 + rel_err_V**2 + rel_err_S**2 + rel_err_L**2)
    Dn_err = Dn * rel_err_Dn
    
    # --------------------------------------------------------
    # Построение графиков (увеличенный размер)
    # --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # было (12,5)
    
    # График U(t)
    ax1.errorbar(t, U, yerr=delta_U, fmt='bo', markersize=4, capsize=2, label='Эксперимент')
    t_smooth = np.linspace(t[0], t[-1], 500)
    ax1.plot(t_smooth, exp_decay(t_smooth, U0_fit, tau_fit), 'r-', linewidth=2,
             label=f'Аппроксимация: τ = {tau_fit:.1f}±{tau_err:.1f} с, R² = {r2:.4f}, χ²_red = {chi2_red_exp:.3f}')
    ax1.set_xlabel('Время t, с', fontsize=12)
    ax1.set_ylabel('Напряжение U, мВ', fontsize=12)
    ax1.set_title(f'Зависимость U(t) при P = {P} торр\nФайл: {fname}', fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='best', fontsize=10)   # увеличено для читаемости
    
    # Полулогарифмический график lnU vs t с весами
    logU = np.log(U)
    sigma_logU = delta_U / U
    slope, intercept, r_val, chi2_lin, chi2_lin_red, err_slope, err_intercept = weighted_linregress(t, logU, sigma_logU)
    ax2.errorbar(t, logU, yerr=sigma_logU, fmt='bo', markersize=4, capsize=2, label='Эксперимент')
    ax2.plot(t, intercept + slope*t, 'r-', linewidth=2,
             label=f'Линейная аппроксимация: slope = {slope:.4f}±{err_slope:.4f} c⁻¹, R² = {r_val**2:.4f}, χ²_red = {chi2_lin_red:.3f}')
    ax2.set_xlabel('Время t, с', fontsize=12)
    ax2.set_ylabel('ln(U)', fontsize=12)
    ax2.set_title(f'Полулогарифмический график ln U(t) при P = {P} торр\nФайл: {fname}', fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    base_name = os.path.splitext(fname)[0]
    plt.savefig(f'fit_{idx+1:02d}_{base_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Печать параметров
    invDn = 1.0 / Dn
    print(f"Файл {fname} (№{idx+1}): P={P} торр, τ={tau_fit:.1f}±{tau_err:.1f} с, R²={r2:.4f}, χ²_red={chi2_red_exp:.3f}, Dn={Dn:.5f}±{Dn_err:.5f} см²/с, 1/Dn={invDn:.2f} с/см²")
    
    return P, tau_fit, tau_err, Dn, Dn_err, r2, chi2_red_exp, chi2_exp, invDn, invDn * np.sqrt((Dn_err/Dn)**2 + (sigma_V/V1)**2)  # упрощённая погрешность

# ------------------------------------------------------------
# Обработка всех файлов
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Сводная таблица
# ------------------------------------------------------------
print("\n" + "="*90)
print("Сводная таблица результатов:")
print("="*90)
print(f"{'P (torr)':>10} {'tau (s)':>18} {'Dn (cm2/s)':>18} {'1/Dn (s/cm2)':>15} {'R²':>10} {'χ²_red':>10}")
print("-"*90)
for P, tau, tau_err, Dn, Dn_err, r2, chi2_red, chi2, invDn, invDn_err in results:
    print(f"{P:10.1f} {tau:8.1f}±{tau_err:.1f} {Dn:10.5f}±{Dn_err:.5f} {invDn:10.2f} {r2:10.4f} {chi2_red:10.3f}")
print("="*90)

# ------------------------------------------------------------
# График 1/Dn vs P (увеличенный)
# ------------------------------------------------------------
P_vals = np.array([r[0] for r in results])
invDn_vals = np.array([r[8] for r in results])
invDn_err = np.array([r[9] for r in results])

# Линейная регрессия для 1/Dn(P) – взвешенная
slope, intercept, r_val, chi2_lin, chi2_lin_red, err_slope, err_intercept = weighted_linregress(P_vals, invDn_vals, invDn_err)
print(f"\nЛинейная регрессия 1/Dn(P):")
print(f"  1/Dn = ({slope:.3f} ± {err_slope:.3f}) * P + ({intercept:.1f} ± {err_intercept:.1f})")
print(f"  R² = {r_val**2:.4f}")
print(f"  χ² = {chi2_lin:.2f}, χ²_red = {chi2_lin_red:.3f} (степеней свободы = {len(P_vals)-2})")

# Построение увеличенного графика
plt.figure(figsize=(10, 8))  # было (8,6)
plt.errorbar(P_vals, invDn_vals, yerr=invDn_err, fmt='o', capsize=4, markersize=8, label='Эксперимент')
P_fit = np.linspace(0, max(P_vals)*1.1, 100)
inv_fit = slope * P_fit + intercept
plt.plot(P_fit, inv_fit, 'r-', linewidth=2,
         label=f'Линейная аппроксимация: 1/D = ({slope:.3f}±{err_slope:.3f})·P + ({intercept:.1f}±{err_intercept:.1f})\nR² = {r_val**2:.4f}, χ²_red = {chi2_lin_red:.3f}')
plt.xlabel('Давление P, торр', fontsize=14)
plt.ylabel('1/D_n, с/см²', fontsize=14)
plt.title('Зависимость 1/D_n от давления', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig('invDn_vs_P.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# Параметры пористой среды
# ------------------------------------------------------------
D0 = 0.30  # см2/с при 760 торр
a = slope
b = intercept
delta_xi2 = 1.0 / (a * D0 * 760)
D_Kn = 1.0 / b

# Погрешности параметров
err_a = err_slope
err_b = err_intercept
delta_xi2_err = delta_xi2 * np.sqrt((err_a/a)**2)   # относительная погрешность
D_Kn_err = D_Kn * (err_b/b)

T = 293  # K
R_gas = 8.314  # Дж/(моль·К)
mu_He = 0.004  # кг/моль
v = np.sqrt(8 * R_gas * T / (np.pi * mu_He)) * 100  # см/с
v_err = v * 0.01  # условно 1% точность скорости
d_pore = 3 * D_Kn / (delta_xi2 * v)
# Погрешность диаметра
rel_err = np.sqrt((err_b/b)**2 + (err_a/a)**2 + (0.01)**2)
d_pore_err = d_pore * rel_err

print("\n" + "="*80)
print("Параметры пористой среды:")
print("="*80)
print(f"δξ² = {delta_xi2:.4f} ± {delta_xi2_err:.4f}")
print(f"D_Kn = {D_Kn:.4f} ± {D_Kn_err:.4f} см²/с")
print(f"Средняя скорость He: v = {v:.0f} ± {v_err:.0f} см/с")
print(f"Диаметр пор: d_пор = {d_pore:.2e} ± {d_pore_err:.2e} см = {d_pore*1e4:.1f} ± {d_pore_err*1e4:.1f} мкм")
print("="*80)
