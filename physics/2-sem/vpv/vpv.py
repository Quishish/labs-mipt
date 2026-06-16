import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Параметры установки
V1 = 220.0   # cm^3
V2 = 220.0   # cm^3
d = 0.95     # cm
L = 1.10     # cm
S = np.pi * (d/2)**2  # cm^2

sigma_V = 20   # cm^3
sigma_d = 0.01 # cm
sigma_L = 0.01 # cm

# Данные по давлениям (торр) и имена файлов
data_list = [
    (11.5, "10(11.5).csv"),
    (11.0, "10_2(11).csv"),
    (22.0, "20(22).csv"),
    (22.0, "20_2(22).csv"),
    (33.5, "30(33.5).csv"),
    (33.5, "30_2(33.5).csv"),
    (44.0, "40(44).csv"),
    (44.0, "40_2(44).csv"),
    (55.0, "50(55).csv"),
    (55.0, "50_2(55).csv")
]

def exp_decay(t, U0, tau):
    return U0 * np.exp(-t / tau)

results = []
for P, fname in data_list:
    df = pd.read_csv(fname, names=['t', 'U'], skiprows=1)
    t = df['t'].values
    U = df['U'].values
    
    # Начальное приближение
    U0_guess = U[0]
    tau_guess = (t[-1] - t[0]) / np.log(U0_guess / U[-1])
    try:
        popt, pcov = curve_fit(exp_decay, t, U, p0=[U0_guess, tau_guess])
        U0_fit, tau_fit = popt
        tau_err = np.sqrt(pcov[1,1])
    except:
        # Если не сходится, используем логарифмическую линейную регрессию
        logU = np.log(U)
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, logU)
        tau_fit = -1.0 / slope
        tau_err = std_err / slope**2
        U0_fit = np.exp(intercept)
    
    # Вычисляем D_n
    Dn = (V1 * V2 / (V1 + V2)) * (L / (S * tau_fit))
    # Погрешность Dn
    rel_err_tau = tau_err / tau_fit
    rel_err_V = np.sqrt(2) * sigma_V / V1   # V1=V2
    rel_err_S = 2 * sigma_d / d
    rel_err_L = sigma_L / L
    rel_err_Dn = np.sqrt(rel_err_tau**2 + rel_err_V**2 + rel_err_S**2 + rel_err_L**2)
    Dn_err = Dn * rel_err_Dn
    
    results.append((P, tau_fit, tau_err, Dn, Dn_err))

# Печать таблицы
print("P (torr)\ttau (s)\t\tDn (cm2/s)\t1/Dn (s/cm2)")
for P, tau, tau_err, Dn, Dn_err in results:
    print(f"{P:5.1f}\t{tau:6.1f} ± {tau_err:.1f}\t{Dn:.5f} ± {Dn_err:.5f}\t{1/Dn:.2f}")

# Построение графика 1/Dn vs P
P_vals = np.array([r[0] for r in results])
invDn_vals = 1.0 / np.array([r[3] for r in results])
invDn_err = invDn_vals * (np.array([r[4] for r in results]) / np.array([r[3] for r in results]))

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = stats.linregress(P_vals, invDn_vals)
print(f"\nЛинейная регрессия: 1/Dn = {slope:.3f} * P + {intercept:.1f}")
print(f"R^2 = {r_value**2:.3f}")

# Построение графика
plt.figure(figsize=(8,6))
plt.errorbar(P_vals, invDn_vals, yerr=invDn_err, fmt='o', capsize=3, label='Эксперимент')
P_fit = np.linspace(0, 60, 100)
plt.plot(P_fit, slope*P_fit + intercept, 'r-', label='Линейная аппроксимация')
plt.xlabel('Давление P, торр')
plt.ylabel('1/D_n, с/см²')
plt.title('Зависимость 1/D_n от давления')
plt.grid(True)
plt.legend()
plt.savefig('invDn_vs_P.png', dpi=150)
plt.show()

# Вычисление параметров пористой среды
D0 = 0.30  # см2/с при 760 торр
a = slope
b = intercept
delta_xi2 = 1.0 / (a * D0 * 760)
D_Kn = 1.0 / b
T = 293  # K
R = 8.314  # Дж/(моль·К)
mu_He = 0.004  # кг/моль
v = np.sqrt(8 * R * T / (np.pi * mu_He)) * 100  # см/с
d_pore = 3 * D_Kn / (delta_xi2 * v)

print(f"\nδξ² = {delta_xi2:.4f}")
print(f"D_Kn = {D_Kn:.4f} см²/с")
print(f"Средняя скорость He: v = {v:.0f} см/с")
print(f"Диаметр пор: d_пор = {d_pore:.2e} см = {d_pore*1e4:.2f} мкм")