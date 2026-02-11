import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

# Настройка шрифтов для кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def calculate_beta(filename, delta_phi_n):
    """
    Рассчитывает коэффициент вязкого трения β по методу 1 из теории.
    
    Parameters:
    -----------
    filename : str
        Путь к файлу с данными измерений
    delta_phi_n : float
        Эффективный угловой размер спицы (в радианах) из предварительных измерений
    
    Returns:
    --------
    beta_values : numpy.array
        Массив значений β для каждого полупериода
    beta_mean : float
        Среднее значение β
    amplitudes : numpy.array
        Массив амплитуд (в радианах) для каждого колебания
    """
    
    # Константа преобразования тактов таймера в секунды
    TIMER_FREQ = 1193180.0
    
    # Чтение данных из файла
    dt_list = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Парсим файл
    for line in lines:
        line = line.strip()
        
        # Пропускаем заголовок и пустые строки
        if not line or 'Данные' in line or 'Всего' in line \
           or 'T1' in line or '#N' in line or '|' in line:
            continue
        
        # Парсим строку формата: " 0 , 1209927,  1213049,  3122"
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 3:
            try:
                dt = int(parts[3])
                dt_list.append(dt)
            except (ValueError, IndexError):
                continue
    
    # Конвертируем в numpy array и переводим в секунды
    dt_ticks = np.array(dt_list)
    dt_seconds = dt_ticks / TIMER_FREQ
    
    print(f"Прочитано точек: {len(dt_seconds)}")
    
    # Вычисляем безразмерные скорости
    phi_prime = delta_phi_n / dt_seconds
    
    # Массивы для результатов
    beta_values = []
    amplitudes = []
    
    # Вычисляем β для каждой пары соседних качаний
    for i in range(len(phi_prime) - 1):
        phi_i = phi_prime[i]
        phi_i1 = phi_prime[i + 1]
        
        # Формула (11в): sin²(φ_x/2) = (φ'_i - φ'_{i+1})²/16
        sin2_half_phi = ((phi_i - phi_i1) ** 2) / 16.0
        
        # Проверка на физичность
        if sin2_half_phi > 1.0:
            sin2_half_phi = 1.0
        
        # Амплитуда
        phi_x = 2.0 * np.arcsin(np.sqrt(sin2_half_phi))
        amplitudes.append(phi_x)
        
        # Вычисление β через формулу (23) с эллиптическими интегралами
        k_squared = sin2_half_phi
        
        K = ellipk(k_squared)
        E = ellipe(k_squared)
        
        numerator = phi_i**2 - phi_i1**2
        denominator = 16.0 * (E + (k_squared - 1.0) * K)
        
        if abs(denominator) > 1e-10:
            beta = numerator / denominator
            beta_values.append(beta)
        else:
            beta_values.append(np.nan)
    
    beta_values = np.array(beta_values)
    amplitudes = np.array(amplitudes)
    
    # Среднее по валидным значениям
    beta_valid = beta_values[~np.isnan(beta_values)]
    beta_mean = np.mean(beta_valid) if len(beta_valid) > 0 else np.nan
    
    return beta_values, beta_mean, amplitudes


def plot_results(filename, delta_phi_n, T0):
    """
    Рассчитывает β и строит графики
    """
    beta_values, beta_mean, amplitudes = calculate_beta(filename, delta_phi_n)
    
    # Информация в консоль
    print("\n" + "="*60)
    print(f"Файл: {filename}")
    print(f"Δφ_n = {delta_phi_n:.6e} рад")
    print(f"T₀ = {T0:.4f} с")
    print("="*60)
    print(f"\nРассчитано значений β: {len(beta_values)}")
    print(f"Среднее β = {beta_mean:.6e}")
    print(f"Станд. отклонение = {np.nanstd(beta_values):.6e}")
    
    if beta_mean < 0.1:
        print(f"✓ Условие β < 0.1 выполнено")
    else:
        print(f"✗ ВНИМАНИЕ: β = {beta_mean:.3f} ≥ 0.1!")
    
    amp_deg = np.degrees(amplitudes)
    print(f"\nАмплитуды: {amp_deg.min():.2f}° - {amp_deg.max():.2f}°")
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Анализ параметра вязкого трения β\n(средн. β = {beta_mean:.6e})', 
                 fontsize=14, fontweight='bold')
    
    # График 1: β от номера измерения
    ax1 = axes[0, 0]
    ax1.plot(beta_values, 'o-', markersize=3, linewidth=0.5, alpha=0.7)
    ax1.axhline(beta_mean, color='red', linestyle='--', linewidth=1.5, 
                label=f'Среднее = {beta_mean:.6e}')
    ax1.axhline(0.1, color='orange', linestyle=':', linewidth=1, 
                label='Предел β = 0.1')
    ax1.set_xlabel('Номер измерения', fontsize=11)
    ax1.set_ylabel('β (безразм.)', fontsize=11)
    ax1.set_title('Параметр затухания vs номер измерения', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График 2: β от амплитуды
    ax2 = axes[0, 1]
    ax2.scatter(amp_deg, beta_values, s=10, alpha=0.6)
    ax2.axhline(beta_mean, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Амплитуда φ (градусы)', fontsize=11)
    ax2.set_ylabel('β (безразм.)', fontsize=11)
    ax2.set_title('Зависимость β от амплитуды', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Амплитуда от номера (затухание)
    ax3 = axes[1, 0]
    ax3.plot(amp_deg, 'o-', markersize=3, linewidth=0.5, color='green', alpha=0.7)
    ax3.set_xlabel('Номер измерения', fontsize=11)
    ax3.set_ylabel('Амплитуда φ (градусы)', fontsize=11)
    ax3.set_title('Затухание амплитуды колебаний', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Гистограмма распределения β
    ax4 = axes[1, 1]
    beta_clean = beta_values[~np.isnan(beta_values)]
    ax4.hist(beta_clean, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(beta_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Среднее = {beta_mean:.6e}')
    ax4.set_xlabel('β (безразм.)', fontsize=11)
    ax4.set_ylabel('Количество', fontsize=11)
    ax4.set_title('Распределение значений β', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return beta_values, beta_mean, amplitudes


# ИСПОЛЬЗОВАНИЕ
if __name__ == "__main__":
    
    # ==== УКАЖИ СВОИ ПАРАМЕТРЫ ====
    
    # Из предварительных измерений (пункт 3 программы)
    DELTA_PHI_N = 0.005215511288  # эффективный угловой размер спицы, рад
    T0 = 1.330762391729              # период линейных колебаний, с
    
    # Имя файла
    FILENAME = "ELP_005.txt"
    
    # ===============================
    
    # Расчёт и построение графиков
    beta_vals, beta_avg, amps = plot_results(FILENAME, DELTA_PHI_N, T0)
