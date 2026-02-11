import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class DoubleMassPendulumFitter:
    """
    Класс для подбора параметров маятника с двумя одинаковыми грузами:
    - Нижний груз на расстоянии R_н от оси
    - Верхний груз на расстоянии R_в от оси (на 43 мм выше оси)
    - Момент инерции стержня I_0
    """
    
    def __init__(self, g=9.81, pi=np.pi, R_upper_fixed=0.043):
        """
        Инициализация параметров
        
        Parameters:
        -----------
        g : float
            Ускорение свободного падения (м/с²)
        pi : float
            Число π
        R_upper_fixed : float
            Фиксированное расстояние верхнего груза от оси (43 мм = 0.043 м)
        """
        self.g = g
        self.pi = pi
        self.R_upper_fixed = R_upper_fixed  # 43 мм выше оси
        self.R_lower_vals = None  # расстояния нижнего груза
        self.T_vals = None
        self.T_err_vals = None
        
    def set_experimental_data(self, R_lower_vals, T_vals, T_err_vals=None):
        """
        Задание экспериментальных данных
        
        Parameters:
        -----------
        R_lower_vals : array-like
            Расстояния от оси до нижнего груза (м)
        T_vals : array-like
            Измеренные периоды колебаний (с)
        T_err_vals : array-like, optional
            Ошибки измерения периодов
        """
        self.R_lower_vals = np.array(R_lower_vals, dtype=float)
        self.T_vals = np.array(T_vals, dtype=float)
        
        if T_err_vals is not None:
            self.T_err_vals = np.array(T_err_vals, dtype=float)
            if len(self.T_err_vals) != len(self.T_vals):
                print(f"Внимание: размер T_err_vals ({len(self.T_err_vals)}) не совпадает с T_vals ({len(self.T_vals)})")
                self.T_err_vals = np.full_like(self.T_vals, np.mean(self.T_err_vals))
        else:
            self.T_err_vals = 0.01 * self.T_vals
    
    def model_double_mass(self, R_lower, I_0, m):
        """
        Модель маятника с двумя одинаковыми грузами:
        
        Общий момент инерции:
        I_total = I_0 + m·R_н² + m·R_в²
        где R_в = 0.043 м (фиксировано)
        
        Расстояние до центра масс:
        R_cm = (m·R_н - m·R_в) / (2m) = (R_н - R_в) / 2
        
        Период:
        T² = 4π²·I_total / (2m·g·R_cm)
           = 4π²·(I_0 + m·R_н² + m·R_в²) / (2m·g·(R_н - R_в)/2)
           = 4π²·(I_0 + m·(R_н² + R_в²)) / (m·g·(R_н - R_в))
        
        Parameters:
        -----------
        R_lower : float or array
            Расстояние от оси до нижнего груза
        I_0 : float
            Момент инерции стержня (без грузов) относительно оси
        m : float
            Масса одного груза (оба груза одинаковые)
        
        Returns:
        --------
        T² : float or array
            Квадрат периода колебаний
        """
        R_upper = self.R_upper_fixed  # верхний груз на фиксированном расстоянии
        
        # Избегаем деления на ноль и отрицательных значений
        denominator = m * self.g * (R_lower - R_upper)
        denominator = np.where(denominator > 0, denominator, 1e-10)
        
        # Общий момент инерции
        I_total = I_0 + m * (R_lower**2 + R_upper**2)
        
        numerator = 4 * self.pi**2 * I_total
        return numerator / denominator
    
    def fit_parameters(self, I0_guess=0.01, m_guess=0.1, bounds=None):
        """
        Подбор параметров I_0 и m
        
        Parameters:
        -----------
        I0_guess : float
            Начальное приближение для момента инерции стержня (кг·м²)
        m_guess : float
            Начальное приближение для массы одного груза (кг)
        bounds : tuple of arrays, optional
            Границы для параметров в формате (lower_bounds, upper_bounds)
        
        Returns:
        --------
        popt : array
            Оптимальные значения параметров
        pcov : 2D array
            Ковариационная матрица оценок
        """
        if self.R_lower_vals is None or self.T_vals is None:
            raise ValueError("Сначала задайте экспериментальные данные")
        
        # Проверяем совпадение размеров
        if len(self.R_lower_vals) != len(self.T_vals):
            min_len = min(len(self.R_lower_vals), len(self.T_vals))
            self.R_lower_vals = self.R_lower_vals[:min_len]
            self.T_vals = self.T_vals[:min_len]
            self.T_err_vals = self.T_err_vals[:min_len]
            print(f"Использую первые {min_len} значений")
        
        # Преобразуем данные: T²
        T2_vals = self.T_vals**2
        
        # Вычисляем ошибки для T²
        if self.T_err_vals is not None and len(self.T_err_vals) == len(self.T_vals):
            T2_err_vals = 2 * self.T_vals * self.T_err_vals
        else:
            T2_err_vals = 0.01 * T2_vals
        
        print(f"Подгонка параметров для маятника с двумя грузами")
        print(f"Верхний груз зафиксирован на R_в = {self.R_upper_fixed*1000:.1f} мм")
        print(f"Нижний груз перемещается: {len(self.R_lower_vals)} положений")
        
        # Определение границ параметров
        if bounds is None:
            bounds_lower = [0, 0]  # I_0 ≥ 0, m ≥ 0
            bounds_upper = [np.inf, np.inf]
            bounds = (bounds_lower, bounds_upper)
        
        # Начальные приближения
        p0 = [I0_guess, m_guess]
        
        # Подгонка кривой
        try:
            popt, pcov = curve_fit(
                self.model_double_mass, 
                self.R_lower_vals, 
                T2_vals,
                p0=p0,
                sigma=T2_err_vals,
                bounds=bounds,
                maxfev=5000
            )
            
            I0_fit, m_fit = popt
            I0_err, m_err = np.sqrt(np.diag(pcov))
            
            print(f"\nРезультаты подгонки:")
            print(f"  Момент инерции стержня I_0 = {I0_fit:.6f} ± {I0_err:.6f} кг·м²")
            print(f"  Масса одного груза m = {m_fit:.6f} ± {m_err:.6f} кг")
            print(f"                     = {m_fit*1000:.1f} ± {m_err*1000:.1f} г")
            print(f"  Общая масса двух грузов: {2*m_fit*1000:.1f} ± {2*m_err*1000:.1f} г")
            
        except Exception as e:
            print(f"Ошибка при подгонке: {e}")
            print("Пробую подгонку без учета ошибок...")
            popt, pcov = curve_fit(
                self.model_double_mass, 
                self.R_lower_vals, 
                T2_vals,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            I0_fit, m_fit = popt
            print(f"\nРезультаты подгонки (без ошибок):")
            print(f"  Момент инерции стержня I_0 = {I0_fit:.6f} кг·м²")
            print(f"  Масса одного груза m = {m_fit:.6f} кг = {m_fit*1000:.1f} г")
            pcov = np.eye(2) * 1e-6
        
        return popt, pcov
    
    def plot_fit(self, popt, pcov=None, show_confidence=True):
        """
        Построение графика с экспериментальными данными и подобранной моделью
        """
        if self.R_lower_vals is None or self.T_vals is None:
            raise ValueError("Сначала задайте экспериментальные данные")
        
        # Создаем график
        fig, (ax1) = plt.subplots(1, figsize=(14, 6))
        
        # График 1: T от R_н
        ax1.errorbar(self.R_lower_vals * 1000, self.T_vals, 
                    yerr=self.T_err_vals, fmt='o', capsize=5,
                    label='Экспериментальные данные', color='blue', markersize=8)
        
        # Генерация гладкой кривой для модели
        R_min = max(self.R_upper_fixed * 1.1, min(self.R_lower_vals) * 0.9)
        R_max = max(self.R_lower_vals) * 1.1
        R_fine = np.linspace(R_min, R_max, 200)
        
        I0_fit, m_fit = popt
        T_fine = np.sqrt(self.model_double_mass(R_fine, I0_fit, m_fit))
        label = f'Модель: I_0={I0_fit:.4f}, m={m_fit:.4f}'
        
        # Доверительный интервал
        if pcov is not None and show_confidence:
            n_samples = 100
            T_samples = []
            
            for _ in range(n_samples):
                try:
                    params_sample = np.random.multivariate_normal(popt, pcov)
                    I0_sample, m_sample = params_sample
                    
                    if I0_sample > 0 and m_sample > 0:
                        T_sample = np.sqrt(self.model_double_mass(R_fine, I0_sample, m_sample))
                        T_samples.append(T_sample)
                except:
                    continue
            
            if T_samples:
                T_samples = np.array(T_samples)
                T_mean = np.mean(T_samples, axis=0)
                T_std = np.std(T_samples, axis=0)
                
                ax1.fill_between(R_fine * 1000, T_mean - 2*T_std, T_mean + 2*T_std,
                                alpha=0.2, color='red')
        
        ax1.plot(R_fine * 1000, T_fine, 'r-', linewidth=2, label=label)
        ax1.set_xlabel('Расстояние до нижнего груза R_н (мм)', fontsize=20)
        ax1.set_ylabel('Период T (с)', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='minor', labelsize=12)
        ax1.set_title('Маятник с двумя грузами', fontsize=30)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize = 14)
        
        # Вертикальная линия для положения верхнего груза
        ax1.axvline(x=self.R_upper_fixed * 1000, color='green', linestyle='--', 
                   alpha=0.5, label=f'Верхний груз (R_в={self.R_upper_fixed*1000:.0f} мм)')
        ax1.legend(loc='best', fontsize = 14)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_residuals(self, popt):
        """Вычисление невязок модели"""
        I0_fit, m_fit = popt
        T2_model = self.model_double_mass(self.R_lower_vals, I0_fit, m_fit)
        T2_experimental = self.T_vals**2
        residuals = T2_experimental - T2_model
        relative_residuals = residuals / T2_experimental * 100
        
        return residuals, relative_residuals
    
    def analyze_physical_consistency(self, popt):
        """Анализ физической осмысленности подобранных параметров"""
        I0_fit, m_fit = popt
        
        print("\n" + "="*60)
        print("АНАЛИЗ ФИЗИЧЕСКОЙ ОСМЫСЛЕННОСТИ ПАРАМЕТРОВ")
        print("="*60)
        
        # Проверка 1: Положительность параметров
        print("1. Проверка положительности:")
        print(f"   I_0 = {I0_fit:.6f} кг·м² > 0: {'✓' if I0_fit > 0 else '✗'}")
        print(f"   m = {m_fit:.6f} кг > 0: {'✓' if m_fit > 0 else '✗'}")
        
        # Проверка 2: Масса в разумных пределах
        print(f"\n2. Проверка массы груза:")
        print(f"   Масса одного груза: {m_fit:.4f} кг = {m_fit*1000:.1f} г")
        print(f"   Общая масса двух грузов: {2*m_fit*1000:.1f} г")
        
        if 0.01 <= m_fit <= 0.5:  # от 10 г до 500 г
            print("   ✓ Масса в разумных пределах для лабораторных грузов")
        elif m_fit < 0.01:
            print("   ⚠ Масса слишком мала (<10 г)")
        else:
            print("   ⚠ Масса великовата для лабораторных грузов (>500 г)")
        
        # Проверка 3: Момент инерции стержня
        print(f"\n3. Анализ момента инерции стержня:")
        
        # Оценка момента инерции для равномерного стержня
        # Допустим, стержень длиной L = 0.3 м, массой m_rod = 0.1 кг
        L_est = 0.3  # м
        m_rod_est = 0.1  # кг
        I_rod_theory = (1/12) * m_rod_est * L_est**2  # относительно центра
        
        print(f"   I_0 (подобранный) = {I0_fit:.6f} кг·м²")
        print(f"   I_rod (теория для стержня {L_est*1000:.0f}мм×{m_rod_est*1000:.0f}г) = {I_rod_theory:.6f} кг·м²")
        print(f"   Отношение I_0/I_rod = {I0_fit/I_rod_theory:.2f}")
        
        # Проверка 4: Общий момент инерции для среднего положения
        R_avg = np.mean(self.R_lower_vals)
        R_upper = self.R_upper_fixed
        I_total = I0_fit + m_fit * (R_avg**2 + R_upper**2)
        
        print(f"\n4. Общий момент инерции (среднее положение):")
        print(f"   I_total = I_0 + m·(R_н² + R_в²)")
        print(f"           = {I0_fit:.6f} + {m_fit:.4f}·({R_avg**2:.4f} + {R_upper**2:.4f})")
        print(f"           = {I_total:.6f} кг·м²")
        
        # Проверка 5: Центр масс системы
        R_cm = (m_fit * R_avg - m_fit * R_upper) / (2 * m_fit)
        print(f"\n5. Положение центра масс (относительно оси):")
        print(f"   R_cm = (m·R_н - m·R_в) / 2m = (R_н - R_в)/2")
        print(f"        = ({R_avg:.3f} - {R_upper:.3f}) / 2 = {R_cm:.3f} м")
        print(f"        = {R_cm*1000:.1f} мм {'ниже' if R_cm > 0 else 'выше'} оси")
        
        return {
            'I0': I0_fit, 
            'm': m_fit,
            'I_total': I_total,
            'R_cm': R_cm
        }


# ==============================================
# АЛЬТЕРНАТИВНАЯ МОДЕЛЬ: ТОЛЬКО НИЖНИЙ ГРУЗ
# ==============================================

class SingleMassPendulumFitter:
    """
    Класс для сравнения: модель только с нижним грузом
    (верхний груз отсутствует или его масса пренебрежимо мала)
    """
    
    def __init__(self, g=9.81, pi=np.pi):
        self.g = g
        self.pi = pi
        self.R_vals = None
        self.T_vals = None
        self.T_err_vals = None
        
    def set_experimental_data(self, R_vals, T_vals, T_err_vals=None):
        self.R_vals = np.array(R_vals, dtype=float)
        self.T_vals = np.array(T_vals, dtype=float)
        
        if T_err_vals is not None:
            self.T_err_vals = np.array(T_err_vals, dtype=float)
            if len(self.T_err_vals) != len(self.T_vals):
                self.T_err_vals = np.full_like(self.T_vals, np.mean(self.T_err_vals))
        else:
            self.T_err_vals = 0.01 * self.T_vals
    
    def model_single_mass(self, R, I0, m):
        """
        Модель с одним грузом (нижним):
        T² = 4π²·(I_0 + m·R²) / (m·g·R)
        """
        denominator = m * self.g * R
        denominator = np.where(denominator > 0, denominator, 1e-10)
        
        numerator = 4 * self.pi**2 * (I0 + m * R**2)
        return numerator / denominator
    
    def fit_parameters(self, I0_guess=0.01, m_guess=0.1):
        T2_vals = self.T_vals**2
        
        if self.T_err_vals is not None and len(self.T_err_vals) == len(self.T_vals):
            T2_err_vals = 2 * self.T_vals * self.T_err_vals
        else:
            T2_err_vals = 0.01 * T2_vals
        
        p0 = [I0_guess, m_guess]
        bounds = ([0, 0], [np.inf, np.inf])
        
        try:
            popt, pcov = curve_fit(
                self.model_single_mass, 
                self.R_vals, 
                T2_vals,
                p0=p0,
                sigma=T2_err_vals,
                bounds=bounds,
                maxfev=5000
            )
        except:
            popt, pcov = curve_fit(
                self.model_single_mass, 
                self.R_vals, 
                T2_vals,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            pcov = np.eye(2) * 1e-6
        
        return popt, pcov


# ==============================================
# ОСНОВНОЙ АНАЛИЗ
# ==============================================

def analyze_double_mass_pendulum():
    """
    Основная функция анализа маятника с двумя грузами
    """
    print("АНАЛИЗ МАЯТНИКА С ДВУМЯ ГРУЗАМИ")
    print("="*60)
    
    # Ваши экспериментальные данные (7 измерений)
    R_lower_vals = np.array([0.1265, 0.1585, 0.1896, 0.2210, 0.2530])
    T_vals = np.array([1.484268, 1.374836, 1.336012, 1.321739, 1.330762])
    
    # Ошибки периодов (можно оценить или взять из файлов)
    T_err_vals = np.array([0.00087, 0.00087, 0.00087, 0.00087, 0.0009, 0.0010, 0.0012])
    
    print(f"Экспериментальные данные ({len(R_lower_vals)} измерений):")
    print("R_н (мм)   T (с)          T_err (с)")
    for R, T, T_err in zip(R_lower_vals, T_vals, T_err_vals):
        print(f"{R*1000:6.0f}     {T:.6f}      {T_err:.6f}")
    
    print(f"\nВерхний груз зафиксирован на расстоянии: 43 мм = {0.043*1000:.1f} мм")
    
    # 1. Анализ с двумя грузами
    print("\n" + "="*60)
    print("1. МОДЕЛЬ С ДВУМЯ ГРУЗАМИ (верхний + нижний)")
    print("="*60)
    
    fitter_double = DoubleMassPendulumFitter(g=9.81, R_upper_fixed=0.043)
    fitter_double.set_experimental_data(R_lower_vals, T_vals, T_err_vals)
    
    # Подбор параметров с разными начальными приближениями
    results_double = []
    
    for attempt in range(3):
        if attempt == 0:
            I0_guess, m_guess = 0.005, 0.05
        elif attempt == 1:
            I0_guess, m_guess = 0.01, 0.1
        else:
            I0_guess, m_guess = 0.02, 0.15
        
        print(f"\nПопытка {attempt+1}: I0_guess={I0_guess:.3f}, m_guess={m_guess:.3f}")
        
        try:
            popt, pcov = fitter_double.fit_parameters(I0_guess, m_guess)
            residuals, rel_residuals = fitter_double.calculate_residuals(popt)
            avg_rel_error = np.mean(np.abs(rel_residuals))
            
            results_double.append({
                'popt': popt,
                'pcov': pcov,
                'avg_rel_error': avg_rel_error,
                'I0_guess': I0_guess,
                'm_guess': m_guess
            })
            
            print(f"  Средняя ошибка: {avg_rel_error:.2f}%")
            
        except Exception as e:
            print(f"  Ошибка: {e}")
    
    # Выбираем лучший результат
    if results_double:
        best_idx = min(range(len(results_double)), key=lambda i: results_double[i]['avg_rel_error'])
        #best_idx=2
        best_double = results_double[best_idx]
        
        print(f"\nЛучший результат (попытка {best_idx+1}):")
        print(f"Средняя ошибка: {best_double['avg_rel_error']:.2f}%")
        
        popt_double = best_double['popt']
        pcov_double = best_double['pcov']
        
        # Анализ физической осмысленности
        analysis_double = fitter_double.analyze_physical_consistency(popt_double)
        
        # Построение графика
        fig_double = fitter_double.plot_fit(popt_double, pcov_double)
        
        # Коэффициент детерминации R²
        I0_fit, m_fit = popt_double
        T2_model = fitter_double.model_double_mass(R_lower_vals, I0_fit, m_fit)
        T2_experimental = T_vals**2
        
        ss_res = np.sum((T2_experimental - T2_model)**2)
        ss_tot = np.sum((T2_experimental - np.mean(T2_experimental))**2)
        r_squared_double = 1 - (ss_res / ss_tot)
        
        print(f"\nКачество подгонки (модель с двумя грузами):")
        print(f"  R² = {r_squared_double:.6f}")
    
    # 2. Анализ с одним грузом (для сравнения)
    print("\n" + "="*60)
    print("2. МОДЕЛЬ ТОЛЬКО С НИЖНИМ ГРУЗОМ (для сравнения)")
    print("="*60)
    
    fitter_single = SingleMassPendulumFitter(g=9.81)
    fitter_single.set_experimental_data(R_lower_vals, T_vals, T_err_vals)
    
    popt_single, pcov_single = fitter_single.fit_parameters(0.01, 0.1)
    I0_single, m_single = popt_single
    
    print(f"Результаты модели с одним грузом:")
    print(f"  I_0 = {I0_single:.6f} кг·м²")
    print(f"  m = {m_single:.6f} кг = {m_single*1000:.1f} г")
    
    # Сравнение двух моделей
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ")
    print("="*60)
    
    if 'analysis_double' in locals() and 'm_fit' in analysis_double:
        print(f"Модель с двумя грузами:")
        print(f"  Масса одного груза: {analysis_double['m']*1000:.1f} г")
        print(f"  Общая масса: {2*analysis_double['m']*1000:.1f} г")
        print(f"  Момент инерции стержня: {analysis_double['I0']:.6f} кг·м²")
        
        print(f"\nМодель с одним грузом:")
        print(f"  Масса груза: {m_single*1000:.1f} г")
        print(f"  Момент инерции стержня: {I0_single:.6f} кг·м²")
        
        print(f"\nРазница в массе: {abs(analysis_double['m'] - m_single)/analysis_double['m']*100:.1f}%")
        print(f"Разница в I_0: {abs(analysis_double['I0'] - I0_single)/analysis_double['I0']*100:.1f}%")
        
        # Какой модель лучше?
        residuals_double, rel_double = fitter_double.calculate_residuals(popt_double)
        avg_rel_double = np.mean(np.abs(rel_double))
        
        # Для модели с одним грузом
        T2_model_single = fitter_single.model_single_mass(R_lower_vals, I0_single, m_single)
        rel_single = (T2_experimental - T2_model_single) / T2_experimental * 100
        avg_rel_single = np.mean(np.abs(rel_single))
        
        print(f"\nСредняя относительная ошибка:")
        print(f"  Модель с двумя грузами: {avg_rel_double:.2f}%")
        print(f"  Модель с одним грузом: {avg_rel_single:.2f}%")
        
        if avg_rel_double < avg_rel_single:
            print(f"\n✓ Модель с двумя грузами лучше на {(avg_rel_single - avg_rel_double)/avg_rel_single*100:.1f}%")
        else:
            print(f"\n✓ Модель с одним грузом лучше на {(avg_rel_double - avg_rel_single)/avg_rel_double*100:.1f}%")
    
    # 3. Визуализация обеих моделей
    print("\n" + "="*60)
    print("3. ВИЗУАЛИЗАЦИЯ ОБЕИХ МОДЕЛЕЙ")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Экспериментальные данные
    ax.errorbar(R_lower_vals * 1000, T_vals, yerr=T_err_vals, 
                fmt='o', capsize=5, label='Экспериментальные данные', 
                color='black', markersize=8)
    
    # Модель с двумя грузами
    if 'popt_double' in locals():
        R_fine = np.linspace(min(R_lower_vals), max(R_lower_vals), 200)
        T_double = np.sqrt(fitter_double.model_double_mass(R_fine, *popt_double))
        ax.plot(R_fine * 1000, T_double, 'r-', linewidth=2, 
                label=f'2 груза: m={analysis_double["m"]*1000:.1f}г')
    
    # Модель с одним грузом
    T_single = np.sqrt(fitter_single.model_single_mass(R_fine, *popt_single))
    ax.plot(R_fine * 1000, T_single, 'b--', linewidth=2, 
            label=f'1 груз: m={m_single*1000:.1f}г')
    
    ax.set_xlabel('Расстояние до нижнего груза R_н (мм)', fontsize=20)
    ax.set_ylabel('Период T (с)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_title('Сравнение моделей маятника', fontsize=30)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize = 20)
    
    # Вертикальная линия для верхнего груза
    ax.axvline(x=43, color='green', linestyle=':', alpha=0.7, 
               label='Верхний груз (43 мм)')
    ax.legend(loc='best', fontsize = 14)
    
    plt.tight_layout()
    plt.show()
    
    # Сохранение результатов
    save_double_mass_results(R_lower_vals, T_vals, T_err_vals, 
                            analysis_double if 'analysis_double' in locals() else None,
                            popt_single if 'popt_single' in locals() else None)
    
    return {
        'double_mass': analysis_double if 'analysis_double' in locals() else None,
        'single_mass': {'I0': I0_single, 'm': m_single} if 'popt_single' in locals() else None
    }


def save_double_mass_results(R_vals, T_vals, T_err_vals, analysis_double, analysis_single):
    """Сохранение результатов в файл"""
    
    report = f"""
РЕЗУЛЬТАТЫ АНАЛИЗА МАЯТНИКА С ДВУМЯ ГРУЗАМИ

ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ:
-------------------------
Количество измерений: {len(R_vals)}
Верхний груз зафиксирован на R_в = 43 мм

R_н (мм)   T (с)          T_err (с)
"""
    
    for R, T, T_err in zip(R_vals, T_vals, T_err_vals):
        report += f"{R*1000:6.0f}     {T:.6f}      {T_err:.6f}\n"
    
    if analysis_double:
        report += f"""

МОДЕЛЬ С ДВУМЯ ГРУЗАМИ:
-----------------------
Момент инерции стержня I_0 = {analysis_double['I0']:.8f} кг·м²
Масса одного груза m = {analysis_double['m']:.6f} кг = {analysis_double['m']*1000:.2f} г
Общая масса двух грузов = {2*analysis_double['m']*1000:.2f} г

Общий момент инерции (среднее положение):
I_total = I_0 + m·(R_н² + R_в²) = {analysis_double['I_total']:.8f} кг·м²

Положение центра масс:
R_cm = {analysis_double['R_cm']:.4f} м = {analysis_double['R_cm']*1000:.1f} мм
"""
    
    if analysis_single:
        report += f"""

МОДЕЛЬ С ОДНИМ ГРУЗОМ (для сравнения):
--------------------------------------
Момент инерции стержня I_0 = {analysis_single['I0']:.8f} кг·м²
Масса груза m = {analysis_single['m']:.6f} кг = {analysis_single['m']*1000:.2f} г
"""
    
    report += f"""

ВЫВОДЫ:
-------
1. Модель с двумя грузами более реалистична, если верхний груз действительно присутствует
2. Если разница между моделями небольшая (<5%), можно использовать упрощенную модель
3. Для точных измерений необходимо учитывать все массы системы

Рекомендации:
- Проверить точное расстояние верхнего груза от оси
- Учесть возможную неидентичность грузов
- Учесть момент инерции креплений и других деталей

Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('double_mass_pendulum_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nРезультаты сохранены в файл: double_mass_pendulum_analysis.txt")
    
    return report


# ==============================================
# ЗАПУСК ПРОГРАММЫ
# ==============================================

if __name__ == "__main__":
    # Для работы с датами
    import pandas as pd
    
    print("АНАЛИЗ МАЯТНИКА С ДВУМЯ ГРУЗАМИ")
    print("="*70)
    print("Модель: T² = 4π²[I_0 + m·(R_н² + R_в²)] / [m·g·(R_н - R_в)]")
    print("где R_в = 0.043 м (43 мм) - фиксированное расстояние верхнего груза")
    print("="*70)
    
    # Запуск анализа
    try:
        results = analyze_double_mass_pendulum()
        print("\n" + "="*70)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*70)
    except Exception as e:
        print(f"\nОшибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()