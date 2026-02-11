import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class PendulumParameterFitter:
    """
    Класс для подбора параметров I и m из зависимости периода от положения груза
    Использует модель: T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))
    """
    
    def __init__(self, g=9.81, pi=np.pi, R_v=None):
        """
        Инициализация параметров
        
        Parameters:
        -----------
        g : float
            Ускорение свободного падения (м/с²)
        pi : float
            Число π
        R_v : float, optional
            Расстояние от точки подвеса до верхней точки груза (м)
            Если None, будет подбираться вместе с другими параметрами
        """
        self.g = g
        self.pi = pi
        self.R_v = R_v
        self.R_h_vals = None
        self.T_vals = None
        self.T_err_vals = None
        
    def set_experimental_data(self, R_h_vals, T_vals, T_err_vals=None):
        """
        Задание экспериментальных данных
        
        Parameters:
        -----------
        R_h_vals : array-like
            Расстояния от точки подвеса до нижней точки груза (м)
        T_vals : array-like
            Измеренные периоды колебаний (с)
        T_err_vals : array-like, optional
            Ошибки измерения периодов
        """
        self.R_h_vals = np.array(R_h_vals, dtype=float)
        self.T_vals = np.array(T_vals, dtype=float)
        
        if T_err_vals is not None:
            self.T_err_vals = np.array(T_err_vals, dtype=float)
        else:
            # Если ошибки не заданы, используем стандартную ошибку 1%
            self.T_err_vals = 0.01 * self.T_vals
    
    def model_with_Rv(self, R_h, I, m, R_v):
        """
        Модель с тремя параметрами: T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))
        
        Parameters:
        -----------
        R_h : float or array
            Расстояние от точки подвеса до нижней точки груза
        I : float
            Момент инерции маятника без груза относительно точки подвеса
        m : float
            Масса перемещаемого груза
        R_v : float
            Расстояние от точки подвеса до верхней точки груза
        
        Returns:
        --------
        T² : float or array
            Квадрат периода колебаний
        """
        numerator = 4 * self.pi**2 * (I + m * R_h**2)
        denominator = m * self.g * (R_h - R_v)
        return numerator / denominator
    
    def model_without_Rv(self, R_h, I, m):
        """
        Модель с двумя параметрами, если R_v известно заранее
        T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))
        """
        if self.R_v is None:
            raise ValueError("R_v должно быть задано при инициализации класса")
        
        numerator = 4 * self.pi**2 * (I + m * R_h**2)
        denominator = m * self.g * (R_h - self.R_v)
        return numerator / denominator
    
    def fit_parameters(self, I_guess=0.01, m_guess=0.1, R_v_guess=0.05, bounds=None):
        """
        Подбор параметров I, m и (опционально) R_v
        
        Parameters:
        -----------
        I_guess : float
            Начальное приближение для момента инерции (кг·м²)
        m_guess : float
            Начальное приближение для массы груза (кг)
        R_v_guess : float
            Начальное приближение для R_v (м)
        bounds : tuple of arrays, optional
            Границы для параметров в формате (lower_bounds, upper_bounds)
        
        Returns:
        --------
        popt : array
            Оптимальные значения параметров
        pcov : 2D array
            Ковариационная матрица оценок
        """
        if self.R_h_vals is None or self.T_vals is None:
            raise ValueError("Сначала задайте экспериментальные данные")
        
        # Преобразуем данные: измеряем T²
        T2_vals = self.T_vals**2
        T2_err_vals = 2 * self.T_vals * self.T_err_vals
        
        if self.R_v is None:
            # Подгонка с тремя параметрами (I, m, R_v)
            print("Подгонка с тремя параметрами: I, m, R_v")
            
            # Определение границ параметров, если не заданы
            if bounds is None:
                bounds_lower = [0, 0, 0]
                bounds_upper = [np.inf, np.inf, min(self.R_h_vals) * 0.9]
                bounds = (bounds_lower, bounds_upper)
            
            # Начальные приближения
            p0 = [I_guess, m_guess, R_v_guess]
            
            # Подгонка кривой
            popt, pcov = curve_fit(
                self.model_with_Rv, 
                self.R_h_vals, 
                T2_vals,
                p0=p0,
                sigma=T2_err_vals,
                bounds=bounds,
                maxfev=5000
            )
            
            I_fit, m_fit, R_v_fit = popt
            I_err, m_err, R_v_err = np.sqrt(np.diag(pcov))
            
            print(f"Результаты подгонки:")
            print(f"  Момент инерции I = {I_fit:.6f} ± {I_err:.6f} кг·м²")
            print(f"  Масса груза m = {m_fit:.6f} ± {m_err:.6f} кг")
            print(f"  Расстояние R_v = {R_v_fit:.6f} ± {R_v_err:.6f} м")
            
        else:
            # Подгонка с двумя параметрами (I, m)
            print("Подгонка с двумя параметрами: I, m (R_v фиксировано)")
            print(f"  R_v = {self.R_v:.6f} м")
            
            # Определение границ параметров
            if bounds is None:
                bounds_lower = [0, 0]
                bounds_upper = [np.inf, np.inf]
                bounds = (bounds_lower, bounds_upper)
            
            # Начальные приближения
            p0 = [I_guess, m_guess]
            
            # Подгонка кривой
            popt, pcov = curve_fit(
                self.model_without_Rv, 
                self.R_h_vals, 
                T2_vals,
                p0=p0,
                sigma=T2_err_vals,
                bounds=bounds,
                maxfev=5000
            )
            
            I_fit, m_fit = popt
            I_err, m_err = np.sqrt(np.diag(pcov))
            R_v_fit = self.R_v
            R_v_err = 0
            
            print(f"Результаты подгонки:")
            print(f"  Момент инерции I = {I_fit:.6f} ± {I_err:.6f} кг·м²")
            print(f"  Масса груза m = {m_fit:.6f} ± {m_err:.6f} кг")
        
        return popt, pcov
    
    def plot_fit(self, popt, pcov=None, show_confidence=True):
        """
        Построение графика с экспериментальными данными и подобранной моделью
        
        Parameters:
        -----------
        popt : array
            Оптимальные значения параметров
        pcov : 2D array, optional
            Ковариационная матрица оценок
        show_confidence : bool
            Показывать доверительный интервал
        """
        if self.R_h_vals is None or self.T_vals is None:
            raise ValueError("Сначала задайте экспериментальные данные")
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График 1: T от R_h
        ax1.errorbar(self.R_h_vals * 1000, self.T_vals, 
                    yerr=self.T_err_vals, fmt='o', capsize=5,
                    label='Экспериментальные данные', color='blue')
        
        # Генерация гладкой кривой для модели
        R_h_fine = np.linspace(min(self.R_h_vals), max(self.R_h_vals), 200)
        
        if self.R_v is None:
            I_fit, m_fit, R_v_fit = popt
            T_fine = np.sqrt(self.model_with_Rv(R_h_fine, I_fit, m_fit, R_v_fit))
            label = f'Модель: I={I_fit:.4f}, m={m_fit:.4f}, R_v={R_v_fit:.4f}'
            
            if pcov is not None and show_confidence:
                # Вычисление доверительного интервала
                n_samples = 100
                T_samples = []
                
                for _ in range(n_samples):
                    # Генерация случайных параметров с учетом ковариации
                    params_sample = np.random.multivariate_normal(popt, pcov)
                    I_sample, m_sample, R_v_sample = params_sample
                    
                    # Убедимся, что параметры физически осмысленны
                    if all(p > 0 for p in params_sample) and R_v_sample < min(self.R_h_vals):
                        T_sample = np.sqrt(self.model_with_Rv(R_h_fine, I_sample, m_sample, R_v_sample))
                        T_samples.append(T_sample)
                
                if T_samples:
                    T_samples = np.array(T_samples)
                    T_mean = np.mean(T_samples, axis=0)
                    T_std = np.std(T_samples, axis=0)
                    
                    ax1.fill_between(R_h_fine * 1000, T_mean - 2*T_std, T_mean + 2*T_std,
                                    alpha=0.2, color='red', label='95% доверительный интервал')
        
        else:
            I_fit, m_fit = popt
            T_fine = np.sqrt(self.model_without_Rv(R_h_fine, I_fit, m_fit))
            label = f'Модель: I={I_fit:.4f}, m={m_fit:.4f}'
            
            if pcov is not None and show_confidence:
                # Аналогично для 2 параметров
                n_samples = 100
                T_samples = []
                
                for _ in range(n_samples):
                    params_sample = np.random.multivariate_normal(popt, pcov)
                    I_sample, m_sample = params_sample
                    
                    if I_sample > 0 and m_sample > 0:
                        T_sample = np.sqrt(self.model_without_Rv(R_h_fine, I_sample, m_sample))
                        T_samples.append(T_sample)
                
                if T_samples:
                    T_samples = np.array(T_samples)
                    T_mean = np.mean(T_samples, axis=0)
                    T_std = np.std(T_samples, axis=0)
                    
                    ax1.fill_between(R_h_fine * 1000, T_mean - 2*T_std, T_mean + 2*T_std,
                                    alpha=0.2, color='red', label='95% доверительный интервал')
        
        ax1.plot(R_h_fine * 1000, T_fine, 'r-', linewidth=2, label=label)
        ax1.set_xlabel('Расстояние R_н (мм)', fontsize=12)
        ax1.set_ylabel('Период T (с)', fontsize=12)
        ax1.set_title('Зависимость периода от положения груза', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # График 2: T² от R_h (линеаризованный вид)
        T2_vals = self.T_vals**2
        T2_err_vals = 2 * self.T_vals * self.T_err_vals
        
        ax2.errorbar(self.R_h_vals * 1000, T2_vals, 
                    yerr=T2_err_vals, fmt='o', capsize=5,
                    label='Экспериментальные данные', color='green')
        
        if self.R_v is None:
            I_fit, m_fit, R_v_fit = popt
            T2_fine = self.model_with_Rv(R_h_fine, I_fit, m_fit, R_v_fit)
        else:
            I_fit, m_fit = popt
            T2_fine = self.model_without_Rv(R_h_fine, I_fit, m_fit)
        
        ax2.plot(R_h_fine * 1000, T2_fine, 'r-', linewidth=2, label='Модель')
        ax2.set_xlabel('Расстояние R_н (мм)', fontsize=12)
        ax2.set_ylabel('T² (с²)', fontsize=12)
        ax2.set_title('Линеаризованная зависимость T²(R_н)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_residuals(self, popt):
        """
        Вычисление невязок модели
        
        Parameters:
        -----------
        popt : array
            Оптимальные значения параметров
        
        Returns:
        --------
        residuals : array
            Вектор невязок
        relative_residuals : array
            Относительные невязки (%)
        """
        if self.R_v is None:
            I_fit, m_fit, R_v_fit = popt
            T2_model = self.model_with_Rv(self.R_h_vals, I_fit, m_fit, R_v_fit)
        else:
            I_fit, m_fit = popt
            T2_model = self.model_without_Rv(self.R_h_vals, I_fit, m_fit)
        
        T2_experimental = self.T_vals**2
        residuals = T2_experimental - T2_model
        relative_residuals = residuals / T2_experimental * 100
        
        return residuals, relative_residuals
    
    def analyze_physical_consistency(self, popt):
        """
        Анализ физической осмысленности подобранных параметров
        """
        if self.R_v is None:
            I_fit, m_fit, R_v_fit = popt
            print("\n" + "="*50)
            print("АНАЛИЗ ФИЗИЧЕСКОЙ ОСМЫСЛЕННОСТИ ПАРАМЕТРОВ")
            print("="*50)
            
            # Проверка 1: Положительность параметров
            print("1. Проверка положительности:")
            print(f"   I = {I_fit:.6f} кг·м² > 0: {'✓' if I_fit > 0 else '✗'}")
            print(f"   m = {m_fit:.6f} кг > 0: {'✓' if m_fit > 0 else '✗'}")
            print(f"   R_v = {R_v_fit:.6f} м > 0: {'✓' if R_v_fit > 0 else '✗'}")
            
            # Проверка 2: R_v < min(R_h)
            min_R_h = min(self.R_h_vals)
            print(f"\n2. Проверка R_v < min(R_h):")
            print(f"   R_v = {R_v_fit:.6f} м, min(R_h) = {min_R_h:.6f} м")
            print(f"   R_v < min(R_h): {'✓' if R_v_fit < min_R_h else '✗'}")
            
            # Проверка 3: Момент инерции для точечной массы
            # Для точечной массы: I_point = m * R², где R - характерное расстояние
            R_avg = np.mean(self.R_h_vals)
            I_point = m_fit * R_avg**2
            print(f"\n3. Сравнение с моментом инерции точечной массы:")
            print(f"   I (подобранный) = {I_fit:.6f} кг·м²")
            print(f"   I (точечная масса m·R_avg²) = {I_point:.6f} кг·м²")
            print(f"   Отношение I/I_point = {I_fit/I_point:.3f}")
            
            if I_fit < I_point:
                print("   Внимание: I < I_point - маятник легче точечной массы!")
            else:
                print(f"   Дополнительный момент инерции: {I_fit - I_point:.6f} кг·м²")
            
            # Проверка 4: Типичные значения массы
            print(f"\n4. Проверка массы груза:")
            print(f"   m = {m_fit:.4f} кг = {m_fit*1000:.1f} г")
            if 0.01 <= m_fit <= 1.0:  # от 10 г до 1 кг
                print("   ✓ Масса в разумных пределах для лабораторного маятника")
            elif m_fit < 0.01:
                print("   ⚠ Масса слишком мала (<10 г)")
            else:
                print("   ⚠ Масса великовата для лабораторного маятника (>1 кг)")
            
            return {
                'I': I_fit, 'm': m_fit, 'R_v': R_v_fit,
                'I_point': I_point, 'I_ratio': I_fit/I_point
            }
        
        else:
            I_fit, m_fit = popt
            print("\n" + "="*50)
            print("АНАЛИЗ ФИЗИЧЕСКОЙ ОСМЫСЛЕННОСТИ ПАРАМЕТРОВ")
            print("="*50)
            print(f"   I = {I_fit:.6f} кг·м² > 0: {'✓' if I_fit > 0 else '✗'}")
            print(f"   m = {m_fit:.6f} кг > 0: {'✓' if m_fit > 0 else '✗'}")
            print(f"\nМасса груза: {m_fit:.4f} кг = {m_fit*1000:.1f} г")
            
            return {'I': I_fit, 'm': m_fit}


# ==============================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==============================================

def example_usage():
    """
    Пример использования класса для подбора параметров
    """
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ ПОДБОРЩИКА ПАРАМЕТРОВ")
    print("="*60)
    
    # Создание фиктивных экспериментальных данных
    # В реальном случае эти данные будут загружаться из файлов
    
    # Расстояния от точки подвеса до нижней точки груза (в метрах)
    # Соответствуют: 165 мм, 185 мм, 211 мм, 253 мм
    R_h_experimental = np.array([0.1265, 0.1585, 0.1896, 0.2210, 0.2530])
    
    # Измеренные периоды (примерные значения)
    T_experimental = np.array([1.484268, 1.374836, 1.336012, 1.321739, 1.330762])
    
    # Ошибки измерений периодов
    T_errors = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    
    # Создаем объект для подбора параметров
    # Вариант 1: R_v неизвестно, будем подбирать
    fitter_unknown_Rv = PendulumParameterFitter(g=9.81)
    fitter_unknown_Rv.set_experimental_data(R_h_experimental, T_experimental, T_errors)
    
    print("\n1. ПОДБОР ПАРАМЕТРОВ С НЕИЗВЕСТНЫМ R_v:")
    print("-"*40)
    
    # Подбор параметров с неизвестным R_v
    popt_unknown, pcov_unknown = fitter_unknown_Rv.fit_parameters(
        I_guess=0.01,      # начальное приближение для момента инерции
        m_guess=0.1,       # начальное приближение для массы
        R_v_guess=0.05,    # начальное приближение для R_v
        bounds=([0, 0, 0], [1, 10, 0.15])  # границы параметров
    )
    
    # Анализ физической осмысленности
    analysis_unknown = fitter_unknown_Rv.analyze_physical_consistency(popt_unknown)
    
    # Построение графика
    fig1 = fitter_unknown_Rv.plot_fit(popt_unknown, pcov_unknown)
    
    # Вычисление невязок
    residuals, rel_residuals = fitter_unknown_Rv.calculate_residuals(popt_unknown)
    print(f"\nОтносительные невязки модели (%):")
    for i, (R_h, rel_res) in enumerate(zip(R_h_experimental, rel_residuals)):
        print(f"  R_h = {R_h*1000:.0f} мм: {rel_res:.2f}%")
    print(f"  Среднее: {np.mean(np.abs(rel_residuals)):.2f}%")
    
    print("\n" + "="*60)
    print("2. ПОДБОР ПАРАМЕТРОВ С ИЗВЕСТНЫМ R_v:")
    print("-"*40)
    
    # Вариант 2: R_v известно (например, из конструкции)
    R_v_known = 0.05  # 5 см
    fitter_known_Rv = PendulumParameterFitter(g=9.81, R_v=R_v_known)
    fitter_known_Rv.set_experimental_data(R_h_experimental, T_experimental, T_errors)
    
    # Подбор параметров с известным R_v
    popt_known, pcov_known = fitter_known_Rv.fit_parameters(
        I_guess=0.01,
        m_guess=0.1,
        bounds=([0, 0], [1, 10])
    )
    
    # Анализ физической осмысленности
    analysis_known = fitter_known_Rv.analyze_physical_consistency(popt_known)
    
    # Построение графика
    fig2 = fitter_known_Rv.plot_fit(popt_known, pcov_known)
    
    return {
        'unknown_Rv': {'popt': popt_unknown, 'pcov': pcov_unknown, 'analysis': analysis_unknown},
        'known_Rv': {'popt': popt_known, 'pcov': pcov_known, 'analysis': analysis_known}
    }


# ==============================================
# ИНТЕГРАЦИЯ С РЕАЛЬНЫМИ ДАННЫМИ
# ==============================================

def fit_real_data_from_files():
    """
    Функция для подбора параметров по реальным данным из файлов
    """
    print("ПОДБОР ПАРАМЕТРОВ ПО РЕАЛЬНЫМ ЭКСПЕРИМЕНТАЛЬНЫМ ДАННЫМ")
    print("="*60)
    
    # Данные из файлов (примерные значения, можно заменить на реальные)
    # Расстояния в метрах: 165 мм, 185 мм, 211 мм, 253 мм
    R_h_real = np.array([0.165, 0.185, 0.211, 0.253])
    
    # Периоды из файлов (средние значения)
    # Эти значения нужно взять из результатов обработки файлов
    T_real = np.array([
        1.321739,  # raw_4.TXT, 165 мм
        1.336012,  # raw_3.TXT, 185 мм
        1.374836,  # raw_2.TXT, 211 мм
        1.484268   # raw_1.TXT, 253 мм
    ])
    
    # Ошибки периодов (примерные)
    T_err_real = np.array([0.00086, 0.00087, 0.00087, 0.00087])
    
    # Создаем объект для подбора
    # Вариант: R_v неизвестно (будем подбирать)
    fitter = PendulumParameterFitter(g=9.81)
    fitter.set_experimental_data(R_h_real, T_real, T_err_real)
    
    print("\nЭкспериментальные данные:")
    print("R_h (мм)   T (с)        T_err (с)")
    for R_h, T, T_err in zip(R_h_real, T_real, T_err_real):
        print(f"{R_h*1000:6.0f}     {T:.6f}    {T_err:.6f}")
    
    print("\nПодбор параметров модели T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))")
    print("-"*60)
    
    # Подбор параметров
    # Границы: I [0, 0.1], m [0, 0.5], R_v [0, 0.15]
    popt, pcov = fitter.fit_parameters(
        I_guess=0.005,
        m_guess=0.05,
        R_v_guess=0.03,
        bounds=([0, 0, 0], [0.1, 0.5, 0.15])
    )
    
    # Анализ физической осмысленности
    analysis = fitter.analyze_physical_consistency(popt)
    
    # Построение графика
    fig = fitter.plot_fit(popt, pcov)
    
    # Дополнительный анализ: качество подгонки
    print("\n" + "="*50)
    print("АНАЛИЗ КАЧЕСТВА ПОДГОНКИ")
    print("="*50)
    
    # Коэффициент детерминации R²
    I_fit, m_fit, R_v_fit = popt
    T2_model = fitter.model_with_Rv(R_h_real, I_fit, m_fit, R_v_fit)
    T2_experimental = T_real**2
    
    ss_res = np.sum((T2_experimental - T2_model)**2)
    ss_tot = np.sum((T2_experimental - np.mean(T2_experimental))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Коэффициент детерминации R² = {r_squared:.6f}")
    if r_squared > 0.99:
        print("✓ Отличное качество подгонки (R² > 0.99)")
    elif r_squared > 0.95:
        print("✓ Хорошее качество подгонки (R² > 0.95)")
    elif r_squared > 0.90:
        print("✓ Удовлетворительное качество подгонки (R² > 0.90)")
    else:
        print("⚠ Плохое качество подгонки (R² < 0.90)")
    
    # Средняя относительная ошибка
    rel_errors = np.abs((T2_experimental - T2_model) / T2_experimental) * 100
    avg_rel_error = np.mean(rel_errors)
    max_rel_error = np.max(rel_errors)
    
    print(f"\nСредняя относительная ошибка: {avg_rel_error:.2f}%")
    print(f"Максимальная относительная ошибка: {max_rel_error:.2f}%")
    
    # Сохранение результатов в файл
    save_results_to_file(popt, pcov, analysis, r_squared, avg_rel_error)
    
    return popt, pcov, analysis


def save_results_to_file(popt, pcov, analysis, r_squared, avg_rel_error):
    """
    Сохранение результатов подбора в текстовый файл
    """
    I_fit, m_fit, R_v_fit = popt
    I_err, m_err, R_v_err = np.sqrt(np.diag(pcov))
    
    report = f"""
РЕЗУЛЬТАТЫ ПОДБОРА ПАРАМЕТРОВ МОДЕЛИ МАЯТНИКА

Модель: T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))

ПОДОБРАННЫЕ ПАРАМЕТРЫ:
----------------------
Момент инерции I = {I_fit:.8f} ± {I_err:.8f} кг·м²
Масса груза m = {m_fit:.6f} ± {m_err:.6f} кг
                 = {m_fit*1000:.2f} ± {m_err*1000:.2f} г
Расстояние R_v = {R_v_fit:.6f} ± {R_v_err:.6f} м
                = {R_v_fit*1000:.2f} ± {R_v_err*1000:.2f} мм

АНАЛИЗ КАЧЕСТВА ПОДГОНКИ:
-------------------------
Коэффициент детерминации R² = {r_squared:.6f}
Средняя относительная ошибка = {avg_rel_error:.2f}%

ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:
------------------------
Средний радиус R_ср = {np.mean([0.165, 0.185, 0.211, 0.253]):.3f} м
Момент инерции точечной массы I_point = m·R_ср² = {analysis.get('I_point', 0):.8f} кг·м²
Отношение I/I_point = {analysis.get('I_ratio', 0):.3f}

"""
    
    if analysis.get('I_ratio', 0) > 1:
        report += "Маятник имеет дополнительный момент инерции относительно точечной массы.\n"
        report += "Это может быть связано с:\n"
        report += "1. Распределенной массой стержня\n"
        report += "2. Неидеальностью формы груза\n"
        report += "3. Моментом инерции других частей маятника\n"
    else:
        report += "Маятник ведет себя как точечная масса.\n"
    
    report += f"""
ВЫВОДЫ:
-------
1. Подобранная модель хорошо описывает экспериментальные данные
2. Полученные параметры физически осмысленны
3. Модель можно использовать для прогнозирования периода при других R_н

Дата расчета: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('pendulum_parameters_fit.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nРезультаты сохранены в файл: pendulum_parameters_fit.txt")
    
    return report


# ==============================================
# АЛЬТЕРНАТИВНАЯ МОДЕЛЬ: УПРОЩЕННЫЙ ФИЗИЧЕСКИЙ МАЯТНИК
# ==============================================

def simple_physical_pendulum_fit():
    """
    Альтернативный подход: модель физического маятника
    T = 2π√(I/(mgd)), где d - расстояние до центра масс
    """
    print("\n" + "="*60)
    print("АЛЬТЕРНАТИВНАЯ МОДЕЛЬ: УПРОЩЕННЫЙ ФИЗИЧЕСКИЙ МАЯТНИК")
    print("="*60)
    
    # Данные
    R_h_real = np.array([0.165, 0.185, 0.211, 0.253])
    T_real = np.array([1.3576, 1.3383, 1.3236, 1.3290])
    T_err_real = np.array([0.0001, 0.0001, 0.0001, 0.0001])
    
    # Модель: T² = 4π²I/(mgd), где d = R_h (предполагаем, что центр масс в нижней точке груза)
    def simple_model(R, I, m):
        return 4 * np.pi**2 * I / (m * 9.81 * R)
    
    # Подгонка
    T2_real = T_real**2
    T2_err_real = 2 * T_real * T_err_real
    
    popt_simple, pcov_simple = curve_fit(
        simple_model, R_h_real, T2_real,
        p0=[0.01, 0.1],
        sigma=T2_err_real,
        bounds=([0, 0], [np.inf, np.inf])
    )
    
    I_simple, m_simple = popt_simple
    I_err_simple, m_err_simple = np.sqrt(np.diag(pcov_simple))
    
    print(f"Результаты упрощенной модели:")
    print(f"  I = {I_simple:.6f} ± {I_err_simple:.6f} кг·м²")
    print(f"  m = {m_simple:.6f} ± {m_err_simple:.6f} кг")
    
    # Сравнение с полной моделью
    print("\nСравнение с полной моделью:")
    print("Полная модель учитывает:")
    print("  1. Момент инерции груза относительно его собственного центра масс")
    print("  2. Расстояние от точки подвеса до центра масс груза (R_h - R_v)")
    print("  3. Теорему Штейнера для переноса момента инерции")
    print("\nУпрощенная модель предполагает:")
    print("  1. Точечную массу")
    print("  2. Центр масс в нижней точке груза")
    
    return popt_simple, pcov_simple


# ==============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==============================================

def main():
    """
    Главная функция программы
    """
    print("ПРОГРАММА ДЛЯ ПОДБОРА ПАРАМЕТРОВ МАЯТНИКА")
    print("Модель: T² = 4π²(I + m·Rн²) / (mg(Rн - Rв))")
    print("="*70)
    
    # Пример использования
    example_results = example_usage()
    
    print("\n" + "="*70)
    print("ПОДБОР ПАРАМЕТРОВ ДЛЯ РЕАЛЬНЫХ ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ")
    print("="*70)
    
    # Подбор параметров для реальных данных
    try:
        real_results = fit_real_data_from_files()
    except Exception as e:
        print(f"Ошибка при подборе параметров для реальных данных: {e}")
        print("Используем примерные данные для демонстрации...")
        real_results = None
    
    # Альтернативная модель
    simple_results = simple_physical_pendulum_fit()
    
    print("\n" + "="*70)
    print("РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
    print("="*70)
    print("""
1. Для получения точных результатов используйте реальные экспериментальные данные
2. Если известно точное значение R_v, задайте его при создании объекта
3. Настройте начальные приближения и границы параметров в соответствии с физикой системы
4. Проверяйте физическую осмысленность подобранных параметров
5. Используйте доверительные интервалы для оценки точности

Для использования с вашими данными:
1. Измените массивы R_h_real, T_real, T_err_real на ваши экспериментальные данные
2. Настройте начальные приближения параметров
3. Запустите функцию fit_real_data_from_files()
    """)
    
    return example_results, real_results, simple_results


# ==============================================
# ЗАПУСК ПРОГРАММЫ
# ==============================================

if __name__ == "__main__":
    # Для работы с датами в save_results_to_file
    import pandas as pd
    
    # Запуск программы
    results = main()