import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_insert_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Загружаем данные
    try:
        df_std = pd.read_csv("results/test1_insert-vector.csv")
        ax.loglog(df_std['size'], df_std['avg_time_ms'], 'o-', label='std::vector', linewidth=2)
    except FileNotFoundError:
        print("⚠️ Не найден results/test1_insert_std.csv")
        
    try:
        df_cust = pd.read_csv("results/test1_insert-subvector.csv")
        ax.loglog(df_cust['size'], df_cust['avg_time_ms'], 's--', label='subvector', linewidth=2)
    except FileNotFoundError:
        print("⚠️ Не найден results/test1_insert_custom.csv")

    # Референсная линия O(N) для визуальной проверки асимптотики
    if 'df_std' in locals():
        x_ref = df_std['size']
        # Масштабируем линию так, чтобы она проходила через первую точку
        k = df_std['avg_time_ms'].iloc[0] / df_std['size'].iloc[0]
        ax.loglog(x_ref, k * x_ref, 'k:', label='O(n) референс', alpha=0.6)

    ax.set_xlabel('Размер контейнера (N)', fontsize=12)
    ax.set_ylabel('Среднее время вставки, мс', fontsize=12)
    ax.set_title('Пункт 1: Вставка в произвольное место вектора', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/plot_test1_insert.png", dpi=300)
    print("✅ График сохранён: results/plot_test1_insert.png")

if __name__ == "__main__":
    plot_insert_comparison()