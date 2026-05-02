# plot_test0.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_capacity_growth(csv_file, output_png):
    df = pd.read_csv(csv_file)
    
    # Для наглядности прореживаем данные (каждую 100-ю точку), если их много
    if len(df) > 1000:
        df = df.iloc[::100]
    
    plt.figure(figsize=(12, 7))
    
    # Линия size — сплошная
    plt.plot(df['iteration'], df['size'], 
             label='size()', color='#2E86AB', linewidth=2)
    
    # Линия capacity — пунктирная
    plt.plot(df['iteration'], df['capacity'], 
             label='capacity()', color='#A23B72', linewidth=2, linestyle='--')
    
    # Оформление
    plt.xlabel('Номер итерации (количество push_back)', fontsize=11)
    plt.ylabel('Количество элементов', fontsize=11)
    plt.title('Рост std::vector: size() vs capacity() при push_back', fontsize=13, pad=15)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Подпись про стратегию роста
    plt.text(0.02, 0.98, 
             '📌 Ступеньки capacity — геометрический рост (обычно ×1.5 или ×2)', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранён: {output_png}")
    plt.close()

if __name__ == "__main__":
    plot_capacity_growth("results/test0_push_back.csv", "results/plot_test0.png")