// test0_push_back.cpp
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    const size_t MAX_ITERATIONS = 100000;  // Можно менять под нужный объём (~10 МБ)
    
    std::ofstream out("results/test0_push_back.csv");
    if (!out.is_open()) {
        std::cerr << "Не удалось открыть файл для записи!\n";
        return 1;
    }
    
    // Заголовок CSV
    out << "iteration,size,capacity,size_bytes,capacity_bytes\n";
    
    std::vector<int> vec;  // 🔥 Меняйте тип контейнера здесь, если тестируете свой
    // using TestContainer = subvector<int>;  // ← для рукописного контейнера
    
    std::cout << "🚀 Запуск теста 0: push_back growth...\n";
    
    for (size_t i = 0; i < MAX_ITERATIONS; ++i) {
        vec.push_back(static_cast<int>(i));
        
        size_t sz = vec.size();
        size_t cap = vec.capacity();
        
        // Пишем в CSV: номер итерации, size, capacity, и размер в байтах (для наглядности)
        out << i << "," 
            << sz << "," 
            << cap << ","
            << sz * sizeof(int) << ","
            << cap * sizeof(int) << "\n";
        
        // Опционально: вывод прогресса каждые 20000 итераций
        if (i % 20000 == 0 && i > 0) {
            std::cout << "  ✓ Итерация " << i << " / " << MAX_ITERATIONS 
                      << " | size=" << sz << ", capacity=" << cap << "\n";
        }
    }
    
    out.close();
    
    std::cout << "✅ Тест завершён!\n";
    std::cout << "📊 Данные сохранены в: results/test0_push_back.csv\n";
    std::cout << "📈 Размер вектора в конце: " << vec.size() 
              << " элементов (~" << (vec.size() * sizeof(int) / 1024 / 1024) << " МБ)\n";
    
    return 0;
}