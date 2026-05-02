// lab2_erase_benchmark.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <string>

// 🔌 Подключение рукописного контейнера (раскомментируйте при необходимости)
#include "subvector.hpp"

// 🔥 СМЕНА КОНТЕЙНЕРА: замените только эту строку!
//using TestContainer = std::vector<int>;
using TestContainer = subvector<int>;

// Размеры контейнеров (до ~10 МБ для int: 2.5M элементов)
const std::vector<size_t> SIZES = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000};
const int TRIALS = 100;  // Повторений для усреднения

int main() {
    std::string csv_name = "results/test2_erase-subvector.csv";
    std::ofstream out(csv_name);
    if (!out.is_open()) {
        std::cerr << "❌ Не удалось открыть " << csv_name << "\n";
        return 1;
    }
    out << "size,avg_time_ms\n";

    std::mt19937 gen(42);  // Фиксированный seed для воспроизводимости

    for (size_t sz : SIZES) {
        // 1. Подготовка: заполняем контейнер (время НЕ замеряется)
        TestContainer cont;
        //cont.reserve(sz);  // Оптимизация для std::vector
        for (size_t i = 0; i < sz; ++i) {
            cont.push_back(static_cast<int>(i));
        }

        double total_ms = 0.0;

        // 2. Замер удаления из случайной позиции
        for (int t = 0; t < TRIALS; ++t) {
            std::uniform_int_distribution<size_t> dist(0, sz - 1);
            size_t pos_idx = dist(gen);

            // ⏱ Замеряем ТОЛЬКО erase
            auto start = std::chrono::high_resolution_clock::now();
            cont.erase(cont.begin() + pos_idx);  // 🔹 ИЗМЕРЯЕМАЯ ОПЕРАЦИЯ
            auto end = std::chrono::high_resolution_clock::now();

            total_ms += std::chrono::duration<double, std::milli>(end - start).count();

            // 🔁 Восстанавливаем элемент, чтобы размер не менялся
            cont.insert(cont.begin() + pos_idx, static_cast<int>(pos_idx));
        }

        double avg = total_ms / TRIALS;
        out << sz << "," << avg << "\n";
        std::cout << "📏 Size: " << sz << " | ⏱ Avg erase: " << avg << " ms\n";
    }

    out.close();
    std::cout << "\n✅ Готово. Данные сохранены в: " << csv_name << "\n";
    return 0;
}