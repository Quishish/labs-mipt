// lab3_push_front.cpp
#include <iostream>
#include <list>
#include <forward_list>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

// 🔌 Подключение своего контейнера
// #include "subforward_list.hpp"

// 🔥 СМЕНА КОНТЕЙНЕРА: меняйте только эту строку!
using TestContainer = std::forward_list<int>;
// using TestContainer = std::list<int>;
// using TestContainer = subforward_list<int>;

// Размеры до ~10 МБ (для int: ~2.5M элементов)
const std::vector<size_t> SIZES = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000};
const int TRIALS = 10;

int main() {
    std::string csv_name = "results/test3_push_front-forwardlist.csv";
    std::ofstream out(csv_name);
    if (!out) {
        std::cerr << "❌ Ошибка создания " << csv_name << "\n";
        return 1;
    }
    out << "size,avg_time_ms\n";

    for (size_t sz : SIZES) {
        double total_ms = 0.0;

        for (int t = 0; t < TRIALS; ++t) {
            TestContainer cont;
            
            // ⏱ Замеряем ТОЛЬКО push_front
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < sz; ++i) {
                cont.push_front(static_cast<int>(i));
            }
            auto end = std::chrono::high_resolution_clock::now();

            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }

        double avg = total_ms / TRIALS;
        out << sz << "," << avg << "\n";
        std::cout << "📏 Size: " << sz << " | ⏱ Avg: " << avg << " ms\n";
    }

    out.close();
    std::cout << "\n Готово. Данные в: " << csv_name << "\n";
    return 0;
}