// lab1_insert_benchmark.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <string>

#include "subvector.hpp"

// СМЕНА КОНТЕЙНЕРА: меняйте только эту строку!
//using TestContainer = std::vector<int>;
using TestContainer = subvector<int>;

// Размеры до ~10 МБ (для int: 2.5 млн элементов ≈ 10 МБ)
const std::vector<size_t> SIZES = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000};
const int TRIALS = 100; // усреднение по повторениям

int main() {
    std::string csv_name = "results/test1_insert-subvector.csv";
    std::ofstream out(csv_name);
    if (!out) {
        std::cerr << "❌ Ошибка создания файла " << csv_name << "\n";
        return 1;
    }
    out << "size,avg_time_ms\n";

    std::mt19937 gen(42); // фиксированный seed

    for (size_t sz : SIZES) {
        TestContainer cont;
        for (size_t i = 0; i < sz; ++i) {
            cont.push_back(static_cast<int>(i));
        }

        

        double total_ms = 0.0;

        // 2. Замер вставки в случайную позицию
        for (int t = 0; t < TRIALS; ++t) {
            std::uniform_int_distribution<size_t> dist(0, sz);
            size_t pos_idx = dist(gen);

            auto start = std::chrono::high_resolution_clock::now();
            cont.insert(cont.begin() + pos_idx, -1); // 🔹 ИЗМЕРЯЕМАЯ ОПЕРАЦИЯ
            auto end = std::chrono::high_resolution_clock::now();

            total_ms += std::chrono::duration<double, std::milli>(end - start).count();

            // Удаляем вставленный элемент, чтобы size оставался = sz
            cont.erase(cont.begin() + pos_idx);
        }

        double avg = total_ms / TRIALS;
        out << sz << "," << avg << "\n";
        std::cout << "📏 Size: " << sz << " | ⏱ Avg: " << avg << " ms\n";
    }

    out.close();
    std::cout << "\nГотово. Данные в: " << csv_name << "\n";
    return 0;
}