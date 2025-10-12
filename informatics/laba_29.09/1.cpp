//почти то же самое что и в 1 пункте, но надо поочереди прогнать с разной оптимизацией чек методичку.

#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

using namespace std;

// Функция для генерации массива случайных чисел
std::vector<double> generateRandomArray(int n, double lowerBound, double upperBound) {
    // Инициализация генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(lowerBound, upperBound);

    // Создание и заполнение массива
    std::vector<double> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }
    return arr;
}

int main()
{
    auto filename = "1.csv";
    int sizes[] = {1000, 5000, 10000, 50000, 100000};

    ofstream f(filename, ios::out); // csv - стандартный формат для хранения данных прямым текстом.

    for (int i = 0; i < 5; i++) {
        int size = sizes[i];

        auto arr0 = generateRandomArray(size, -10000, 10000);
        auto start = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < size; i++)
            for (auto j = 0; j < size - 1; j++)
                if (arr0[j]>arr0[j + 1]) {
                    int tmp = arr0[j];
                    arr0[j] = arr0[j + 1];
                    arr0[j + 1] = tmp;
                }

        // здесь то что вы хотите измерить
        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        std::cout << "bubblesort " << size << " " << nsec.count() << " нсек." << std::endl;
        f << "bubblesort;" << size << ";" << nsec.count() << endl;
    }
    return 0;
}
