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
    auto filename = "1o1.csv";
    int sizes[] = {1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000, 400000};

    ofstream f(filename, ios::out); // csv - стандартный формат для хранения данных прямым текстом.

    for (int i = 0; i < 11; i++) {
        int size = sizes[i];

        auto arr = generateRandomArray(size, -10000, 10000);
        auto start = std::chrono::high_resolution_clock::now();

        int n = size;

        for (int i = 0; i < n - 1; ++i) {

            // Assume the current position holds
            // the minimum element
            int min_idx = i;

            // Iterate through the unsorted portion
            // to find the actual minimum
            for (int j = i + 1; j < n; ++j) {
                if (arr[j] < arr[min_idx]) {

                    // Update min_idx if a smaller
                    // element is found
                    min_idx = j;
                }
            }

            // Move minimum element to its
            // correct position
            swap(arr[i], arr[min_idx]);
        }


        // здесь то что вы хотите измерить
        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        std::cout << "selectionsort " << size << " " << nsec.count() << " ns." << std::endl;
        f << "selectionsort;" << size << ";" << nsec.count() << endl;
    }
    return 0;
}
