#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <bits/stdc++.h>

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
    int sizes[] = {1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000, 400000, 500000};
    auto filename = "0.csv";

    ofstream f(filename, ios::out); // csv - стандартный формат для хранения данных прямым текстом.
     // работаете как с привычным cout

    for (int i = 0; i < 12; i++) {
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
        auto end12 = std::chrono::high_resolution_clock::now();
        auto nsec = end12 - start;

        std::cout << "bubblesort " << size << " " << nsec.count() << " нсек." << std::endl;
        f << "bubblesort;" << size << ";" << nsec.count() << endl;

        auto arr1 = generateRandomArray(size, -10000, 10000);

        auto start1 = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < size; i++) {
            for (auto j = i; (j > 0) && (arr1[j] > arr1[j - 1]); j--) {
                std::swap(arr1[j], arr1[j - 1]);
            }
        }

        // здесь то что вы хотите измерить
        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;
        std::cout << "insertions " << size << " " << nsec1.count() << " нсек." << std::endl;
        f << "insertions;" << size << ";" << nsec1.count() << endl;

        auto arr2 = generateRandomArray(size, -10000, 10000);
        auto start2 = std::chrono::high_resolution_clock::now();

        int n = size;

        for (int i = 0; i < n - 1; ++i) {

            // Assume the current position holds
            // the minimum element
            int min_idx = i;

            // Iterate through the unsorted portion
            // to find the actual minimum
            for (int j = i + 1; j < n; ++j) {
                if (arr2[j] < arr2[min_idx]) {

                    // Update min_idx if a smaller
                    // element is found
                    min_idx = j;
                }
            }

            // Move minimum element to its
            // correct position
            swap(arr2[i], arr2[min_idx]);
        }

        // здесь то что вы хотите измерить
        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end2 - start2;

        std::cout << "selectionsort " << size << " " << nsec2.count() << " нсек." << std::endl;
        f << "selectionsort;" << size << ";" << nsec2.count() << endl;


    }
    return 0;
}

