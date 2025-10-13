#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm> // для std::swap

using namespace std;

std::vector<double> generateSortedArray(int n) {
    std::vector<double> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }
    return arr;
}

std::vector<double> generateUnSortedArray(int n) {
    std::vector<double> arr(n);
    for (int i = n-1; i >= 0; i--) {
        arr[i] = n - i;
    }
    return arr;
}

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

void combSort(vector<double>& arr) {
    int n = arr.size();
        int gap = n;
        const float shrinkFactor = 1.3;
        bool sorted = false;
        while (gap > 1 || !sorted) {
            gap = max(1, int(gap / shrinkFactor));
            sorted = true;
            for (int i = 0; i < n - gap; ++i) {
                if (arr[i] > arr[i + gap]) {
                    swap(arr[i], arr[i + gap]);
                    sorted = false;
                }
            }
        }

}

int main() {
    int sizes[] = {1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000};
    auto filename = "4.csv";
    ofstream f(filename, ios::out);

    for (int i = 0; i < 10; i++) {
        vector<double> UnsortedArray = generateUnSortedArray(sizes[i]);
        vector<double> SortedArray = generateSortedArray(sizes[i]);
        vector<double> randomArray = generateRandomArray(sizes[i], -10000, 10000);

        int size = sizes[i];
        auto start = std::chrono::high_resolution_clock::now();

        combSort(randomArray);

        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        auto start1 = std::chrono::high_resolution_clock::now();

        combSort(SortedArray);

        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;

        auto start2 = std::chrono::high_resolution_clock::now();

        combSort(UnsortedArray);

        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end2 - start2;

        std::cout << "combsort " << size << " " << nsec.count() << " нсек." << std::endl;
        f << "combsort;" << size << ";" << nsec.count() << ";" << nsec1.count() << ";" << nsec2.count() << ";nsec." << endl;

        }
}

