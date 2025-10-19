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

void bubbleSort(vector<double>& arr0) {
    int size = arr0.size();
    for (auto i = 0; i < size; i++) {
        for (auto j = 0; j < size - 1; j++)
            if (arr0[j]>arr0[j + 1]) {
                int tmp = arr0[j];
                arr0[j] = arr0[j + 1];
                arr0[j + 1] = tmp;
        }
    }

}

void selectionSort(vector<double>& arr0) {
    int n = std::size(arr0);

    for (int i = 0; i < n - 1; ++i) {

        // Assume the current position holds
        // the minimum element
        int min_idx = i;

        // Iterate through the unsorted portion
        // to find the actual minimum
        for (int j = i + 1; j < n; ++j) {
            if (arr0[j] < arr0[min_idx]) {

                // Update min_idx if a smaller
                // element is found
                min_idx = j;
            }
        }

        // Move minimum element to its
        // correct position
        swap(arr0[i], arr0[min_idx]);
    }
}

void insertionSort(vector<double>& arr0) {
    int size = arr0.size(); 
    for (auto i = 0; i < size; i++) {
        for (auto j = i; (j > 0) && (arr0[j] < arr0[j - 1]); j--) {
            std::swap(arr0[j], arr0[j - 1]);
        }
    }
}

int main() {
    int sizes[] = {1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000};
    auto filename = "4-selections.csv";
    ofstream f(filename, ios::out);

    for (int i = 0; i < 10; i++) {
        vector<double> UnsortedArray = generateUnSortedArray(sizes[i]);
        vector<double> SortedArray = generateSortedArray(sizes[i]);
        vector<double> randomArray = generateRandomArray(sizes[i], -10000, 10000);

        int size = sizes[i];
        auto start = std::chrono::high_resolution_clock::now();

        // bubbleSort(randomArray);
        //insertionSort(randomArray);
        selectionSort(randomArray);

        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        auto start1 = std::chrono::high_resolution_clock::now();

        //bubbleSort(SortedArray);
        //insertionSort(SortedArray);
        selectionSort(SortedArray);

        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;

        auto start2 = std::chrono::high_resolution_clock::now();

        //bubbleSort(UnsortedArray);
        //insertionSort(UnsortedArray);
        selectionSort(UnsortedArray);

        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end2 - start2;

        std::cout << size << " " << nsec.count() << " нсек." << std::endl;
        f << size << ";" << nsec.count() << ";" << nsec1.count() << ";" << nsec2.count() << ";nsec." << endl;

        }
}

