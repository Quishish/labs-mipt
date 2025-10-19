#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm> // для std::swap

using namespace std; // Добавлено

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

// Исправлено: передача по ссылке &
int partition(vector<double> &vec, int low, int high) { // Добавлено &

    // Selecting last element as the pivot
    double pivot = vec[high]; // Исправлен тип на double

    // Index of element just before the last element
    // It is used for swapping
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {

        // If current element is smaller than or
        // equal to pivot
        if (vec[j] <= pivot) {
            i++;
            swap(vec[i], vec[j]);
        }
    }

    // Put pivot to its position
    swap(vec[i + 1], vec[high]);

    // Return the point of partition
    return (i + 1);
}

void my_qsort(vector<double>& arr, int l, int r)
{
    int left = l;
    int right = r;
    double mid = arr[(left + right) / 2];

    while (left <= right) {
        while (arr[left] < mid) {
            left++;
        }
        while (arr[right] > mid) {
            right--;
        }
        if (left <= right) {
            swap(arr[left++], arr[right--]);
        }
    }
    if (l < right) {
        my_qsort(arr, l, right);
    }
    if (r > left) {
        my_qsort(arr, left, r);
    }
}


// Исправлено: передача по ссылке &
void quickSort(vector<double> &vec, int low, int high) { // Добавлено &


    if (low < high) {

        // pi is Partitioning Index, arr[p] is now at
        // right place
        int pi = partition(vec, low, high);


        quickSort(vec, low, pi - 1);
        quickSort(vec, pi + 1, high);
    }
}

int main() {
    auto arr = generateRandomArray(1000, -1000, 1000);
    auto arr1 = arr; // ОШИБКА 6: нужно копировать массив для честного сравнения
    auto arr2 = arr; // Для второго теста

    // Тестируем первую версию quickSort
    quickSort(arr, 0, arr.size() - 1);

    // Тестируем my_qsort на копии массива
    my_qsort(arr1, 0, arr1.size() - 1);

    bool sorted = true;
    for (size_t i = 0; i < arr.size() - 1; i++) {
        if (arr[i] > arr[i+1]) {
            cout << "First algorithm: unsorted at index " << i << endl;
            sorted = false;
            break;
        }
    }

    if (sorted) {
        cout << "First array is sorted correctly" << endl;
    }

    bool sorted1 = true;
    // ОШИБКА 7: проверяем arr1, а не arr
    for (size_t j = 0; j < arr1.size() - 1; j++) {
        if (arr1[j] > arr1[j+1]) { // Исправлено: arr1 вместо arr
            cout << "Second algorithm: unsorted at index " << j << endl;
            sorted1 = false;
            break;
        }
    }

    if (sorted1) {
        cout << "Second array is sorted correctly" << endl;
    }

    // Дополнительная проверка: сравниваем результаты двух алгоритмов
    bool same_result = true;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] != arr1[i]) {
            same_result = false;
            cout << "Algorithms produced different results at index " << i << endl;
            break;
        }
    }

    if (same_result) {
        cout << "Both algorithms produced identical results" << endl;
    }

    return 0;
}
