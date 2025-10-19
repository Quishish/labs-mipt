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

void heapify(vector<double>& arr, int n, int i)
{
    int largest = i; // Initialize largest as root Since we are using 0 based indexing
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (largest != i) {
        swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

// main function to do heap sort
void heapSort(vector<double>& arr)
{
    int n = arr.size();
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // One by one extract an element from heap
    for (int i = n - 1; i >= 0; i--) {
        // Move current root to end
        swap(arr[0], arr[i]);

        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

int main() {
    int sizes[] = {1000, 5000, 10000, 15000, 25000, 50000, 75000, 100000, 150000, 250000};
    auto filename = "4-heap.csv";
    ofstream f(filename, ios::out);

    for (int i = 0; i < 10; i++) {
        vector<double> UnsortedArray = generateUnSortedArray(sizes[i]);
        vector<double> SortedArray = generateSortedArray(sizes[i]);
        vector<double> randomArray = generateRandomArray(sizes[i], -10000, 10000);

        int size = sizes[i];
        auto start = std::chrono::high_resolution_clock::now();

        //combSort(randomArray);
        //my_qsort(randomArray, 0, size-1);
        heapSort(randomArray);

        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        auto start1 = std::chrono::high_resolution_clock::now();

        //combSort(SortedArray);
        //my_qsort(SortedArray, 0, size-1);
        heapSort(SortedArray);

        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;

        auto start2 = std::chrono::high_resolution_clock::now();

        //combSort(UnsortedArray);
        //my_qsort(UnsortedArray, 0, size-1);
        heapSort(UnsortedArray);

        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end2 - start2;

        std::cout << size << " " << nsec.count() << " нсек." << std::endl;
        f << size << ";" << nsec.count() << ";" << nsec1.count() << ";" << nsec2.count() << ";nsec." << endl;

        }
}

