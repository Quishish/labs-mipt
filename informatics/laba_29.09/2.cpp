#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm> // для std::swap

using namespace std;

int rand_uns(int min, int max) {
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    static std::default_random_engine e(seed);
    std::uniform_int_distribution<int> d(min, max);
    return d(e);
}


// Функция для генерации массива случайных чисел
void generateRandomArray(int n, int lowerBound, int upperBound) {
    int arr[n];

    for (int i = 0; i < n; ++i) {
        arr[i] = rand_uns(lowerBound, upperBound);
    }
        
    return arr;
}

void CocktailSort(std::vector<double> a[], int n)
{
    bool swapped = true;
    int start = 0;
    int end = n - 1;

    while (swapped) {
        // reset the swapped flag on entering
        // the loop, because it might be true from
        // a previous iteration.
        swapped = false;

        // loop from left to right same as
        // the bubble sort
        for (int i = start; i < end; ++i) {
            if (a[i] > a[i + 1]) {
                swap(a[i], a[i + 1]);
                swapped = true;
            }
        }

        // if nothing moved, then array is sorted.
        if (!swapped)
            break;

        // otherwise, reset the swapped flag so that it
        // can be used in the next stage
        swapped = false;

        // move the end point back by one, because
        // item at the end is in its rightful spot
        --end;

        // from right to left, doing the
        // same comparison as in the previous stage
        for (int i = end - 1; i >= start; --i) {
            if (a[i] > a[i + 1]) {
                swap(a[i], a[i + 1]);
                swapped = true;
            }
        }

        // increase the starting point, because
        // the last stage would have moved the next
        // smallest number to its rightful spot.
        ++start;
    }
}



int main()
{
//    ofstream f("1.csv", ios::out); // csv - стандартный формат для хранения данных прямым текстом.
//    f << "uno uno uno dos quatro" << endl; // работаете как с привычным cout
//    return 0;
    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    std::cout << generateRandomArray(10, -10, 10);
    for (int i = 0; i < 5; i++) {
        int size = sizes[i];

        auto arr0 = generateRandomArray(size, 10000, -10000);
        auto start = std::chrono::high_resolution_clock::now();

        int n = arr0.size();  
        int gap = n;  
        const float shrinkFactor = 1.3;  
        bool sorted = false;  
        while (gap > 1 || !sorted) {  
            gap = std::max(1, int(gap / shrinkFactor));  
            sorted = true;  
            for (int i = 0; i < n - gap; ++i) {  
                if (arr0[i] > arr0[i + gap]) {  
                    std::swap(arr0[i], arr0[i + gap]);  
                    sorted = false;  
                }  
            }  
        }  

        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        std::cout << "combsort " << size << " " << nsec.count() << " нсек." << std::endl;

        auto arr1 = generateRandomArray(size, 10000, -10000);
        auto start1 = std::chrono::high_resolution_clock::now();

        CocktailSort(arr1, sizes[i]);

        // здесь то что вы хотите измерить
        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;
        std::cout << "quickSort " << size << " " << nsec1.count() << " нсек." << std::endl;

        auto arr2 = generateRandomArray(size, -10000, 10000);
        auto start2 = std::chrono::high_resolution_clock::now();

        

        // здесь то что вы хотите измерить
        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end - start;

        std::cout << "cocktailSort " << size << " " << nsec2.count() << " нсек." << std::endl;


    }
    return 0;
}

