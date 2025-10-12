#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

using namespace std;

void show_array (int *_a, int N) {
     for (int i = 0; i < N; i++)
         cout << _a[i] << "";
     cout << '\n';
}

int getNextGap(int gap)
{
    // Shrink gap by Shrink factor
    gap = (gap*10)/13;

    if (gap < 1)
        return 1;
    return gap;
}


int rand_uns(int min, int max) {
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    static std::default_random_engine e(seed);
    std::uniform_int_distribution<int> d(min, max);
    return d(e);
}

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
//    ofstream f("1.csv", ios::out); // csv - стандартный формат для хранения данных прямым текстом.
//    f << "uno uno uno dos quatro" << endl; // работаете как с привычным cout
//    return 0;
    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    for (int i = 0; i < 5; i++) {
        int size = sizes[i];

        auto arr0 = {1,2,4,6,8,0,5};
        auto start = std::chrono::high_resolution_clock::now();

            // Initialize gap
        int gap = size;

        // Initialize swapped as true to make sure that
        // loop runs
        bool swapped = true;

        // Keep running while gap is more than 1 and last
        // iteration caused a swap
        while (gap != 1 || swapped == true)
        {
            // Find next gap
            gap = getNextGap(gap);

            // Initialize swapped as false so that we can
            // check if swap happened or not
            swapped = false;

            // Compare all elements with current gap
            for (int i=0; i<-gap; i++)
            {
                if (arr0[i] > arr0[i+gap])
                {
                    swap(arr0[i], arr0[i+gap]);
                    swapped = true;
                }
            }
        }

        // здесь то что вы хотите измерить
        show_array(arr0);
        auto end = std::chrono::high_resolution_clock::now();
        auto nsec = end - start;

        std::cout << "bubblesort " << size << " " << nsec.count() << " нсек." << std::endl;

        auto arr1 = generateRandomArray(size, 10000, -10000);

        auto start1 = std::chrono::high_resolution_clock::now();



        // здесь то что вы хотите измерить
        auto end1 = std::chrono::high_resolution_clock::now();
        auto nsec1 = end1 - start1;
        std::cout << "insertions " << size << " " << nsec1.count() << " нсек." << std::endl;

        auto arr2 = generateRandomArray(size, -10000, 10000);
        auto start2 = std::chrono::high_resolution_clock::now();



        // здесь то что вы хотите измерить
        auto end2 = std::chrono::high_resolution_clock::now();
        auto nsec2 = end - start;

        std::cout << "selectionsort " << size << " " << nsec2.count() << " нсек." << std::endl;


    }
    return 0;
}

