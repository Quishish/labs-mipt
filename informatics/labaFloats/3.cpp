#include <iostream>
#include <cmath>

int main() {
    std::cout << std::fixed;
    std::cout.precision(2);

    float base = 10.00;

    for (float i = 100000000-10; i < 100000000; i++) {
        std::cout << i << std::endl;
    }

    return 0;
}
