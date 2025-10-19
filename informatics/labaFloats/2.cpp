#include <iostream>
#include <cmath>

void printBinary(unsigned value) {
    auto mask = 1u << 31;

    for (auto i = 1u; i <= 32u; i++) {
        std::cout << ((value & mask) != 0);
        mask = mask >> 1;
        if (i % 4 == 0) std::cout << ' ';
        if (i % 8 == 0) std::cout << ' ';
    }

    std::cout << std::endl;
}

union FloatUnsigned {
    unsigned Unsigned;
    float Float;
};


int main() {
    std::cout << std::fixed;
    std::cout.precision(2);

    float base = 10.00;

    for (int i = 1;i < 20;i++) {
        float result = pow(base, i);
        std::cout << result << " " << i << "\n" << std::endl;
    }

    return 0;
}

