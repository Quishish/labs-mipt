#include <iostream>

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
    FloatUnsigned fu;

    std::cin >> fu.Float;
    printBinary(fu.Unsigned);

    return 0;
}

