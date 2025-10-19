#include <iostream>
#include <cmath>

double f(double x) {
    return x*x;
}

double integ(double n, double a , double b) {
    double h = (b - a)/n;
    double integral = 0.0;

    for (int i = 0; i < n; i++) {
        double x_i = a + i * h;
        double area = f(x_i) * h;
        integral += area;
    }

    return integral;

}

int main() {
    std::cout << std::fixed;
    std::cout.precision(3);
    double sizes[] = {10, 100, 500, 1000, 10000, 20000, 100000, 500000, 1000000, 10000000};
    double a = 1;
    double b = 6;

    for (int i = 0; i < 10; i++) {
        double n = sizes[i];

        double result = integ(n,a,b);

        std::cout << result << " " << n << std::endl;
    }

    return 0;
}