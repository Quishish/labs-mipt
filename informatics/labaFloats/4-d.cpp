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
    double n; double a; double b;
    std::cin >> n >> a >> b;

    double result = integ(n,a,b);

    std::cout << result << std::endl;

    return 0;
}