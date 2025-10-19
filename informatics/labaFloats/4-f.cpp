#include <iostream>
#include <cmath>

float f(float x) {
    return x*x;
}

float integ(float n, float a , float b) {
    float h = (b - a)/n;
    float integral = 0.0;

    for (int i = 0; i < n; i++) {
        float x_i = a + i * h;
        float area = f(x_i) * h;
        integral += area;
    }

    return integral;

}

int main() {
    float n; float a; float b;
    std::cin >> n >> a >> b;

    float result = integ(n,a,b);

    std::cout << result << std::endl;

    return 0;
}