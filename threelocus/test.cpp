#include <iostream>
#include <cstdlib>

#include "cexp.h"

using namespace helpers;

int main(int argc, char** argv) {
    Params p;
    p.init = {0.3, 0.2, 0.01, 0.02, 0.2, 0.05, 0.05};
    p.n = 2000;
    p.h = 0.5;
    p.s = 0.10;
    p.rLR = 2e-6;
    p.rLS = 1e-6;
    p.rRS = 1e-6;
    p.rLS_R = 1e-6;
    p.rRS_L = 1e-6;
    p.rLR_S = 0.0;
    std::vector<int> times = {1, 2, 5, 10};
    // std::cout << E(p, 10, Marginal::L) << std::endl;
    std::cout << covLR(p, 5, 10) << std::endl;
    std::cout << covLR(p, 10, 15) << std::endl;
    // std::cout << covLS(p, 50, 100) << std::endl;
    // std::cout << covLL(p, 50, 100) << std::endl;
    // std::cout << covSS(p, 100, 100) << std::endl;
}
