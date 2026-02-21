#pragma once
#include <cstddef>

namespace optim {

struct Vec2 {
    double x;
    double y;
};

struct OptimizerResult {
    Vec2 final_point;
    std::size_t iterations;
    double final_grad_norm;
};

}