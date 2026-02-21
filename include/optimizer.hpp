#pragma once

#include <cstddef>
#include <cmath>
#include <chrono>
#include "types.hpp"

namespace optim {

// Vanilla gradient Descent
template <typename Gradient>
OptimizerResult gradient_descent(
    const Vec2& initial,
    double alpha,
    std::size_t max_iters,
    double tol,
    Gradient&& gradient)
{
    Vec2 current = initial;

    std::size_t iter = 0;
    double grad_norm_sq = 0.0;

    for (; iter < max_iters; ++iter)
    {
        Vec2 grad = gradient(current);

        grad_norm_sq = grad.x * grad.x + grad.y * grad.y;
        if (grad_norm_sq < tol * tol)
            break;

        current.x -= alpha * grad.x;
        current.y -= alpha * grad.y;
    }

    return OptimizerResult{
        current,
        iter,
        std::sqrt(grad_norm_sq)
    };
}

// Gradient Descent with Momentum
template <typename Gradient>
OptimizerResult gradient_descent_momentum(
    const Vec2& initial,
    double alpha,
    double beta,
    std::size_t max_iters,
    double tol,
    Gradient&& gradient)
{
    Vec2 current = initial;
    Vec2 velocity{0.0, 0.0};

    std::size_t iter = 0;
    double grad_norm_sq = 0.0;

    for (; iter < max_iters; ++iter)
    {
        Vec2 grad = gradient(current);

        grad_norm_sq = grad.x * grad.x + grad.y * grad.y;
        if (grad_norm_sq < tol * tol)
            break;

        velocity.x = beta * velocity.x - alpha * grad.x;
        velocity.y = beta * velocity.y - alpha * grad.y;

        current.x += velocity.x;
        current.y += velocity.y;
    }

    return OptimizerResult{
        current,
        iter,
        std::sqrt(grad_norm_sq)
    };
}

// Nesterov Momentum
template <typename Gradient>
OptimizerResult nesterov_momentum(
    const Vec2& initial,
    double alpha,
    double beta,
    std::size_t max_iters,
    double tol,
    Gradient&& gradient)
{
    Vec2 current = initial;
    Vec2 velocity{0.0, 0.0};

    std::size_t iter = 0;
    double grad_norm_sq = 0.0;

    for (; iter < max_iters; ++iter)
    {
        Vec2 lookahead{
            current.x + beta * velocity.x,
            current.y + beta * velocity.y
        };

        Vec2 grad = gradient(lookahead);

        grad_norm_sq = grad.x * grad.x + grad.y * grad.y;
        if (grad_norm_sq < tol * tol)
            break;

        velocity.x = beta * velocity.x - alpha * grad.x;
        velocity.y = beta * velocity.y - alpha * grad.y;

        current.x += velocity.x;
        current.y += velocity.y;
    }

    return OptimizerResult{
        current,
        iter,
        std::sqrt(grad_norm_sq)
    };
}

}