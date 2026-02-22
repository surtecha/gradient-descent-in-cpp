#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include "../include/optimizer.hpp"

using namespace optim;

Vec2 grad_quadratic(Vec2 p)  { 
    return {2*p.x, 2*p.y}; 
}
Vec2 grad_rosenbrock(Vec2 p) { 
    return {-2*(1-p.x) - 400*p.x*(p.y-p.x*p.x), 200*(p.y-p.x*p.x)}; 
}
Vec2 grad_himmelblau(Vec2 p) { 
    return {4*p.x*(p.x*p.x+p.y-11)+2*(p.x+p.y*p.y-7), 2*(p.x*p.x+p.y-11)+4*p.y*(p.x+p.y*p.y-7)}; 
}
Vec2 grad_beale(Vec2 p) {
    return {
        -2*(1.5-p.x+p.x*p.y)*p.y - 2*(2.25-p.x+p.x*p.y*p.y)*p.y*p.y - 2*(2.625-p.x+p.x*p.y*p.y*p.y)*p.y*p.y*p.y,
         2*(1.5-p.x+p.x*p.y)*p.x + 4*(2.25-p.x+p.x*p.y*p.y)*p.x*p.y + 6*(2.625-p.x+p.x*p.y*p.y*p.y)*p.x*p.y*p.y
    };
}

struct FnDef {
    const char* name;
    Vec2 (*grad)(Vec2);
    Vec2   default_start;
    double default_alpha;
};

const FnDef fns[] = {
    {"quadratic",   grad_quadratic,   {2.0,  2.0},  1e-2},
    {"rosenbrock",  grad_rosenbrock,  {-1.0, 1.0},  1e-4},
    {"himmelblau",  grad_himmelblau,  {0.0,  0.0},  1e-4},
    {"beale",       grad_beale,       {1.0,  1.0},  1e-5},
};

double prompt_double(const std::string& msg, double def) {
    std::cout << msg << " [default: " << def << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return def; return std::stod(line);
}

void record(std::ofstream& traj, std::ofstream& bench,
            const std::string& opt, const std::string& fn,
            const std::vector<Vec2>& path, const OptimizerResult& res, double ms,
            std::size_t max_iters, double tol)
{
    for (std::size_t i = 0; i < path.size(); ++i)
        traj << opt << "," << fn << "," << i << "," << path[i].x << "," << path[i].y << "\n";
    bench << opt << "," << fn << "," << res.iterations << "," << res.final_grad_norm
          << "," << res.final_point.x << "," << res.final_point.y << "," << ms << "\n";

    std::string stopping = (res.iterations >= max_iters) ? "max iterations reached"
                         : (res.final_grad_norm < tol)   ? "tolerance reached"
                                                         : "convergence";
    std::cout << "\n  [ " << opt << " ]\n";
    std::cout << "    stopped due to     : " << stopping << "\n";
    std::cout << "    iterations     : " << res.iterations << "\n";
    std::cout << "    final point    : (" << res.final_point.x << ", " << res.final_point.y << ")\n";
    std::cout << "    gradient norm  : " << res.final_grad_norm << "\n";
    std::cout << "    time elapsed   : " << ms << " ms\n";
}

int main() {
    std::cout << "available functions:\n";
    for (int i = 0; i < 4; ++i) 
        std::cout << "  " << (i+1) << ": " << fns[i].name << "\n";

    int choice;
    std::cout << "function: "; std::cin >> choice;
    if (choice < 1 || choice > 4) { 
        std::cerr << "invalid choice\n"; 
        return 1; 
    }

    std::size_t max_iters;
    std::cout << "max_iters: "; std::cin >> max_iters;

    int tol_exp;
    std::cout << "tolerance exponent (e.g. 5 means 1e-5) [default: 5]: ";
    std::string tol_line;
    std::cin.ignore();
    std::getline(std::cin, tol_line);
    double tol = tol_line.empty() ? 1e-5 : std::pow(10.0, -std::stod(tol_line));

    const FnDef& f = fns[choice-1];
    std::cout << "\nstarting point (the initial guess where all optimizers begin):\n";
    double sx    = prompt_double("  start x", f.default_start.x);
    double sy    = prompt_double("  start y", f.default_start.y);

    std::cout << "\nhyperparameters:\n";
    double alpha = prompt_double("  alpha (step size, how far to move each iteration)", f.default_alpha);
    double beta  = prompt_double("  beta  (momentum factor, how much to carry previous velocity)", 0.9);

    Vec2 init{sx, sy};
    std::string fn = f.name;
    auto gfn = f.grad;

    std::cout << "\nrunning all three optimizers on '" << fn << "' "
              << "from (" << sx << ", " << sy << ") "
              << "for up to " << max_iters << " iterations...\n";
    std::cout << "tolerance: gradient norm must fall below " << tol << " to stop early.\n";

    std::ofstream traj("trajectory.csv");
    traj << "optimizer,function,iteration,x,y\n";
    std::ofstream bench("results.csv");
    bench << "optimizer,function,iterations,grad_norm,final_x,final_y,time_ms\n";

    {
        std::vector<Vec2> path;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = gradient_descent(init, alpha, max_iters, tol, [&](Vec2 p){ path.push_back(p); return gfn(p); });
        double ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        record(traj, bench, "vanilla gradient descent", fn, path, res, ms, max_iters, tol);
    }
    {
        std::vector<Vec2> path;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = gradient_descent_momentum(init, alpha, beta, max_iters, tol, [&](Vec2 p){ path.push_back(p); return gfn(p); });
        double ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        record(traj, bench, "gradient descent + momentum", fn, path, res, ms, max_iters, tol);
    }
    {
        std::vector<Vec2> path;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = nesterov_momentum(init, alpha, beta, max_iters, tol, [&](Vec2 p){ path.push_back(p); return gfn(p); });
        double ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        record(traj, bench, "nesterov momentum", fn, path, res, ms, max_iters, tol);
    }

    std::cout << "\ntrajectory written to trajectory.csv\n";
    std::cout << "benchmark summary written to results.csv\n";

    return 0;
}
