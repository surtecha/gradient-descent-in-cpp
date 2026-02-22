# Implementing and Benchmarking Gradient Descent Algorithms in C++

![trajectories_himmelblau](https://github.com/user-attachments/assets/64ddea08-8b61-4fb0-9381-004764d636c5)

Vanilla gradient descent, gradient descent with momentum, and Nesterov momentum implemented from scratch in C++20. Each optimizer runs on four classical 2D test functions, with trajectory and timing data exported to CSV and visualised in Python.

---

## Setup

```bash
git clone https://github.com/surtecha/gradient-descent-in-cpp.git
cd gradient-descent-in-cpp
```

## Build & Run

```bash
g++ -std=c++20 -O2 -o benchmark src/main.cpp
./benchmark
```

The program will prompt for:

| Input | Description | Example |
|-------|-------------|---------|
| Function | Choose from quadratic, rosenbrock, himmelblau, beale | `2` |
| Max iterations | Upper bound on steps | `10000` |
| Start x, y | Initial point | `−1`, `1` |
| Alpha | Step size (learning rate) | `1e-4` |
| Beta | Momentum factor, used by momentum and Nesterov | `0.9` |
| Tolerance exponent | Stops early if gradient norm drops below $10^{-n}$ | `5` for $10^{-5}$ |

All three optimizers run on the same inputs. Output is written to `trajectory.csv` and `results.csv`.

## Visualisation

```bash
python plot.py
```

Dependencies: `matplotlib`, `pandas`, `numpy`.

---

## Test Functions

**Quadratic**

Strictly convex, smooth, single global minimum at the origin. Gradient magnitude scales linearly with distance.

$$f(x, y) = x^2 + y^2$$

**Rosenbrock**

Non-convex. Global minimum at $(1, 1)$ inside a narrow curved valley. Gradients are large in one direction and nearly flat in the other.

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

**Himmelblau**

Non-convex with four equal global minima at $(3, 2)$, $(-2.805, 3.131)$, $(-3.779, -3.283)$, and $(3.584, -1.848)$.

$$f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$$

**Beale**

Non-convex with a flat landscape and sharp curvature near the minimum at $(3, 0.5)$. Sensitive to step size.

$$f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$$

---

## Optimizers

**Vanilla Gradient Descent**

$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

**Gradient Descent with Momentum**

$$v_{t+1} = \beta v_t - \alpha \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

**Nesterov Momentum**

$$v_{t+1} = \beta v_t - \alpha \nabla f(\theta_t + \beta v_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

---

## Results

### Quadratic

<img width="1512" height="885" alt="1" src="https://github.com/user-attachments/assets/6600f571-3518-481a-ac46-e0041cf1af6a" />
<img width="1000" height="500" alt="1-bench" src="https://github.com/user-attachments/assets/ca17d8d7-0d43-46de-abfc-2e93972374a9" />

### Rosenbrock

<img width="1512" height="885" alt="2" src="https://github.com/user-attachments/assets/cbd6af0c-9436-49cd-a5aa-abdc301ba791" />
<img width="1000" height="500" alt="2-bench" src="https://github.com/user-attachments/assets/49e11cc2-0c23-4ef6-bfa3-2b9c96c57b77" />

### Himmelblau

<img width="1512" height="885" alt="3" src="https://github.com/user-attachments/assets/c029f12f-8963-4737-85a3-dea48582ad79" />
<img width="1000" height="500" alt="3-bench" src="https://github.com/user-attachments/assets/bee2e7c8-8139-47c3-b39e-03efacd67fd3" />

### Beale

<img width="1512" height="885" alt="4" src="https://github.com/user-attachments/assets/e0fd861e-a50c-4757-ae65-3615f0dfd400" />
<img width="1000" height="500" alt="4-bench" src="https://github.com/user-attachments/assets/e700969a-9040-4919-8c6f-0c95c611c07c" />
