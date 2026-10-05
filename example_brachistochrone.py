"""
Brachistochrone example problem
"""

import math
import casadi
from casadi_mocp import MultiPhaseOptimalControlProblem

GRAVITY = 9.80665
X_END = 2.0
Y_END = 2.0

def solve_brachistochrone(n_intervals=24):
    mocp = MultiPhaseOptimalControlProblem()

    duration = mocp.create_phase('descent', init=1.0, n_intervals=n_intervals)
    x = mocp.add_trajectory('descent', 'x', init=0.5 * X_END)
    y = mocp.add_trajectory('descent', 'y', init=0.5 * Y_END)
    v = mocp.add_trajectory('descent', 'v', init=3.0)
    theta = mocp.add_trajectory('descent', 'theta', init=0.5 * math.pi)

    mocp.set_derivative(x, v * casadi.sin(theta))
    mocp.set_derivative(y, v * casadi.cos(theta))
    mocp.set_derivative(v, GRAVITY * casadi.cos(theta))

    mocp.add_path_constraint(theta > 0.0)
    mocp.add_path_constraint(theta < math.pi)

    mocp.add_constraint(mocp.start(x) == 0.0)
    mocp.add_constraint(mocp.start(y) == 0.0)
    mocp.add_constraint(mocp.start(v) == 0.0)
    mocp.add_constraint(mocp.end(x) == X_END)
    mocp.add_constraint(mocp.end(y) == Y_END)
    mocp.add_constraint(duration > 0.01)

    mocp.add_objective(duration)

    mocp.solve()

    return mocp, duration

def analytic_optimum():
    def residual(u):
        return (u - math.sin(u)) / (1.0 - math.cos(u)) - X_END / Y_END
    lo, hi = 1e-6, 2.0 * math.pi - 1e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    u_end = 0.5 * (lo + hi)
    a = Y_END / (1.0 - math.cos(u_end))
    return math.sqrt(a / GRAVITY) * u_end

def main():
    mocp, duration = solve_brachistochrone()
    optimal_duration = mocp.get_value(duration)
    reference = analytic_optimum()

    print()
    print('descent time:             {:.6f} s'.format(optimal_duration))
    print('analytic cycloid optimum: {:.6f} s'.format(reference))
    print('relative error:           {:.2e}'.format(abs(optimal_duration - reference) / reference))

if __name__ == '__main__':
    main()