import numpy as np
from scipy.integrate import solve_ivp
from params import m, c, k1, k2, k3, A, t_span, t, init_cond, dt
import matplotlib.pyplot as plt


def duffing_equation(t, y, m, c, k1, k2, k3, A):
    """
    Duffing's equation:
        m*y'' + c*y' + k1*y + k2*y^2 + k3*y^3 = amp*dirac_del

    """
    x, vx = y[0], y[1]

    dx_dt  = vx
    
    if t <= dt:
        imp = 1/dt
    else:
        imp = 0
    
    dvx_dt = (A*imp - c*vx - k1*x - k2*x**2 - k3*x**3) / m

    return np.array([dx_dt, dvx_dt])


# Solve Duffing's equation
sol = solve_ivp(
    duffing_equation, t_span, init_cond, method='RK45', t_eval=t,
    args=(m, c, k1, k2, k3, A)
)

y = sol.y[0]


if __name__ == "__main__":
    plt.plot(t, y, label="Runge")
    plt.title(
        f"Runge Kutta with Dirac delta amplitude: {A}"
    )
