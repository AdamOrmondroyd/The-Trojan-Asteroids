"""
Comparing the solving methods DOP853, Radau, BDF and LSODA included in 
scipy.integrate.solve_ivp

For each solver, the wander from L4 was found in both the stationary
and rotating frames.
"""
import numpy as np
from stationaryframe import StationaryAsteroid
from rotatingframe import RotatingAsteroid

# Instantiate asteroids for both frames
ast0 = StationaryAsteroid()
ast1 = RotatingAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 100 * ast0.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

methods = ["DOP853", "Radau", "BDF", "LSODA"]
wanders = np.zeros((2, len(methods)))

for i in range(len(methods)):
    wanders[0, i] = ast0.wander(
        ts, r_0=ast0.l4(0), v_0=ast0.omega_cross(ast0.l4(0)), method=methods[i]
    )

    wanders[1, i] = ast1.wander(
        ts,
        r_0=ast1.L4,
        v_0=np.array([0, 0, 0]),
        stability_point=ast1.L4,
        method=methods[i],
    )

    print(
        "{}: stationary frame: {} au, rotating frame: {}".format(
            methods[i], wanders[0, i], wanders[1, i]
        )
    )
