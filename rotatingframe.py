"""
Code for performing Trojan asteroid simulations in the rotating frame

Vectors are represented by NumPy arrays.

scipy.integrate.solve_ivp is used to integrate the equations of motion
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import G, M_JUPITER, M_SUN, R_0


class RotatingAsteroid:
    """Simulation of a Trojan asteroid in the rotating frame

    The class should be initialised with the desired planetary mass
    and Sun-planet distance R in solar system units.

    The properties default to those of Jupiter
    """

    def __init__(self, M_P=M_JUPITER, R=5.2):
        """
        Simulation of a Trojan asteroid in the rotating frame.
        M_P = planetary mass
        R = Sun-planet distance
        """
        self._M_P = M_P
        self._R = R

        self._R_SUN = R * self._M_P / (self._M_P + M_SUN)
        self._R_P = R * M_SUN / (self._M_P + M_SUN)
        self._r_sun = np.array([-self._R_SUN, 0, 0])
        self._r_p = np.array([self._R_P, 0, 0])

        self._W = np.sqrt(G * (self._M_P + M_SUN) / R ** 3)
        self._T = 2 * np.pi / self._W

        self._L4 = np.array([self._R / 2 - self._R_SUN, self._R * np.sqrt(3) / 2, 0])
        self._L5 = np.array([self._R / 2 - self._R_SUN, -self._R * np.sqrt(3) / 2, 0])

    ### Getter methods
    @property
    def L4(self):
        """L4"""
        return self._L4

    @property
    def L5(self):
        """L5"""
        return self._L5

    @property
    def M_P(self):
        """Mass of Planet"""
        return self._M_P

    @property
    def R(self):
        """Distance between Sun and planet"""
        return self._R

    @property
    def R_SUN(self):
        """Distance from origin to Sun"""
        return self._R_SUN

    @property
    def R_P(self):
        """Distance from origin to Jupiter"""
        return self._R_P

    @property
    def T(self):
        """Time period of planetary orbit"""
        return self._T

    @property
    def W(self):
        """Angular frequency of planetary orbit"""
        return self._W

    def _acceleration(self, t, r, v):
        """Acceleration of an asteroid at position r with velocity v at time t"""
        real_accel = -G * (
            M_SUN * (r - self._r_sun) / np.linalg.norm(r - self._r_sun) ** 3
            + self._M_P * (r - self._r_p) / np.linalg.norm(r - self._r_p) ** 3
        )

        # explicit form for virtual acceleration with rotation in the z direction
        virtual_accel = self._W * np.array(
            [2 * v[1] + self._W * r[0], -2 * v[0] + self._W * r[1], 0]
        )
        return real_accel + virtual_accel

    def _derivs(self, t, y):
        """derivatives for solver"""
        return np.hstack((y[3:6], self._acceleration(t, y[0:3], y[3:6])))

    def trajectory(self, t_eval, r_0, v_0, events=None, method="Radau"):
        """Trajectory of asteroid calculated using solve_ivp"""
        y0 = np.append(r_0, v_0)

        return solve_ivp(
            self._derivs,
            (0, t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=method,
            events=events,
        )

    def wander(self, t_eval, r_0, v_0, stability_point, method="Radau"):
        """Find the maximum distance from the stability point for given initial conditions in the rotating frame"""
        sol = self.trajectory(t_eval, r_0, v_0, method=method)
        rs = sol.y[0:3]  # extract positions from solution

        deltas = (rs.T - stability_point).T  # Displacements from L4
        # Transpose used to stick to array convention
        norms = np.linalg.norm(deltas, axis=0)
        return norms.max()
