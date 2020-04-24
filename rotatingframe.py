import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import G, M_SUN


class asteroid:
    def __init__(self, M_J=0.001, R=5.2):
        self._M_J = M_J
        self._R = R

        self._R_SUN = R * self._M_J / (self._M_J + M_SUN)
        self._R_J = R * M_SUN / (self._M_J + M_SUN)
        self._r_sun = np.array([-self._R_SUN, 0, 0])
        self._r_j = np.array([self._R_J, 0, 0])

        self._W = np.sqrt(G * (self._M_J + M_SUN) / R ** 3)
        self._T = 2 * np.pi / self._W

        self._L4 = np.array([self._R / 2 - self._R_SUN, self._R * np.sqrt(3) / 2, 0])
        self._L5 = np.array([self._R / 2 - self._R_SUN, -self._R * np.sqrt(3) / 2, 0])

    @property
    def M_J(self):
        """Get mass of Jupiter"""
        return self._M_J

    @property
    def R(self):
        """Get distance between Sun and Jupiter"""
        return self._R

    @property
    def R_SUN(self):
        """Get distance from origin to Sun"""
        return self._R_SUN

    @property
    def R_J(self):
        """Get distance from origin to Jupiter"""
        return self._R_J

    @property
    def W(self):
        """Get angular frequency"""
        return self._W

    @property
    def T(self):
        """Get time period"""
        return self._T

    @property
    def L4(self):
        """Get L4"""
        return self._L4

    @property
    def L5(self):
        """Get L5"""
        return self._L5

    def trajectory(self, t_eval, r_0, v_0):
        """Trajectory of asteroid calculated using solve_ivp"""
        y0 = np.append(r_0, v_0)

        def _acceleration(t, r, v):
            """Acceleration of an asteroid at position r with velocity v at time t"""
            return -G * (
                M_SUN * (r - self._r_sun) / np.linalg.norm(r - self._r_sun) ** 3
                + self._M_J * (r - self._r_j) / np.linalg.norm(r - self._r_j) ** 3
            ) + self._W * np.array(
                [2 * v[1] + self._W * r[0], -2 * v[0] + self._W * r[1], 0]
            )  # added coriolis and centrifugal forces

        def _derivs(t, y):
            """derivatives for solver"""
            return np.hstack((y[3:6], _acceleration(t, y[0:3], y[3:6])))

        return solve_ivp(_derivs, (0, t_eval[-1]), y0, t_eval=t_eval, method="LSODA")

    def max_wander(self, t_eval, r_0, v_0, stability_point):
        """Find the maximum distance from the starting point for given initial conditions in the rotating frame"""
        sol = self.trajectory(t_eval, r_0, v_0)
        rs = sol.y[0:3]  # extract positions from solution
        deltas = (
            rs.T - stability_point
        ).T  # Transpose used to stick to array convention
        norms = np.linalg.norm(deltas, axis=0)
        return norms.max()
