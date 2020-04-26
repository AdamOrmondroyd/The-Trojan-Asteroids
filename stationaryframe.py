import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import G, M_JUPITER, M_SUN, R_0


class StationaryAsteroid:
    def __init__(self, M_P=M_JUPITER, R=R_0):
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

    @property
    def M_P(self):
        """Get mass of Jupiter"""
        return self._M_P

    @property
    def R(self):
        """Get distance between Sun and Jupiter"""
        return self._R

    @property
    def R_SUN(self):
        """Get distance from origin to Sun"""
        return self._R_SUN

    @property
    def R_P(self):
        """Get distance from origin to Jupiter"""
        return self._R_P

    @property
    def W(self):
        """Get angular frequency"""
        return self._W

    @property
    def T(self):
        """Get time period"""
        return self._T

    def l4(self, t):
        """Position of L_4 at time t"""
        return np.stack(
            (
                self._L4[0] * np.cos(self._W * t) - self._L4[1] * np.sin(self._W * t),
                self._L4[0] * np.sin(self._W * t) + self._L4[1] * np.cos(self._W * t),
                0.0 * t,
            )
        )

    def l5(self, t):
        """Position of L_4 at time t"""
        return np.stack(
            [
                self._L5[0] * np.cos(self._W * t) - self._L5[1] * np.sin(self._W * t),
                self._L5[0] * np.sin(self._W * t) + self._L5[1] * np.cos(self._W * t),
                0.0 * t,
            ]
        )

    def r_sun(self, t):
        """Position of the sun at time t"""
        return np.stack(
            [
                -self._R_SUN * np.cos(self._W * t),
                -self.R_SUN * np.sin(self._W * t),
                0.0 * t,
            ]
        )

    def r_p(self, t):
        """Position of Jupiter at time t"""
        return np.stack(
            [self._R_P * np.cos(self._W * t), self._R_P * np.sin(self._W * t), 0.0 * t]
        )

    def specific_energy(self, t, r, v):
        """Specific energy of an asteroid"""
        kinetic = 0.5 * np.linalg.norm(v, axis=0) ** 2
        potential = -G * (
            M_SUN / np.linalg.norm(r - self.r_sun(t), axis=0)
            + self._M_P / np.linalg.norm(r - self.r_p(t), axis=0)
        )
        return kinetic + potential

    def omega_cross(self, r):
        """Returns the result of W x r (for convenience)"""
        return np.array([-self.W * r[1], self.W * r[0], 0])

    def _acceleration(self, t, r):
        """Acceleration of an asteroid at position r and time t"""
        return -G * (
            M_SUN * (r - self.r_sun(t)) / np.linalg.norm(r - self.r_sun(t)) ** 3
            + self._M_P * (r - self.r_p(t)) / np.linalg.norm(r - self.r_p(t)) ** 3
        )

    def _derivs(self, t, y):
        """derivatives of asteroid for solver"""
        return np.hstack((y[3:6], self._acceleration(t, y[0:3])))

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

    def wander(self, t_eval, r_0, v_0, method="Radau"):
        """Find the maximum distance from L4 point for given initial conditions in the rotating frame"""
        sol = self.trajectory(t_eval, r_0, v_0, method=method)
        rs = sol.y[0:3]  # extract positions from solution
        deltas = rs - self.l4(t_eval)  # Transpose used to stick to array convention
        norms = np.linalg.norm(deltas, axis=0)
        return norms.max()
