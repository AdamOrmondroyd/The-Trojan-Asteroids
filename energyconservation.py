import numpy as np
import matplotlib.pyplot as plt
from stationaryframe import asteroid, omega_cross, specific_energy, r_sun, r_j
from constants import L4, L5, R, R_SUN, R_J
import time

end_time = 1000
points_per_year = 10
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = asteroid(t_eval=ts, r_0=L4, v_0=omega_cross(L4))
# sol = asteroid(t_eval=ts, r_0=L4, v_0=np.array([0, 0, 0]))

energies = specific_energy(ts, sol.y[0:3], sol.y[3:6])

print("Mean specific energy: " + str(np.mean(energies)) + "(au/year)²")
print("Range: " + str(np.ptp(energies)) + "(au/year)²")
print("Range/mean: " + str(np.abs(np.ptp(energies) / np.mean(energies))))

plt.plot(ts, energies, label="energy", marker="+", linestyle="None")
plt.xlabel("time/years")
plt.ylabel("energy/(au/year)²")

plt.legend()
plt.show()
