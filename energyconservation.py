import numpy as np
import matplotlib.pyplot as plt
from stationaryframe import asteroid, omega_cross, specific_energy, r_sun, r_j
from constants import L4, L5, R, R_SUN, R_J
import time

print(specific_energy(0, L4, np.array([0, 0, 0])))

end_time = 1000
points_per_year = 10
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = asteroid(run_time=end_time, t_eval=ts, r_0=L4, v_0=omega_cross(L4))
# sol = asteroid(run_time=end_time, t_eval=ts, r_0=L4, v_0=np.array([0, 0, 0]))

sun_path = r_sun(ts)
jupiter_path = r_j(ts)
plt.plot(sun_path[0], sun_path[1], label="sun", color="orange")
plt.plot(jupiter_path[0], jupiter_path[1], label="Jupiter", color="red")
plt.plot(sol.y[0], sol.y[1], "-", label="Greeks", color="green")
plt.axis("scaled")
plt.legend()
plt.show()

energies, kit_energies = specific_energy(ts, sol.y[0:3], sol.y[3:6])

print("Mean specific energy: " + str(np.mean(energies)) + "(AU/year)²")
print("Range: " + str(np.ptp(energies)) + "(AU/year)²")
print("Range/mean: " + str(np.abs(np.ptp(energies) / np.mean(energies))))

print("Mean Kit energy: " + str(np.mean(kit_energies)) + "(AU/year)²")
print("Range: " + str(np.ptp(kit_energies)) + "(AU/year)²")
print("Range/mean: " + str(np.abs(np.ptp(kit_energies) / np.mean(kit_energies))))

plt.plot(ts, energies, label="energy")
plt.plot(ts, kit_energies, label="Kit energy")
plt.xlabel("time/years")
plt.ylabel("energy/(AU/year)²")

plt.legend()
plt.show()
