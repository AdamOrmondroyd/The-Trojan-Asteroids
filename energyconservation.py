import numpy as np
import matplotlib.pyplot as plt
from stationaryframe import asteroid
import time

ast = asteroid()

end_time = 1000
points_per_year = 10
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = ast.trajectory(t_eval=ts, r_0=ast.l4(0), v_0=ast.omega_cross(ast.l4(0)))
# sol = asteroid(t_eval=ts, r_0=L4, v_0=np.array([0, 0, 0]))

energies = ast.specific_energy(ts, sol.y[0:3], sol.y[3:6])

print("Mean specific energy: " + str(np.mean(energies)) + "(au/year)²")
print("Range: " + str(np.ptp(energies)) + "(au/year)²")
print("Range/mean: " + str(np.abs(np.ptp(energies) / np.mean(energies))))

plt.plot(ts, energies, label="energy", marker="+", linestyle="None")
plt.xlabel("time/years")
plt.ylabel("energy/(au/year)²")

plt.legend()
plt.show()
