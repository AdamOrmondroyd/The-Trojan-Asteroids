"""
Testing the conservation of energy using the Radau method for random 
perturbations about L4
"""
import numpy as np
import matplotlib.pyplot as plt
from stationaryframe import StationaryAsteroid
import time

ast = StationaryAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 100 * ast.T
points_per_year = 1
ts = np.linspace(0, end_time, int(end_time * points_per_year))

position_spread = 0.1
velocity_spread = 0.1
r_offset = (np.random.rand(3) - 0.5) * position_spread
v_offset = (np.random.rand(3) - 0.5) * velocity_spread
r_0 = ast.l4(0) + r_offset
v_0 = ast.omega_cross(r_0) + v_offset
print("Initial perturbation about L4")
print("r offset = " + str(r_offset))
print("v offset = " + str(v_offset))

sol = ast.trajectory(ts, r_0, v_0)

energies = ast.specific_energy(ts, sol.y[0:3], sol.y[3:6])

# Print findings
print("Mean specific energy: " + str(np.mean(energies)) + "(au/year)²")
print("Range: " + str(np.ptp(energies)) + "(au/year)²")
print("Range/mean: " + str(np.abs(np.ptp(energies) / np.mean(energies))))

### Plotting ###

fig, ax = plt.subplots()

ax.plot(ts, energies, label="energy", color="c", marker="+", linestyle="None")
ax.set(
    title="Variation of energy",
    xlabel="time/years",
    ylabel="specific energy/(au/year)²",
)

# filename = "plots\\energy_variation"
# plt.savefig(filename + ".png")
# plt.savefig(filename + ".eps")
plt.show()
