import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid
from constants import R, R_SUN, R_J

end_time = 100
points_per_year = 1000
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = asteroid(run_time=end_time, t_eval=ts)

plt.plot(
    -R_SUN, 0, label="sun", color="orange", marker="*", markersize=20, linestyle="None"
)
plt.plot(
    R_J, 0, label="Jupiter", color="red", marker="o", markersize=10, linestyle="None"
)
plt.plot(sol.y[0], sol.y[1], "-", label="Greeks", color="green")
plt.axis("scaled")
plt.legend()
plt.show()
