import numpy as np
import matplotlib.pyplot as plt
import rotatingframe
import constants
from rotatingframe import asteroid, max_wander
from constants import L4, T

end_time = 100 * T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

print(max_wander(end_time, ts, r_0=L4, v_0=np.array([0, 0, 0]), stability_point=L4,))

m_min = 0.0157
m_max = 0.0166
points = 100
ms = np.linspace(m_min, m_max, points)

wanders = np.zeros(points)

for i in range(points):
    print(i)
    constants.M_J = ms[i]
    rotatingframe.M_J = ms[i]
    wanders[i] = max_wander(
        end_time, ts, r_0=L4, v_0=np.array([0, 0, 0]), stability_point=L4,
    )

plt.plot(ms, wanders)
plt.show()
