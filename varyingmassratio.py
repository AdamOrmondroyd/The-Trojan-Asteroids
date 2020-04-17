import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid, max_wander
from constants import M_J

m_max = 0.1
points = 100
ms = np.linspace(0, m_max, points)
