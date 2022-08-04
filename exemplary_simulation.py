from simulator import *
from simulator_plotter import *
from units import *

DISTANCE_BETWEEN_PLANES = 600 * micrometers()
HOW_MANY_PLANES = 10
HOW_MANY_STRIPS_PER_PLANE = 128
PLANE_WIDTH = 24 * milimeters()
PLANE_HEIGHT = 18 * milimeters()


def get_rotation(i):
    if i % 2 == 0:
        return math.pi / 4
    else:
        return -1 * math.pi / 4


planes = [RomanPotPlane(i * DISTANCE_BETWEEN_PLANES, get_rotation(i) * radians(),
                        0 * micrometers(), 0 * micrometers(), HOW_MANY_STRIPS_PER_PLANE, PLANE_WIDTH, PLANE_HEIGHT)
          for i in range(HOW_MANY_PLANES)]

hits = [Hit(planes[i], 60) for i in range(HOW_MANY_PLANES)]
track = Track(hits)

coeffs = track.solve(hll=False)
a1 = coeffs[0][0]
a2 = coeffs[1][0]
a3 = coeffs[2][0]
a4 = coeffs[3][0]
print(f"a1: {a1} a2: {a2} a3: {a3} a4: {a4}")

plot_simulation([a1, a2, a3, a4], hits)
