from simulator import *
from simulator_plotter import *
from units import *

DISTANCE_BETWEEN_PLANES = 600 * micrometers()
HOW_MANY_PLANES = 8
HOW_MANY_STRIPS_PER_PLANE = 128
PLANE_WIDTH = 24 * milimeters()
PLANE_HEIGHT = 18 * milimeters()


def get_rotation(i):
    if i % 2 == 0:
        return math.pi / 4
    else:
        return -1 * math.pi / 4


romanPot = RomanPot(name="testingRomanPot")
for i in range(HOW_MANY_PLANES):
    plane = RomanPotPlane(
        z=i * DISTANCE_BETWEEN_PLANES,
        gamma=get_rotation(i) * radians(),
        center_translation_x=0 * micrometers(),
        center_translation_y=0 * micrometers(),
        how_many_strips=HOW_MANY_STRIPS_PER_PLANE,
        width=PLANE_WIDTH,
        height=PLANE_HEIGHT,
    )
    romanPot.addPlane(plane)

hits = [Hit(romanPot.planes[i], 60 + i*2) for i in range(HOW_MANY_PLANES)]
track = Track(hits)

coeffs = track.solve(hll=False)
a1 = coeffs[0]
a2 = coeffs[1]
a3 = coeffs[2]
a4 = coeffs[3]
print(f"a1: {a1} a2: {a2} a3: {a3} a4: {a4}")

plot_simulation([a1, a2, a3, a4], hits)




simulatedTrack = Track.reverseSolveFromCoefficients(romanPot, a1, a2, a3, a4)
simulatedCoeffs = simulatedTrack.solve(hll=False, quantum=False)
sim_a1 = simulatedCoeffs[0]
sim_a2 = simulatedCoeffs[1]
sim_a3 = simulatedCoeffs[2]
sim_a4 = simulatedCoeffs[3]
print(f"simulated: \t\ta1: {sim_a1} a2: {sim_a2} a3: {sim_a3} a4: {sim_a4}")

plot_simulation([sim_a1, sim_a2, sim_a3, sim_a4], simulatedTrack.hits)

plot_error(coeffs, simulatedCoeffs, romanPot.planes[0].z, romanPot.planes[-1].z)