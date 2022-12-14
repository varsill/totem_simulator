import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PRECISION = 1000
STRIPS_PLOTTING_STEP = 20

def plot_simulation(coefficients, hits):

  a1 = coefficients[0]
  a2 = coefficients[1]
  a3 = coefficients[2]
  a4 = coefficients[3]

  first_hit = hits[0]
  last_hit = hits[-1]

  MIN_Z = first_hit.plane.z-(last_hit.plane.z-first_hit.plane.z)/2 
  MAX_Z = last_hit.plane.z+(last_hit.plane.z-first_hit.plane.z)/2 

  ax = plt.axes(projection='3d')

  # BEAMLINE
  zbeamline = np.linspace(MIN_Z-(MAX_Z-MIN_Z)/2, MAX_Z+(MAX_Z-MIN_Z)/2, PRECISION)
  xbeamline = PRECISION*[0]
  ybeamline = PRECISION*[0]
  ax.plot3D(xbeamline, ybeamline, zbeamline, 'black', alpha=0.2)

  # FOUND PARTICLE TRACK
  zline = np.linspace(MIN_Z, MAX_Z, PRECISION)
  xline = a1+a3*zline
  yline = a2+a4*zline
  ax.plot3D(xline, yline, zline, 'red')

  # HIT POINTS
  xhits = []
  yhits = []
  zhits = []
  for hit in hits:
      xhits.append(hit.global_x)
      yhits.append(hit.global_y)
      zhits.append(hit.global_z)
  ax.scatter3D(xhits, yhits, zhits, c="red")


  for hit in hits:
    plane = hit.plane
    xstart = plane.lower_left_x + hit.ordering_number_of_strip * plane.strip_width * plane.u[0]
    xend = xstart+plane.height*plane.v[0]
    ystart = plane.lower_left_y + hit.ordering_number_of_strip * plane.strip_width * plane.u[1]
    yend = ystart+plane.height*plane.v[1]
    xline = np.linspace(xstart, xend, PRECISION)
    yline = np.linspace(ystart, yend, PRECISION)
    zline = PRECISION*[plane.z]
    ax.plot3D(xline, yline, zline, 'red', alpha=1)

  # PLANES
  for hit in hits:
      plane = hit.plane
      xplane = np.array([plane.lower_left_x, plane.lower_left_x + plane.u[0] * plane.width, \
                         plane.lower_left_x + plane.u[0] * plane.width + plane.v[0] * plane.height, plane.lower_left_x + plane.v[0] * plane.height])
      yplane = np.array([plane.lower_left_y, plane.lower_left_y + plane.u[1] * plane.width, plane.lower_left_y + plane.u[1] * plane.width + plane.v[1] * plane.height \
        , plane.lower_left_y + plane.v[1] * plane.height])
      zplane = np.array([plane.z, plane.z, plane.z, plane.z])
      vertices = [list(zip(xplane, yplane, zplane))]
      poly = Poly3DCollection(vertices, alpha=0.15, color='g')
      ax.add_collection3d(poly)

      for strip_no in range(0, plane.how_many_strips, STRIPS_PLOTTING_STEP):
        xstart = plane.lower_left_x + strip_no * plane.strip_width * plane.u[0]
        xend = xstart+plane.height*plane.v[0]
        ystart = plane.lower_left_y + strip_no * plane.strip_width * plane.u[1]
        yend = ystart+plane.height*plane.v[1]
        xline = np.linspace(xstart, xend, PRECISION)
        yline = np.linspace(ystart, yend, PRECISION)
        zline = PRECISION*[plane.z]
        ax.plot3D(xline, yline, zline, 'blue', alpha=0.7)



      
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  ax.set_xlim(min(xplane), max(xplane))
  ax.set_ylim(min(yplane), max(yplane))
  ax.set_zlim(MIN_Z, MAX_Z)
  plt.show()