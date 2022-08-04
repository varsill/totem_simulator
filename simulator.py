import numpy as np
import math
from common import *


class RomanPotPlane:

    """
    params:
        gamma - tilt angle of the detector (in radians)
        x0 - the offset along the the telescope's  x-axis (how far was the detector moved to the right of the tube center)
        y0 - the offset along the the telescope's  y-axis (how far was the detector moved up from the tube center)
        z0 - the offset along the telescope's z-axis (how far inside of the tube is the detector placed)
        how_many_strips - number of strips per plane
        width - width of of the detector's plane, in the direction perpendicular to the strips
        height - height of the detector's plane, paralel to the strips
    """

    def __init__(self, z, gamma, x0, y0, how_many_strips, width, height):
        self.z = z
        self.gamma = gamma
        self.x0 = x0
        self.y0 = y0
        self.u = np.array([np.cos(gamma), np.sin(gamma)])
        # v is perpendicular to u
        self.v = np.array([np.cos(gamma - math.pi / 2),
                          np.sin(gamma - math.pi / 2)])
        self.how_many_strips = how_many_strips
        self.width = width
        self.height = height
        self.resolution = self.how_many_strips / self.width
        self.x_mid = x0 - self.u[0] * width / 2 - self.v[0] * height / 2
        self.y_mid = y0 - self.u[1] * width / 2 - self.v[1] * height / 2


class Hit:
    """
    params:
        plane - an object describing which roman pot plane was hit
        u_m - the offset along the plane's u-axis (describes which plane's line was hit), with respect to the plane
    """

    def __init__(self, plane, ordering_number_of_strip):
        self.plane = plane
        self.ordering_number_of_strip = ordering_number_of_strip
        self.u_m = ordering_number_of_strip / plane.resolution
        self.u_mg = self.u_m + plane.u @ np.array([plane.x_mid, plane.y_mid]).T

    def get_global_x_y_z(self):
        x = self.u_m * self.plane.u[0] + self.plane.x_mid
        y = self.u_m * self.plane.u[1] + self.plane.y_mid
        z = self.plane.z
        return (x, y, z)


class Track:
    """
    params:
        hits - list of Hit objects
    """

    def __init__(self, hits):
        self.hits = hits

    def solve(self, hll=False, quantum=False):
        """
        Finds the a1, a2, a3 and a4 coefficients in the line equations:
        | x = a1 + a3*z
        | y = a2 + a4*z
        """
        N = len(self.hits)
        G = []
        U_mg = []
        V_inv = np.zeros(shape=(N, N))
        for i, hit in enumerate(self.hits[:N]):
            z = hit.plane.z
            gamma = hit.plane.gamma
            g = [np.cos(gamma), np.sin(gamma), z *
                 np.cos(gamma), z * np.sin(gamma)]
            G.append(g)
            U_mg.append(hit.u_mg)
            V_inv[i][i] = 1 / pow(hit.plane.resolution, 2)
        G = np.array(G)
        if hll:
            return solve_linear_equation(G, U_mg).reshape(-1, 1)
        U_mg = np.array(U_mg).reshape(len(U_mg), -1)
        if quantum:
            buf = multiply(G.T, V_inv)
            buf = multiply(buf, G)
            buf = np.linalg.inv(buf)
            buf = multiply(buf, G.T)
            buf = multiply(buf, V_inv)
            x = multiply(buf, U_mg)
            x = x.reshape(len(b))
            return x
        else:
            return np.linalg.inv(G.T @ V_inv @ G) @ G.T @ V_inv @ U_mg
