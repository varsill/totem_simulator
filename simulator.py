import numpy as np
import math
from common import *
import matplotlib.pyplot as plt


class RomanPotPlane:
    """
    params:
        gamma - angle between the y axis and the strips (in radians)

        center_translation - offset of the plane center from the center of the pipe
        z - offset along the telescope (how far inside of the tube is the detector placed)

        how_many_strips - number of strips per plane
        width - width of the detector's plane (in nanometers), in the direction perpendicular to the strips
        height - height of the detector's plane (in nanometers), paralel to the strips
    """

    def __init__(self, z, gamma, center_translation_x, center_translation_y, how_many_strips, width, height):
        """
        u - unit vector perpendicular to the strips direction
        v - unit vector parallel to the strips direction
        """

        self.gamma = gamma

        self.center_translation_x = center_translation_x
        self.center_translation_y = center_translation_y
        self.z = z

        self.u = np.array([np.cos(gamma), np.sin(gamma)])
        self.v = np.array([np.cos(gamma - math.pi / 2), np.sin(gamma - math.pi / 2)])

        self.lower_left_x = center_translation_x - self.u[0] * width / 2 - self.v[0] * height / 2
        self.lower_left_y = center_translation_y - self.u[1] * width / 2 - self.v[1] * height / 2

        self.width = width
        self.height = height
        self.how_many_strips = how_many_strips
        self.strip_width = self.width / self.how_many_strips



class RomanPot:
    """
    params:
        name - a string representing a singular Roman Pot
        planes - a list of RomanPotPlane objects
    """

    def __init__(self, name: str = "RomanPot_01"):
        self.name = name
        self.planes = []

    def addPlane(self, romanPotPlane: RomanPotPlane):
        self.planes.append(romanPotPlane)


class Hit:
    """
    params:
        plane - an object describing which roman pot plane was hit
        ordering_number_of_strip - Startuje od 0 do max_strip-1 # TODO Dopisz więcej o tym
        hit_u_cord - długość od lewego-dolnego rogu do miejsca uderzenia w przestrzeni (u,v)# TODO Napisz to lepiej
        hit_u_cord_from_global - długość od środka tuby do miejsca uderzenia w przestrzeni (u,v)# TODO Napisz to lepiej
    """

    def __init__(self, plane, ordering_number_of_strip: int):
        self.plane = plane
        self.ordering_number_of_strip = ordering_number_of_strip

        self.hit_u_cord = (ordering_number_of_strip + 0.5) * plane.strip_width
        self.hit_u_cord_from_global = self.hit_u_cord + plane.u @ np.array([plane.lower_left_x, plane.lower_left_y]).T

        self.global_x, self.global_y, self.global_z = self.__calculate_global_cords()

    def __calculate_global_cords(self):
        x = self.hit_u_cord * self.plane.u[0] + self.plane.lower_left_x
        y = self.hit_u_cord * self.plane.u[1] + self.plane.lower_left_y
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
            g = [np.cos(gamma), np.sin(gamma), z * np.cos(gamma), z * np.sin(gamma)]
            G.append(g)
            U_mg.append(hit.hit_u_cord_from_global)
            V_inv[i][i] = pow(hit.plane.strip_width, 2)

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

    @staticmethod
    def reverseSolveFromCoefficients(romanPot: RomanPot, a1=0, a2=0, a3=0, a4=0):
        """
        Creates a Track (list of Hit objects), based on the coefficients of the line equations and specifications of RomanPotPlanes.
        | x = a1 + a3*z
        | y = a2 + a4*z
        """
        generatedHits = []

        for i in range(len(romanPot.planes)):
            currentPlane = romanPot.planes[i]

            z_value = currentPlane.z
            x_value = a1 + a3 * z_value
            y_value = a2 + a4 * z_value

            # oblicz współrzędne w przestrzeni (u,v) w sposob odwrotny do def __calculate_global_cords(self):
            # długość na osi u przelicz na numer paska (dzielenie bez reszty przez szerokość paska)

            def __calculate_global_cords(self):
                x = self.hit_u_cord * self.plane.u[0] + self.plane.lower_left_x
                y = self.hit_u_cord * self.plane.u[1] + self.plane.lower_left_y
                z = self.plane.z
                return (x, y, z)

            hit_u_cord = (x_value - currentPlane.lower_left_x) / currentPlane.u[0]

            strip_number = int(hit_u_cord / currentPlane.strip_width)

            generatedHits.append(
                Hit(
                    currentPlane,
                    strip_number,  # TODO oblicz tutaj w poprawny sposób na podstawie wcześniejszych danych, który z pasków powinien wykryć cząstkę
                )
            )

        return Track(generatedHits)


def compare_coefficients(a1, a2, a3, a4, b1, b2, b3, b4, z_start=0, z_end=1):
    """Compares coefficients of two linear regression models. The first one is described by the following system
    of equations:
      x = a1*z+a3
      y = a2*z+a4
      The second one is described by:
      x = b1*z+b3
      y = b2*z+b4

    The method calculates the cumulative square error between lines determined by these models,
    with z in range (z_start, z_end), using the euclidean distance between the points.
    """
    return (
        z_end**3 / 3 * ((a1 - b1) ** 2 + (a2 - b2) ** 2)
        + z_end**2 * ((a1 - b1) * (a3 - b3) + (a2 - b2) * (a4 - b4))
        + z_end * ((a3 - b3) ** 2 + (a4 - b4) ** 2)
        - (
            z_start**3 / 3 * ((a1 - b1) ** 2 + (a2 - b2) ** 2)
            + z_start**2 * ((a1 - b1) * (a3 - b3) + (a2 - b2) * (a4 - b4))
            + z_start * ((a3 - b3) ** 2 + (a4 - b4) ** 2)
        )
    )


def plot_error(coeffs1, coeffs2, first_z, last_z):
    """Plots the relation describing
    how the cumulative quared euclidean error between lines determined by two linear regression models changes with
    respect to variable z. Variable z describes the upper limit of the range, for which the error is calculated: (first_z, z)

    Args:
        coeffs1 : array of coefficients describing the first regression model
        coeffs2 (_type_): array of coefficients describing the first regression model
        z_start (_type_): starting poitjn
        z_end (_type_): _description_
    """
    a1 = coeffs1[0]
    a2 = coeffs1[1]
    a3 = coeffs1[2]
    a4 = coeffs1[3]

    b1 = coeffs2[0]
    b2 = coeffs2[1]
    b3 = coeffs2[2]
    b4 = coeffs2[3]

    errors = []
    zs = np.linspace(first_z, last_z, 100)
    for z in zs:
        error = compare_coefficients(a1, a2, a3, a4, b1, b2, b3, b4, first_z, z)
        errors.append(error)

    plt.scatter(zs, errors)
