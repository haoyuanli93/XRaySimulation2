import numpy as np
import time

from XRaySimulation import util


class Diode:
    def __init__(self,
                 window_height,
                 window_width,
                 surface_point,
                 normal,
                 gain,
                 noise_level,
                 ):
        """

        :param window_height:
        :param window_width:
        :param surface_point:
        :param normal:
        """
        self.window_height = window_height
        self.window_width = window_width

        self.surface_point = np.copy(surface_point)
        self.normal = np.copy(normal)

        # direction perpendicular to the normal direction
        direction1 = np.cross(np.array([1, 0, 0]), self.normal)
        direction1 /= np.linalg.norm(direction1)
        direction2 = np.cross(self.normal, direction1)

        point0 = self.surface_point - direction1 * self.screen_width / 2. - direction2 * self.screen_height / 2.
        point1 = point0 + direction1 * self.screen_width
        point2 = point1 + direction2 * self.screen_height
        point3 = point2 - direction1 * self.screen_width
        point4 = point3 - direction1 * self.screen_height

        # Assemble
        self.boundary = np.vstack([point0, point1, point2, point3, point4])

        # Store the gain and noise level information
        self.gain = gain
        self.noise_level = noise_level

    def is_Xray_in_window(self, xRayLoc, k0):
        # Calculate the interaction point of the X-ray on the diode window.
        intersect_on_window = util.get_intersection(initial_position=xRayLoc,
                                                    k=k0 / np.linalg.norm(k0),
                                                    normal=self.normal,
                                                    surface_point=self.surface_point)

        # Calculate the circling angle
        difference = self.boundary - intersect_on_window[np.newaxis, :]
        difference /= np.linalg.norm(difference, axis=-1)[np.newaxis, 0]
        cross_vec = np.cross(difference[:-1], difference[1:])

        angle_list = np.arcsin(np.linalg.norm(cross_vec, axis=-1))
        cross_vec /= np.linalg.norm(cross_vec, axis=-1)[:, np.newaxis]
        cross_vec *= angle_list[:, np.newaxis]

        angle = np.sum(cross_vec, axis=-1)
        angle = np.linalg.norm(angle)

        if angle > 0.1:
            print("The beam is in the window of the diode.")
        else:
            print("The beam is not in the window of the diode.")

        return angle, intersect_on_window

    def get_diode_readout(self, eField, voxel):
        randomSeed = int(time.time() * 1e6) % 65536
        np.random.seed(randomSeed)

        pulseEnergy = np.sum(np.square(np.abs(eField)))
        pulseEnergy *= voxel

        pulseEnergyReadOut = pulseEnergy * self.gain + np.random.rand(1) * self.noise_level

        return pulseEnergyReadOut, pulseEnergy

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.surface_point += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.surface_point = np.ascontiguousarray(rot_mat.dot(self.surface_point))
        self.normal = np.ascontiguousarray(rot_mat.dot(self.normal))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)
