import numpy as np
import time
from scipy import interpolate

from XRaySimulation import util


class ScintillatorCamera:
    def __init__(self,
                 screen_height,
                 screen_width,
                 surface_point,
                 normal,
                 camera_mag_factor,
                 camera_pixel_size,
                 camera_pixel_num,
                 camera_center_location_wrt_screen,
                 camera_noise_level,
                 camera_background,
                 ):
        """

        :param window_height:
        :param window_width:
        :param surface_point:
        :param normal:
        """

        # ------------------------------------------------------------------
        #   Store the information for the YAG screen for the camera
        # ------------------------------------------------------------------
        self.screen_height = screen_height
        self.screen_width = screen_width

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

        # ------------------------------------------------------------------
        #   Store the information for the camera pixels
        # ------------------------------------------------------------------
        # Store the gain and noise level information
        self.camera_mag_factor = camera_mag_factor
        self.camera_pixel_size = camera_pixel_size
        self.camera_pixel_num = camera_pixel_num
        self.camera_center_location_wrt_screen = camera_center_location_wrt_screen
        self.camera_noise_level = camera_noise_level
        self.camera_background = camera_background

    def is_Xray_in_window(self, xRayLoc, k0):
        # Calculate the interaction point of the X-ray on the diode window.
        intersect_on_window = util.get_intersection(initial_position=xRayLoc,
                                                    k=k0 / np.linalg.norm(k0),
                                                    normal=self.normal,
                                                    surface_point=self.surface_point)

        # Calculate the circling si111_angle
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

    def get_camera_readout(self, intensityMap2D, intensityMapRes, locOnYAG):

        # Get the coordinate of the detector with respect to the intensity field center
        coordinate_holder = np.zeros((self.camera_pixel_num[0], self.camera_pixel_num[1], 2))
        coordinate_holder[:, :, 0] = np.arange(self.camera_pixel_num[0])[:, np.newaxis] * self.camera_pixel_size
        coordinate_holder[:, :, 1] = np.arange(self.camera_pixel_num[1])[np.newaxis, :] * self.camera_pixel_size
        coordinate_holder[:, :, 0] -= locOnYAG[0] - self.camera_center_location_wrt_screen[0]
        coordinate_holder[:, :, 1] -= locOnYAG[1] - self.camera_center_location_wrt_screen[1]

        coordinate_in_intensity_map_unit = coordinate_holder / intensityMapRes

        # Interpolate to get the camera image
        camera_img = interpolate.interpn(points=(np.arange(-intensityMap2D.shape[0] // 2,
                                                           intensityMap2D.shape[0] // 2),
                                                 np.arange(-intensityMap2D.shape[1] // 2,
                                                           intensityMap2D.shape[1] // 2),),
                                         values=intensityMap2D,
                                         xi=coordinate_holder.reshape(
                                             (self.camera_pixel_num[0] * self.camera_pixel_num[1], 2)),
                                         method='nearest',
                                         bounds_error=False,
                                         fill_value=0.)
        camera_img = np.reshape(camera_img, newshape=self.camera_pixel_num)
        return camera_img

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.surface_point += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.boundary += displacement[np.newaxis, :]
