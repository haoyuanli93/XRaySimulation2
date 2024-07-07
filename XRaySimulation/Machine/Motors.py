"""
Anyway, in the end, I think it is better for me to just
create a module to mimic the motion of the motors I used
for the SD table

To be compatible with all other modules, we use unit
1. um
2. fs
3. rad

In my definition, the relation between the coordinate I used in this code and the
coordinate used by XPP is the following:

 my definition                  XPP                         physical
 0                          y rotation_axis                       vertical
 1                          x rotation_axis                       horizontal
 2                          z rotation_axis                       x-ray proportion direction
"""

"""
When I am implementing this module, 
I am very short in time.
Therefore, I have assumed that the sole purpose of this module is for the TG experiment.
Even though I have tried a bit to make it more general purpose, 
it is pretty much a failure.
If you plan to use this module for your own work, you need to think twice.
With high probability, you need to spend a significant amount of time to revise the code. 
"""

import numpy as np
from XRaySimulation import util


class xyMotor:
    def __init__(self,
                 upperLim=25000.0,
                 lowerLim=-25000.0,
                 res=5.0,
                 backlash=100.0,
                 speed_um_per_ps=1 * 1000 / 1e12,
                 dimension=None,  # The linear dimension of the motor for the visualization
                 height=4e3,
                 color='black',
                 ):
        """
        By default, the motion rotation_axis is along the x rotation_axis.

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        if dimension is None:
            dimension = [100e3, 100e3]
        self.type = "Linear xy"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps
        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.res = res

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.default_motion_dir = np.zeros(3, dtype=np.float64)
        self.default_motion_dir[1] = 1.0
        self.motion_dir = np.copy(self.default_motion_dir)

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement):

        # Change the linear stage platform center
        self.top_mount_pos += displacement
        self.bottom_mount_pos += displacement
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.bottom_mount_pos = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_pos))

        self.top_mount_dir = np.ascontiguousarray(rot_mat.dot(self.top_mount_dir))
        self.bottom_mount_dir = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_dir))

        self.default_motion_dir = np.ascontiguousarray(rot_mat.dot(self.default_motion_dir))
        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical path

                # Get the physical displacement of the table
                physical_motion = delta + self.res * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.control_positive * self.motion_dir

                # Move the stage table
                self.top_mount_pos = self.top_mount_pos + physical_motion
                print("Motor moved from {:.4f} um to to {:.4f} um".format(self.control_location,
                                                                          target))

                # Step 4: Change the status in the control system
                self.control_location = target

                # The motion time
                motion_time = abs(delta / self.control_speed)

                return motion_time, physical_motion

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir

                    motion_record = np.copy(physical_motion)

                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.motion_dir

                    motion_record += physical_motion

                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion
                    print("Motor moved from {:.4f} um to to {:.4f} um".format(self.control_location,
                                                                              target))

                    # Step 4: Change the status in the control system
                    self.control_location = target

                    motion_time = abs((2 * self.control_backlash + delta) / self.control_speed)
                    return motion_time, motion_record

                else:
                    print("The target path {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target path {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False

    def set_positive(self, motion='positive'):
        if motion == "positive":
            self.control_positive = 1.
            self.motion_dir = self.control_positive * self.default_motion_dir
        elif motion == "negative":
            self.control_positive = -1.
            self.motion_dir = self.control_positive * self.default_motion_dir
        else:
            print("motion can only be positive or negative")


class zMotor:
    def __init__(self,
                 upperLim=7e3,
                 lowerLim=-7e3,
                 res=0.5,
                 backlash=100.0,
                 speed_um_per_ps=1e3 / 1e12,
                 dimension=None,  # The linear dimension of the motor for the visualization
                 height=80e3,
                 color='black',
                 ):
        """
        By default, the motion rotation_axis is along the x rotation_axis.

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        if dimension is None:
            dimension = [100e3, 100e3]
        self.type = "Linear z"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps
        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.res = res

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.default_motion_dir = np.zeros(3, dtype=np.float64)
        self.default_motion_dir[0] = 1.0
        self.motion_dir = np.copy(self.default_motion_dir)

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.color = color  # For visualization

    def shift(self, displacement):

        # Change the linear stage platform center
        self.top_mount_pos += displacement
        self.bottom_mount_pos += displacement
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_pos = np.ascontiguousarray(rot_mat.dot(self.top_mount_pos))
        self.bottom_mount_pos = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_pos))

        self.top_mount_dir = np.ascontiguousarray(rot_mat.dot(self.top_mount_dir))
        self.bottom_mount_dir = np.ascontiguousarray(rot_mat.dot(self.bottom_mount_dir))

        self.default_motion_dir = np.ascontiguousarray(rot_mat.dot(self.default_motion_dir))
        self.motion_dir = np.ascontiguousarray(rot_mat.dot(self.motion_dir))
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical path

                # Get the physical displacement of the table
                physical_motion = delta + self.res * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.motion_dir

                # Move the stage table
                self.top_mount_pos = self.top_mount_pos + physical_motion
                print("Motor moved from {:.4f} um to to {:.4f} um".format(self.control_location,
                                                                          target))

                # Step 4: Change the status in the control system
                self.control_location = target
                # The motion time
                motion_time = abs(delta / self.control_speed)

                return motion_time, physical_motion

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.motion_dir
                    motion_record = np.copy(physical_motion)
                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.res * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.motion_dir
                    motion_record += physical_motion
                    # Move the stage table
                    self.top_mount_pos = self.top_mount_pos + physical_motion
                    print("Motor moved from {:.4f} um to to {:.4f} um".format(self.control_location,
                                                                              target))

                    # Step 4: Change the status in the control system
                    self.control_location = target

                    motion_time = abs((2 * self.control_backlash + delta) / self.control_speed)
                    return motion_time, motion_record

                else:
                    print("The target path {:.2f} um plus back clash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target path {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False

    def set_positive(self, motion='positive'):
        if motion == "positive":
            self.control_positive = 1.
            self.motion_dir = self.control_positive * self.default_motion_dir
        elif motion == "negative":
            self.control_positive = -1.
            self.motion_dir = self.control_positive * self.default_motion_dir
        else:
            print("motion can only be positive or negative")


class RotationMotor:
    def __init__(self,
                 upperLim=np.deg2rad(180),
                 lowerLim=-np.deg2rad(-180),
                 res=1.0,
                 backlash=0.05,
                 speed_rad_per_ps=1 * 1000 / 1e12,
                 dimension=None,
                 height=60e3,
                 color='grey'
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        if dimension is None:
            dimension = [100e3, 100e3]

        self.type = "Rotation"
        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.control_location = 0.0  # rad

        self.control_positive = 1.

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps
        self.res = res

        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[1] = 1.0

        self.default_rotation_axis = np.zeros(3, dtype=np.float64)
        self.default_rotation_axis[0] = 1.0
        self.rotation_axis = np.copy(self.default_rotation_axis)

        self.rotation_center = np.zeros(3, dtype=np.float64)
        self.rotation_center[0] = height

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.array([0, 0, 0, ], dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.default_rotation_axis = np.ascontiguousarray(rot_mat.dot(self.default_rotation_axis))
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = (target - self.control_location)

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.res * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Step 4 : change the control system information
                print("Motor moved from {:.5f} to {:.5f} degree".format(np.rad2deg(self.control_location),
                                                                        np.rad2deg(target)))
                self.control_location = target

                motion_time = abs(delta / self.control_speed)
                return motion_time, rotMat

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)
                    # Step 4 : change the control system information
                    print("Motor moved from {:.5f} to {:.5f} degree".format(np.rad2deg(self.control_location),
                                                                            np.rad2deg(target)))
                    self.control_location = target

                    motion_time = abs((2 * self.control_backlash + delta) / self.control_speed)
                    return motion_time, np.dot(rotMat2, rotMat1)

                else:
                    print("The target path {:.2f} rad plus backlash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target path {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

        if getMotionTime:
            return 0

    def user_getPosition(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False

    def set_positive(self, motion='positive'):
        if motion == "positive":
            self.control_positive = 1.
            self.rotation_axis = self.control_positive * self.default_rotation_axis
        elif motion == "negative":
            self.control_positive = -1.
            self.rotation_axis = self.control_positive * self.default_rotation_axis


class SwivalMotor:
    def __init__(self,
                 upperLim=np.deg2rad(5),
                 lowerLim=-np.deg2rad(5),
                 res=np.deg2rad(0.000756),
                 backlash=0.05,
                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                 dimension=None,
                 rot_center_height=70e3,
                 height=26e3,
                 color='grey'
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        if dimension is None:
            dimension = [70e3, 70e3]

        self.type = "Swival"
        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim
        self.control_location = 0.0  # rad

        self.control_positive = 1

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps
        self.res = res

        self.deg0direction = np.zeros(3, dtype=np.float64)
        self.deg0direction[0] = 1.0

        self.default_rotation_axis = np.zeros(3, dtype=np.float64)
        self.default_rotation_axis[2] = 1.0
        self.rotation_axis = np.copy(self.default_rotation_axis)

        self.rotation_center = np.zeros(3, dtype=np.float64)
        self.rotation_center[0] = rot_center_height + height

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface
        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3, dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])
        self.color = color

    def shift(self, displacement):

        # Change the linear stage platform center
        self.rotation_center += np.copy(displacement)
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.deg0direction = np.ascontiguousarray(rot_mat.dot(self.deg0direction))
        self.rotation_center = np.ascontiguousarray(rot_mat.dot(self.rotation_center))
        self.rotation_axis = np.ascontiguousarray(rot_mat.dot(self.rotation_axis))

        self.default_rotation_axis = np.ascontiguousarray(rot_mat.dot(self.default_rotation_axis))
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)
        self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))

    def user_move_abs(self, target):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.res * (np.random.rand() - 0.5),
                    axis=self.rotation_axis)

                # Update the zero deg direction
                self.deg0direction = np.dot(rotMat, self.deg0direction)

                # Step 4 : change the control system information
                print("Motor moved from {:.5f} to {:.5f} degree".format(np.rad2deg(self.control_location),
                                                                        np.rad2deg(target)))
                self.control_location = target

                motion_time = abs(delta) / self.control_speed
                return motion_time, rotMat

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)

                    self.deg0direction = np.dot(rotMat1, self.deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.res * (np.random.rand() - 0.5),
                        axis=self.rotation_axis)
                    self.deg0direction = np.dot(rotMat2, self.deg0direction)

                    # Step 4 : change the control system information
                    print("Motor moved from {:.5f} to {:.5f} degree".format(np.rad2deg(self.control_location),
                                                                            np.rad2deg(target)))
                    self.control_location = target

                    motion_time = abs((2 * self.control_backlash + delta) / self.control_speed)
                    return motion_time, np.dot(rotMat2, rotMat1)

                else:
                    print("The target path {:.2f} rad plus backlash is beyond the limit of this motor.".format(
                        target))
                    print("No motion is committed.")
        else:
            print("The target path {:.2f} um is beyond the limit of this motor.".format(target))
            print("No motion is committed.")

    def user_get_position(self):
        return self.control_location

    def __check_limit(self, val):
        if (val >= self.control_limits[0]) and (val <= self.control_limits[1]):
            return True
        else:
            return False

    def set_positive(self, motion='positive'):
        if motion == "positive":
            self.control_positive = 1.
            self.rotation_axis = self.control_positive * self.default_rotation_axis
        elif motion == "negative":
            self.control_positive = -1.
            self.rotation_axis = self.control_positive * self.default_rotation_axis


class Breadboard:
    def __init__(self, hole_num_x, hole_num_z, thickness=12.7e3, gauge='metric'):
        self.holes_top = np.zeros((hole_num_x, hole_num_z, 3))
        self.holes_top[:, :, 1] = np.arange(hole_num_x)[:, np.newaxis]
        self.holes_top[:, :, 2] = np.arange(hole_num_z)[np.newaxis, :]
        if gauge == 'metric':
            self.holes_top *= 25e3
        elif gauge == 'imperial':
            self.holes_top *= 25.4e3

        self.holes_bottom = np.copy(self.holes_top)
        self.holes_top[:, :, 0] += thickness

        self.normal = np.array([1, 0, 0], dtype=np.float64)

    def shift(self, displacement):
        self.holes_top += displacement[np.newaxis, np.newaxis, :]
        self.holes_bottom += displacement[np.newaxis, np.newaxis, :]

    def rotate(self, rot_mat):
        self.holes_top = np.dot(self.holes_top, rot_mat.T)
        self.holes_bottom = np.dot(self.holes_top, rot_mat.T)
        self.normal = np.dot(rot_mat, self.normal)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


class AdaptorPlate:
    def __init__(self, height=10e3, dimension=None, color='black'):
        if dimension is None:
            dimension = (10e4, 10e4)

        self.top_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface

        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3, dtype=np.float64)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.color = color  # For visualization

    def shift(self, displacement):
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.dot(self.boundary, rot_mat.T)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


class L_Bracket:
    def __init__(self, height=10e3, dimension=None, color='black'):
        if dimension is None:
            dimension = (10e4, 10e4)

        self.top_mount_dir = np.array([0, 1.0, 0, ])  # the normal direction of the top mounting surface
        self.top_mount_pos = np.array([height, 0, 0, ])  # The center of the top mounting surface

        self.bottom_mount_dir = np.array([1.0, 0, 0, ])  # the normal direction of the top mounting surface
        self.bottom_mount_pos = np.zeros(3)  # The center of the top mounting surface

        self.boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                  np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                  np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                  ])

        self.color = color  # For visualization

    def shift(self, displacement):
        self.top_mount_pos += np.copy(displacement)
        self.bottom_mount_pos += np.copy(displacement)
        self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat):
        self.top_mount_dir = np.dot(rot_mat, self.top_mount_dir)
        self.top_mount_pos = np.dot(rot_mat, self.top_mount_pos)
        self.bottom_mount_dir = np.dot(rot_mat, self.bottom_mount_dir)
        self.bottom_mount_pos = np.dot(rot_mat, self.bottom_mount_pos)

        self.boundary = np.dot(self.boundary, rot_mat.T)

    def rotate_wrt_point(self, rot_mat, ref_point):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp))

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp))


def install_motors_on_motor_or_adaptors(motor_tower, motor_or_adaptor):
    """
    :param motor_or_adaptor:
    :param motor_tower:
    :return:
    """
    # Step 2 move the motor such that the center of the bottom mounting surface of the first motor is
    # the same as the top mounting surface of the new motor or adaptor.
    displacement = np.copy(motor_or_adaptor.top_mount_pos) - np.copy(motor_tower[0].bottom_mount_pos)
    for motor in motor_tower:
        motor.shift(displacement=displacement)

    # Add the new object to the motor-tower
    motor_tower = [motor_or_adaptor, ] + motor_tower
    return motor_tower


def install_motors_on_breadboard(motor_stack, breadboard, diag_hole_idx1, diag_hole_idx2):
    """
    Calculate the path to install the motor stack

    :param motor_stack:
    :param breadboard:
    :param diag_hole_idx1:
    :param diag_hole_idx2:
    :return:
    """
    # Get the path where to install the motor stack
    position = breadboard.holes_top[diag_hole_idx1[0], diag_hole_idx1[1]]
    position += breadboard.holes_top[diag_hole_idx2[0], diag_hole_idx2[1]]
    position /= 2.

    displacement = position - motor_stack[0].bottom_mount_pos

    for motor in motor_stack:
        motor.shift(displacement=displacement)

    return motor_stack


def get_motors_with_model_for_axis(model, rot_center_height=70e3, color='k', axis='x'):
    if model == "XA10A":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=12.5 * 1000,
                            lowerLim=-12.5 * 1000,
                            res=0.5,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=30e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "UTS100CC":
        print("Create a {} motor, moving along x rotation_axis.".format(model))
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=2,
                            backlash=100,
                            speed_um_per_ps=2 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=32e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

    elif model == "XA10A-L101":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=2,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[190e3, 100e3],
                            height=50e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass


    elif model == "XA07A":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=10 * 1000,
                            lowerLim=-10 * 1000,
                            res=1,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[70e3, 70e3],
                            height=21e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ABL1000WB":
        print("Create a ABL1000WB motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=25 * 1000,
                            lowerLim=-25 * 1000,
                            res=0.002,
                            backlash=1,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[307e3, 185e3],
                            height=75e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            print("Warning, cannot create ABL1000WB along y rotation_axis automatically.")
            print("Please create this motor manually.")
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "RA10A":
        print("Create a {} motor, rotating around y rotation_axis.".format(model))
        motor_obj = RotationMotor(upperLim=10 * np.pi,
                                  lowerLim=-10 * np.pi,
                                  res=np.deg2rad(0.0002),
                                  backlash=np.deg2rad(0.01),
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)

    elif model == "RA05A":
        print("Create a {} motor, rotating around y rotation_axis.".format(model))
        motor_obj = RotationMotor(upperLim=np.deg2rad(360),
                                  lowerLim=-np.deg2rad(360),
                                  res=np.deg2rad(0.002),
                                  backlash=np.deg2rad(0.01),
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x rotation_axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            pass
        elif axis == "z":
            print("Rotate motor to rotate around z rotation_axis")
            rot_mat = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ZA10A":
        print("Create a XA10A motor, moving along y rotation_axis.")
        motor_obj = zMotor(upperLim=7e3,
                           lowerLim=-7e3,
                           res=0.1,
                           backlash=100,
                           speed_um_per_ps=1 * 1000 / 1e12,
                           dimension=[100e3, 100e3],
                           height=30e3,
                           color=color)

    elif model == "SA07A":
        print("Create a SA07A motor, rotating around z rotation_axis.")
        motor_obj = SwivalMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=np.deg2rad(0.000756),
                                backlash=np.deg2rad(0.01),
                                speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                dimension=[70e3, 70e3],
                                rot_center_height=rot_center_height,
                                height=26e3,
                                color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x rotation_axis")

            rot_mat = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            print("Rotate motor to rotate around y rotation_axis")

            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            pass

    elif model == "SA05A-R2S01":
        print("Create a {} motor pair. The normal is pointing along y".format(model))
        motor_obj1 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=np.deg2rad(0.002126),
                                 backlash=np.deg2rad(0.01),
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=68,
                                 height=18e3,
                                 color=color)
        motor_obj2 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=np.deg2rad(0.002126),
                                 backlash=np.deg2rad(0.01),
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=50,
                                 height=18e3,
                                 color=color)
        rot_mat = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
        motor_obj2.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(motor_obj2.bottom_mount_pos))
        motor_obj = install_motors_on_motor_or_adaptors(motor_tower=[motor_obj2, ],
                                                        motor_or_adaptor=motor_obj1)
        if axis == 'x':
            print("Rotate motor to face x rotation_axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj1.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
            motor_obj2.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
        elif axis == 'y':
            pass
        elif axis == "z":
            print("Warning, cannot create {} along z rotation_axis automatically.".format(model))
            print("Please create this motor manually.")

    else:
        print("Motor with model {} has not been defined in this simulator.".format(model))
        motor_obj = 0

    return motor_obj


def get_ideal_motors_with_model_for_axis(model, rot_center_height=70e3, color='k', axis='x'):
    if model == "XA10A":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=12.5 * 1000,
                            lowerLim=-12.5 * 1000,
                            res=0.,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=30e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "UTS100CC":
        print("Create a {} motor, moving along x rotation_axis.".format(model))
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=0,
                            backlash=100,
                            speed_um_per_ps=2 * 1000 / 1e12,
                            dimension=[100e3, 100e3],
                            height=32e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

    elif model == "XA10A-L101":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=50 * 1000,
                            lowerLim=-50 * 1000,
                            res=0,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[190e3, 100e3],
                            height=50e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))

        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass


    elif model == "XA07A":
        print("Create a XA10A motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=10 * 1000,
                            lowerLim=-10 * 1000,
                            res=0,
                            backlash=100,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[70e3, 70e3],
                            height=21e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ABL1000WB":
        print("Create a ABL1000WB motor, moving along x rotation_axis.")
        motor_obj = xyMotor(upperLim=25 * 1000,
                            lowerLim=-25 * 1000,
                            res=0.002,
                            backlash=1,
                            speed_um_per_ps=1 * 1000 / 1e12,
                            dimension=[307e3, 185e3],
                            height=75e3,
                            color=color)
        if axis == 'x':
            pass
        elif axis == 'y':
            print("Warning, cannot create ABL1000WB along y rotation_axis automatically.")
            print("Please create this motor manually.")
        elif axis == "z":
            rot_mat = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "RA10A":
        print("Create a {} motor, rotating around y rotation_axis.".format(model))
        motor_obj = RotationMotor(upperLim=10 * np.pi,
                                  lowerLim=-10 * np.pi,
                                  res=0,  # np.deg2rad(0.0002),
                                  backlash=np.deg2rad(0.01),
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)

    elif model == "RA05A":
        print("Create a {} motor, rotating around y rotation_axis.".format(model))
        motor_obj = RotationMotor(upperLim=np.deg2rad(360),
                                  lowerLim=-np.deg2rad(360),
                                  res=0,  # np.deg2rad(0.002),
                                  backlash=np.deg2rad(0.01),
                                  speed_rad_per_ps=np.deg2rad(0.1) / 1e12,
                                  dimension=[100e3, 100e3],
                                  height=60e3,
                                  color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x rotation_axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            pass
        elif axis == "z":
            print("Rotate motor to rotate around z rotation_axis")
            rot_mat = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
            pass

    elif model == "ZA10A":
        print("Create a XA10A motor, moving along y rotation_axis.")
        motor_obj = zMotor(upperLim=7e3,
                           lowerLim=-7e3,
                           res=0,  # 0.1,
                           backlash=100,
                           speed_um_per_ps=1 * 1000 / 1e12,
                           dimension=[100e3, 100e3],
                           height=30e3,
                           color=color)

    elif model == "SA07A":
        print("Create a SA07A motor, rotating around z rotation_axis.")
        motor_obj = SwivalMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=0,  # np.deg2rad(0.000756),
                                backlash=np.deg2rad(0.01),
                                speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                dimension=[70e3, 70e3],
                                rot_center_height=rot_center_height,
                                height=26e3,
                                color=color)
        if axis == 'x':
            print("Rotate motor to rotate around x rotation_axis")

            rot_mat = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == 'y':
            print("Rotate motor to rotate around y rotation_axis")

            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            rot_mat = np.dot(rot_mat, np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
            motor_obj.rotate_wrt_point(rot_mat=rot_mat,
                                       ref_point=np.copy(motor_obj.bottom_mount_pos))
        elif axis == "z":
            pass

    elif model == "SA05A-R2S01":
        print("Create a {} motor pair. The normal is pointing along y".format(model))
        motor_obj1 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=0,  # np.deg2rad(0.002126),
                                 backlash=np.deg2rad(0.01),
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=68,
                                 height=18e3,
                                 color=color)
        motor_obj2 = SwivalMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=0,  # np.deg2rad(0.002126),
                                 backlash=np.deg2rad(0.01),
                                 speed_rad_per_ps=np.deg2rad(0.4) / 1e12,
                                 dimension=[50e3, 50e3],
                                 rot_center_height=50,
                                 height=18e3,
                                 color=color)
        rot_mat = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
        motor_obj2.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(motor_obj2.bottom_mount_pos))
        motor_obj = install_motors_on_motor_or_adaptors(motor_tower=[motor_obj2, ],
                                                        motor_or_adaptor=motor_obj1)
        if axis == 'x':
            print("Rotate motor to face x rotation_axis")
            rot_mat = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])
            motor_obj1.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
            motor_obj2.rotate_wrt_point(rot_mat=rot_mat,
                                        ref_point=np.copy(motor_obj1.bottom_mount_pos))
        elif axis == 'y':
            pass
        elif axis == "z":
            print("Warning, cannot create {} along z rotation_axis automatically.".format(model))
            print("Please create this motor manually.")

    else:
        print("Motor with model {} has not been defined in this simulator.".format(model))
        motor_obj = 0

    return motor_obj
