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


class LinearMotor:
    def __init__(self,
                 upperLim=25000.0,
                 lowerLim=-25000.0,
                 res=5.0,
                 backlash=100.0,
                 feedback_noise_level=1.0,
                 speed_um_per_ps=1 * 1000 / 1e12,
                 dimension=None,  # The linear dimension of the motor for the visualization
                 color='black',
                 ):
        """
        By default, the motion rotation_axis is along the x rotation_axis.

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed_um_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        # With respect to the default positive direction, whether change the motor motion direction
        if dimension is None:
            dimension = [100e3, 100e3]
        self.control_motor_type = "Linear"

        self.control_location = 0.0
        self.control_positive = 1.
        self.control_speed = speed_um_per_ps

        self.control_backlash = backlash

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim

        # ---------------------------------------------------
        # Define quantities of the physical system
        # ---------------------------------------------------
        self.physical_location = np.zeros(3, dtype=np.float64)
        self.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.physical_positive_direction[0] = 1.0

        # ---------------------------------------------------
        # Define quantities describing the error between the control system and the physical system
        # I call them device property (dp_...)
        # ---------------------------------------------------
        self.dp_resolution = res
        self.dp_feedback_noise = feedback_noise_level
        self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 4cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.dp_boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                     np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                     np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                     np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                     np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                     ])

        self.color = color  # For visualization

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.physical_location += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.dp_boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.physical_location = np.ascontiguousarray(rot_mat.dot(self.physical_location))
        self.physical_positive_direction = np.ascontiguousarray(rot_mat.dot(self.physical_positive_direction))

        if include_boundary:
            self.dp_boundary = np.asanyarray(np.dot(self.dp_boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location
            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction
                # Step 3: change the physical path

                # Get the physical displacement of the table
                physical_motion = delta + self.dp_resolution * (np.random.rand() - 0.5)
                physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                # Move the stage table
                self.physical_location = self.physical_location + physical_motion

                # Step 4: Change the status in the control system

                self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                self.control_location = target + self.dp_feedback_noise_instance

                print("Motor moved to {:.2f} um".format(self.control_location))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):
                    # Get the physical displacement of the table
                    physical_motion = self.control_backlash + delta + self.dp_resolution * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                    # Move the stage table
                    self.physical_location = self.physical_location + physical_motion

                    # Get the physical displacement of the table
                    physical_motion = -self.control_backlash + self.dp_resolution * (np.random.rand() - 0.5)
                    physical_motion = physical_motion * self.control_positive * self.physical_positive_direction

                    # Move the stage table
                    self.physical_location = self.physical_location + physical_motion

                    # Step 4: Change the status in the control system
                    self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                    self.control_location = target + self.dp_feedback_noise_instance
                    print("Motor moved to {:.2f} um".format(self.control_location))

                else:
                    print("The target path {:.2f} um plus back clash is beyond the limit of this motor.".format(
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


class RotationMotor:
    def __init__(self,
                 upperLim=np.deg2rad(180),
                 lowerLim=-np.deg2rad(-180),
                 res=1.0,
                 backlash=0.05,
                 feedback_noise_level=1.0,
                 speed_rad_per_ps=1 * 1000 / 1e12,
                 dimension=None
                 ):
        """

        :param upperLim:
        :param lowerLim:
        :param res:
        :param backlash:
        :param feedback_noise_level: This is the error associate feed back.
                               It is a static value that only change after each motion.
        :param speed_rad_per_ps: the speed in um / ps
        """

        # ---------------------------------------------------
        # Define quantities in the control system
        # ---------------------------------------------------
        if dimension is None:
            dimension = [100e3, 100e3]
        self.control_motor_type = "Rotation"

        self.control_limits = np.zeros(2)
        self.control_limits[0] = lowerLim
        self.control_limits[1] = upperLim

        self.control_location = 0.0  # rad

        self.control_backlash = backlash
        self.control_speed = speed_rad_per_ps

        # ---------------------------------------------------
        # Define quantities of the physical system
        # ---------------------------------------------------
        self.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.physical_deg0direction[1] = 1.0
        self.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.physical_rotation_axis[0] = 1.0
        self.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # ---------------------------------------------------
        # Define quantities describing the error between the control system and the physical system
        # I call them device property (dp_...)
        # ---------------------------------------------------
        self.dp_resolution = res
        self.dp_feedback_noise = feedback_noise_level
        self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise

        # Define a boundary to visualize
        # Assume that the dimension is 10cm by 10cm
        # Assume that the linear motion stage center is initially at the (0, 0, 0)
        self.dp_boundary = np.array([np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                     np.array([0, -dimension[0] / 2, dimension[1] / 2]),
                                     np.array([0, dimension[0] / 2, dimension[1] / 2]),
                                     np.array([0, dimension[0] / 2, -dimension[1] / 2]),
                                     np.array([0, -dimension[0] / 2, -dimension[1] / 2]),
                                     ])

    def shift(self, displacement, include_boundary=True):

        # Change the linear stage platform center
        self.physical_rotation_center += displacement

        # Change the boundary with the displacement.
        if include_boundary:
            self.dp_boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.physical_deg0direction = np.ascontiguousarray(rot_mat.dot(self.physical_deg0direction))
        self.physical_rotation_center = np.ascontiguousarray(rot_mat.dot(self.physical_rotation_center))
        self.physical_rotation_axis = np.ascontiguousarray(rot_mat.dot(self.physical_rotation_axis))

        if include_boundary:
            self.dp_boundary = np.asanyarray(np.dot(self.dp_boundary, rot_mat.T))

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)

    def user_move_abs(self, target, getMotionTime=True):
        # Step 1: check if the target value is within the limit or not
        if self.__check_limit(val=target):

            # Step 2: if it is with in the limit, then consider the back-clash effect
            delta = target - self.control_location

            if delta * self.control_backlash <= 0:  # Move to the opposite direction as the back-clash direction

                # Step 3: change the physical status of the motor
                # Get the rotation matrix
                rotMat = util.get_rotmat_around_axis(
                    angleRadian=delta + self.dp_resolution * (np.random.rand() - 0.5),
                    axis=self.physical_rotation_axis)

                # Update the zero deg direction
                self.physical_deg0direction = np.dot(rotMat, self.physical_deg0direction)

                # Step 4 : change the control system information
                self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                self.control_location = target + self.dp_feedback_noise_instance

                print("Motor moved to {:.2f} um".format(self.control_location))

            else:
                # Need to move by the delta plus some back clash distance. Therefore need to check boundary again
                if self.__check_limit(val=self.control_location + self.control_backlash + delta):

                    rotMat1 = util.get_rotmat_around_axis(
                        angleRadian=self.control_backlash + delta + self.dp_resolution * (np.random.rand() - 0.5),
                        axis=self.physical_rotation_axis)

                    self.physical_deg0direction = np.dot(rotMat1, self.physical_deg0direction)

                    rotMat2 = util.get_rotmat_around_axis(
                        angleRadian=- self.control_backlash + self.dp_resolution * (np.random.rand() - 0.5),
                        axis=self.physical_rotation_axis)
                    self.physical_deg0direction = np.dot(rotMat2, self.physical_deg0direction)

                    self.dp_feedback_noise_instance = (np.random.rand() - 0.5) * self.dp_feedback_noise
                    self.control_location = target + self.dp_feedback_noise_instance
                    print("Motor moved to {:.2f} rad".format(self.control_location))
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


def get_motors_with_model_for_axis(model, color='k', axis='x'):
    if model == "XA10A":
        motor_obj = LinearMotor(upperLim=12.5 * 1000,
                                lowerLim=-12.5 * 1000,
                                res=5,
                                backlash=100,
                                feedback_noise_level=1,
                                speed_um_per_ps=1 * 1000 / 1e12,
                                dimension=[70e3, 70e3])
    else:
        print("Motor with model {} has not been defined in this simulator.".format(model))
        motor_obj = 0

    return motor_obj


# --------------------------------------------------------------------
#    Here, I define a few commonly used motor composition.
#    Even though they do not have any scientific generality
#    my gut feeling is that they should have a long enough
#    lifetime that deserve such a position in the main
#    body of this simulation package.
# --------------------------------------------------------------------
class CrystalTower_x_y_theta_chi:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 channelCut,
                 crystal_loc):
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.th = RotationMotor(upperLim=np.deg2rad(360),
                                lowerLim=-np.deg2rad(360),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                feedback_noise_level=1e-9,
                                speed_rad_per_ps=0.01 / 1e12, )

        self.chi = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.optics = channelCut

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.th.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.th.physical_deg0direction[1] = 1.0  #
        self.th.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.th.physical_rotation_axis[0] = 1.0
        self.th.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation path of the x stage
        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.th.shift(displacement=th_stage_center)

        self.chi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.chi.physical_deg0direction[0] = 1.0  #
        self.chi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_axis[1] = 1.0
        self.chi.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        chi_stage_center = np.zeros(3, dtype=np.float64)
        chi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.chi.shift(displacement=chi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def th_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th.user_move_abs(target=target, getMotionTime=True)
        self.chi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.physical_rotation_center, include_boundary=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.physical_rotation_center, include_boundary=True)

    def chi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.chi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.chi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.chi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.physical_rotation_center, include_boundary=True)

    # def plot_motors(controller, ax):
    #    # Plot motors and crystals one by one
    #    ax.plot(controller.x.dp_boundary[:, 2], controller.x.dp_boundary[:, 1],
    #            linestyle='--', linewidth=1, label="x", color=controller.color_list[0])
    #    ax.plot(controller.y.dp_boundary[:, 2], controller.y.dp_boundary[:, 1],
    #            linestyle='--', linewidth=1, label="y", color=controller.color_list[1])
    #    ax.plot(controller.th.dp_boundary[:, 2], controller.th.dp_boundary[:, 1],
    #            linestyle='--', linewidth=1, label="th", color=controller.color_list[2])
    #    ax.plot(controller.chi.dp_boundary[:, 2], controller.chi.dp_boundary[:, 1],
    #            linestyle='--', linewidth=1, label="chi", color=controller.color_list[3])
    #    for crystal in controller.optics.crystal_list:
    #        ax.plot(crystal.boundary[:, 2], crystal.boundary[:, 1],
    #                linestyle='-', linewidth=3, label="crystal", color=controller.color_list[4])


class CrystalTower_miniSD_Scan:
    """
    This is simple implementation of the delay scan tower of the miniSD table.
    It is composed of an air-bearing stage and a few other stages.
    """

    def __init__(self,
                 channelCut1, crystal_loc1,
                 channelCut2, crystal_loc2):
        """
        Install the channel-cut crystal such that it moves with the motors
        :param crystal:
        """
        # Create the instance of each motors
        # This is the lower_most stage
        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12,
                             dimension=[250e3, 400e3]
                             )

        self.th1 = RotationMotor(upperLim=np.deg2rad(360),
                                 lowerLim=-np.deg2rad(360),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.01 / 1e12, )

        self.th2 = RotationMotor(upperLim=np.deg2rad(360),
                                 lowerLim=-np.deg2rad(360),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.01 / 1e12, )

        self.chi = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.x1 = LinearMotor(upperLim=12.5 * 1000,
                              lowerLim=-12.5 * 1000,
                              res=5,
                              backlash=100,
                              feedback_noise_level=1,
                              speed_um_per_ps=1 * 1000 / 1e12,
                              dimension=[70e3, 70e3])

        self.optics1 = channelCut1
        self.optics2 = channelCut2

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        # Define the installation path of the tower with respect to that of the lowest X stage
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.th1.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.th1.physical_deg0direction[1] = 1.0  #
        self.th1.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.th1.physical_rotation_axis[0] = 1.0
        self.th1.physical_rotation_center = np.zeros(3, dtype=np.float64)

        self.th2.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.th2.physical_deg0direction[1] = 1.0  #
        self.th2.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.th2.physical_rotation_axis[0] = 1.0
        self.th2.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation path of the theta stage with respect to the X stage
        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage
        th_stage_center[2] = -100e3
        self.th1.shift(displacement=th_stage_center)

        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage
        th_stage_center[2] = 100e3
        self.th2.shift(displacement=th_stage_center)

        # Define the installation path of the chi stage with respect to the x stage
        self.chi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.chi.physical_deg0direction[0] = 1.0  #
        self.chi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_axis[1] = 1.0
        self.chi.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.chi.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        chi_stage_center = np.zeros(3, dtype=np.float64)
        chi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        chi_stage_center[2] = -100e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.chi.shift(displacement=chi_stage_center)

        # Define the installation path of the x stage
        self.x1.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x1.physical_positive_direction[1] = 1.0  #

        x1_stage_center = np.zeros(3, dtype=np.float64)
        x1_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        x1_stage_center[2] = 100e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.x1.shift(displacement=chi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface[2] = -100e3
        crystalSurface += crystal_loc1
        self.optics1.shift(displacement=crystalSurface)

        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface[2] = 100e3
        crystalSurface += crystal_loc2
        self.optics2.shift(displacement=crystalSurface)

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)

        self.th1.shift(displacement=displacement, include_boundary=True)
        self.th2.shift(displacement=displacement)
        self.chi.shift(displacement=displacement, include_boundary=True)
        self.x1.shift(displacement=displacement)
        self.optics1.shift(displacement=displacement)
        self.optics2.shift(displacement=displacement)

    def th1_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th1.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th1.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th1.user_move_abs(target=target, getMotionTime=True)
        self.chi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th1.physical_rotation_center, include_boundary=True)
        self.optics1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th1.physical_rotation_center,
                                      include_boundary=True)

    def th2_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th2.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th2.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th2.user_move_abs(target=target, getMotionTime=True)
        self.x1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th2.physical_rotation_center, include_boundary=True)
        self.optics2.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th2.physical_rotation_center,
                                      include_boundary=True)

    def chi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.chi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.chi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.chi.user_move_abs(target=target, getMotionTime=True)
        self.optics1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.physical_rotation_center,
                                      include_boundary=True)

    def x1_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x1.physical_positive_direction * (target - self.x1.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x1.user_move_abs(target=target, getMotionTime=True)

        self.optics2.shift(displacement=displacement)


class Grating_tower:
    """
    This is a temporary implementation for the TG project.
    For more general purpose usage, one needs to use something different.
    """

    def __init__(self,
                 grating_1,
                 grating_m1):
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(360),
                                lowerLim=-np.deg2rad(360),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                feedback_noise_level=1e-9,
                                speed_rad_per_ps=0.01 / 1e12, )

        self.roll = RotationMotor(upperLim=np.deg2rad(5),
                                  lowerLim=-np.deg2rad(5),
                                  res=1e-6,
                                  backlash=np.deg2rad(-0.005),
                                  feedback_noise_level=1e-9,
                                  speed_rad_per_ps=0.1 / 1e12, )

        self.yaw = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.grating_1 = grating_1
        self.grating_m1 = grating_m1

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.pi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.physical_deg0direction[1] = 1.0  #
        self.pi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.physical_rotation_axis[0] = 1.0
        self.pi.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation path of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.pi.shift(displacement=pi_stage_center)

        self.roll.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.roll.physical_deg0direction[0] = 1.0  #
        self.roll.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.roll.physical_rotation_axis[2] = 1.0
        self.roll.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.roll.physical_rotation_center[1] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        roll_stage_center = np.zeros(3, dtype=np.float64)
        roll_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.roll.shift(displacement=roll_stage_center)

        self.yaw.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.yaw.physical_deg0direction[0] = 1.0  #
        self.yaw.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.yaw.physical_rotation_axis[2] = 1.0
        self.yaw.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.yaw.physical_rotation_center[1] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        yaw_stage_center = np.zeros(3, dtype=np.float64)
        yaw_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.roll.shift(displacement=yaw_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        self.grating_1.shift(displacement=crystalSurface)
        self.grating_m1.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.grating_1.shift(displacement=displacement, include_boundary=True)
        self.grating_m1.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.grating_1.shift(displacement=displacement, include_boundary=True)
        self.grating_m1.shift(displacement=displacement, include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.roll.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center, include_boundary=True)
        self.yaw.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center, include_boundary=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center,
                                         include_boundary=True)

    def roll_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.roll.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.roll.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.roll.user_move_abs(target=target, getMotionTime=True)
        self.yaw.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center, include_boundary=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.physical_rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.physical_rotation_center,
                                         include_boundary=True)

    def yaw_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.yaw.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.yaw.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.yaw.user_move_abs(target=target, getMotionTime=True)
        self.grating_1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.physical_rotation_center,
                                        include_boundary=True)
        self.grating_m1.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.physical_rotation_center,
                                         include_boundary=True)


class Mirror_tower1:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 mirror,
                 crystal_loc):
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                feedback_noise_level=1e-9,
                                speed_rad_per_ps=0.1 / 1e12, )

        self.optics = mirror

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.pi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.physical_deg0direction[0] = 1.0  #
        self.pi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.physical_rotation_axis[2] = 1.0
        self.pi.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.pi.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.pi.shift(displacement=pi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center, include_boundary=True)


class Mirror_tower2:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 mirror,
                 crystal_loc):
        # Create the instance of each motors

        self.z = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.yaw = RotationMotor(upperLim=np.deg2rad(5),
                                 lowerLim=-np.deg2rad(5),
                                 res=1e-6,
                                 backlash=np.deg2rad(-0.005),
                                 feedback_noise_level=1e-9,
                                 speed_rad_per_ps=0.1 / 1e12, )

        self.optics = mirror

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.z.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.z.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        z_stage_center = np.zeros(3, dtype=np.float64)
        self.z.shift(displacement=z_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.yaw.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.yaw.physical_deg0direction[0] = 1.0  #
        self.yaw.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.yaw.physical_rotation_axis[2] = 1.0
        self.yaw.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.yaw.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        yaw_stage_center = np.zeros(3, dtype=np.float64)
        yaw_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.yaw.shift(displacement=yaw_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def z_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.z.physical_positive_direction * (target - self.z.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.z.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.yaw.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def yaw_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.yaw.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.yaw.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.yaw.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.physical_rotation_center, include_boundary=True)


class Silicon_tower:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self,
                 crystal,
                 crystal_loc):
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.z = LinearMotor(upperLim=25000,
                             lowerLim=-25000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.roll = RotationMotor(upperLim=np.deg2rad(360),
                                  lowerLim=-np.deg2rad(360),
                                  res=1e-6,
                                  backlash=np.deg2rad(-0.005),
                                  feedback_noise_level=1e-9,
                                  speed_rad_per_ps=0.01 / 1e12, )

        self.pi = RotationMotor(upperLim=np.deg2rad(5),
                                lowerLim=-np.deg2rad(5),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                feedback_noise_level=1e-9,
                                speed_rad_per_ps=0.1 / 1e12, )

        self.optics = crystal

        self.all_mostors = [self.x, self.y, self.z, self.roll, self.pi]
        self.all_mostors_and_optics = [self.x, self.y, self.z, self.roll, self.pi, self.optics]

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.z.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.z.physical_positive_direction[2] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.z.shift(displacement=x_stage_center)

        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.roll.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.roll.physical_deg0direction[1] = 1.0  #
        self.roll.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.roll.physical_rotation_axis[0] = 1.0
        self.roll.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation path of the x stage
        roll_stage_center = np.zeros(3, dtype=np.float64)
        roll_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.roll.shift(displacement=roll_stage_center)

        self.pi.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.pi.physical_deg0direction[0] = 1.0  #
        self.pi.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.pi.physical_rotation_axis[1] = 1.0
        self.pi.physical_rotation_center = np.zeros(3, dtype=np.float64)
        self.pi.physical_rotation_center[0] = 70e3  # The rotation center of the chi stage is high in the air.

        # Define the installation path of the x stage
        pi_stage_center = np.zeros(3, dtype=np.float64)
        pi_stage_center[0] = 30 * 1000 + 20 * 1000 + 30e3  # The height of the x stage + the height of the y stage
        # + the height of the theta stage
        self.pi.shift(displacement=pi_stage_center)

        # Move the crystal such that the
        crystalSurface = np.zeros(3, dtype=np.float64)
        crystalSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        crystalSurface += crystal_loc
        self.optics.shift(displacement=crystalSurface)

    def y_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.x.shift(displacement=displacement, include_boundary=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def x_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.roll.shift(displacement=displacement, include_boundary=True)
        self.pi.shift(displacement=displacement, include_boundary=True)
        self.optics.shift(displacement=displacement, include_boundary=True)

    def roll_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.roll.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.roll.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.roll.user_move_abs(target=target, getMotionTime=True)
        self.pi.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.physical_rotation_center, include_boundary=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.physical_rotation_center,
                                     include_boundary=True)

    def pi_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.pi.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.pi.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.pi.user_move_abs(target=target, getMotionTime=True)
        self.optics.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.physical_rotation_center, include_boundary=True)


class TG_Sample_tower:
    """
    This class is probability only useful for the TG experiment.
    Therefore, when initializing this class, I do not allow for an arbitrary crystal path
    since there is almost no possibility of using this for a new application.
    """

    def __init__(self,
                 sample,
                 yag_sample,
                 yag1, yag2, yag3, ):
        """
        Install the channel-cut crystal such that it moves with the motors
        :param crystal:
        """
        # Create the instance of each motors

        self.x = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.y = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.z = LinearMotor(upperLim=12.5 * 1000,
                             lowerLim=-12.5 * 1000,
                             res=5,
                             backlash=100,
                             feedback_noise_level=1,
                             speed_um_per_ps=1 * 1000 / 1e12, )

        self.th = RotationMotor(upperLim=np.deg2rad(360),
                                lowerLim=-np.deg2rad(360),
                                res=1e-6,
                                backlash=np.deg2rad(-0.005),
                                feedback_noise_level=1e-9,
                                speed_rad_per_ps=0.01 / 1e12, )

        self.sample = sample
        self.yag_sample = yag_sample
        self.yag1 = yag1
        self.yag2 = yag2
        self.yag3 = yag3

        # ------------------------------------------
        # Change the motor configuration
        # ------------------------------------------
        self.x.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.x.physical_positive_direction[1] = 1.0  #
        # Define the installation path of the x stage
        x_stage_center = np.zeros(3, dtype=np.float64)
        self.x.shift(displacement=x_stage_center)

        self.y.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.y.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        y_stage_center = np.zeros(3, dtype=np.float64)
        y_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.y.shift(displacement=y_stage_center)

        self.z.physical_positive_direction = np.zeros(3, dtype=np.float64)
        self.z.physical_positive_direction[0] = 1.0  #
        # Define the installation path of the x stage
        z_stage_center = np.zeros(3, dtype=np.float64)
        z_stage_center[0] = 30 * 1000  # The height of the x stage.
        self.z.shift(displacement=z_stage_center)

        self.th.physical_deg0direction = np.zeros(3, dtype=np.float64)
        self.th.physical_deg0direction[1] = 1.0  #
        self.th.physical_rotation_axis = np.zeros(3, dtype=np.float64)
        self.th.physical_rotation_axis[0] = 1.0
        self.th.physical_rotation_center = np.zeros(3, dtype=np.float64)

        # Define the installation path of the x stage
        th_stage_center = np.zeros(3, dtype=np.float64)
        th_stage_center[0] = 30 * 1000 + 20 * 1000  # The height of the x stage + the height of the y stage
        self.th.shift(displacement=th_stage_center)

        # Move the crystal such that the
        sampleSurface = np.zeros(3, dtype=np.float64)
        sampleSurface[0] = 30 * 1000 + 20 * 1000 + 30e3 + 20e3
        self.sample.shift(displacement=sampleSurface)

        yag_sample_location = np.zeros(3)
        self.yag_sample.shift(displacement=yag_sample_location)

        yag1_location = np.zeros(3)
        self.yag1.shift(displacement=yag1_location)
        yag2_location = np.zeros(3)
        self.yag2.shift(displacement=yag2_location)
        yag3_location = np.zeros(3)
        self.yag3.shift(displacement=yag3_location)

        # Define the color for the device visualization
        self.color_list = ['red', 'brown', 'yellow', 'purple', 'black']

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """

        # Get the displacement vector for the motion
        displacement = self.x.physical_positive_direction * (target - self.x.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.x.user_move_abs(target=target, getMotionTime=True)
        self.y.shift(displacement=displacement, include_boundary=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def y_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.y.physical_positive_direction * (target - self.y.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.y.user_move_abs(target=target, getMotionTime=True)
        self.z.shift(displacement=displacement, include_boundary=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def z_umv(self, target):
        # Get the displacement vector for the motion
        displacement = self.z.physical_positive_direction * (target - self.z.control_location)

        # Shift all the motors and crystals with it
        motion_time = self.z.user_move_abs(target=target, getMotionTime=True)
        self.th.shift(displacement=displacement, include_boundary=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
        self.yag1.shift(displacement=displacement, include_boundary=True)
        self.yag2.shift(displacement=displacement, include_boundary=True)
        self.yag3.shift(displacement=displacement, include_boundary=True)

    def th_umv(self, target):
        # Get the displacement vector for the motion
        displacement = (target - self.th.control_location)

        # Get the rotation matrix for the stages above the rotation stage
        rotMat = util.get_rotmat_around_axis(angleRadian=displacement, axis=self.th.physical_rotation_axis)

        # Shift all the motors and crystals with it
        motion_time = self.th.user_move_abs(target=target, getMotionTime=True)
        self.sample.shift(displacement=displacement, include_boundary=True)
        self.yag_sample.shift(displacement=displacement, include_boundary=True)
