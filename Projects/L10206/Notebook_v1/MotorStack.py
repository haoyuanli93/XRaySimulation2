import numpy as np

from XRaySimulation import util
from XRaySimulation.Machine import Motors


class CrystalTower_x_y_theta_chi:
    def __init__(self, crystal, crystal_loc):
        # Create the instance of each motors and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=55e3, dimension=[100e3, 100e3])
        self.x = Motors.get_motors_with_model_for_axis(model="XA10A")
        self.y = Motors.get_motors_with_model_for_axis(model="ZA10A")
        self.adaptor2 = Motors.AdaptorPlate(height=10e3, dimension=[70e3, 70e3])
        self.th = Motors.get_motors_with_model_for_axis(model="RA10A")
        self.chi = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.adaptor3 = Motors.AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.optics = crystal

        # Create the list of all components in this tower
        self.optics.shift(displacement=self.adaptor3.top_mount_pos - np.copy(crystal_loc))
        self.all_obj = [self.adaptor3, self.optics]

        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.chi)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.th)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.x, self.y, self.th, self.chi]
        if self.optics.type == "Channel cut with two surfaces":
            self.obj_to_plot = [self.x, self.y, self.th, self.chi, ] + self.optics.crystal_list
            self.all_optics = self.optics.crystal_list
        elif self.optics.type == "Crystal: Bragg Reflection":
            self.obj_to_plot = [self.x, self.y, self.th, self.chi, self.optics]
            self.all_optics = [self.optics, ]

    def x_umv(self, target):
        """
        If one moves the x stage, then one moves the
        y stage, theta stage, chi stage, crystal
        together with it.

        :param target:
        :return:
        """
        # Shift all the motors and crystals with it
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def y_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umvr(self, delta):
        self.y_umv(target=self.y.control_location * 1.0 + delta)

    def th_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.th.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
        return motion_time

    def th_umvr(self, delta):
        self.th_umv(target=self.th.control_location * 1.0 + delta)

    def chi_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.chi.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th.rotation_center)
        return motion_time

    def chi_umvr(self, delta):
        self.chi_umv(target=self.chi.control_location * 1.0 + delta)


class Tower_x_y_pi:
    def __init__(self, mirror):
        # Create the instance of each motors and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=55e3, dimension=[100e3, 100e3])
        self.x = Motors.get_motors_with_model_for_axis(model="XA10A")
        self.y = Motors.get_motors_with_model_for_axis(model="ZA10A")
        self.adaptor2 = Motors.AdaptorPlate(height=10e3, dimension=[70e3, 70e3])
        self.pi = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="x")
        self.adaptor3 = Motors.AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Create the list of all components in this tower
        self.optics.shift(displacement=self.adaptor3.top_mount_pos - np.copy(self.optics.surface_point))
        self.all_obj = [self.adaptor3, self.optics]

        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.pi)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.x, self.y, self.pi]
        self.obj_to_plot = [self.x, self.y, self.pi, self.optics]
        self.all_optics = [self.optics, ]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def y_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umvr(self, delta):
        self.y_umv(target=self.y.control_location * 1.0 + delta)

    def pi_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.pi.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center)
        return motion_time

    def pi_umvr(self, delta):
        self.pi_umv(target=self.pi.control_location * 1.0 + delta)


class CrystalTower_miniSD_Scan:
    """
    This is simple implementation of the delay scan tower of the miniSD table.
    It is composed of an air-bearing stage and a few other stages.
    """

    def __init__(self,
                 channelCut1, crystal_loc1,
                 channelCut2, crystal_loc2):
        # Create the instance of each motors and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=10e3, dimension=[307e3, 270e3])
        self.delay = Motors.get_motors_with_model_for_axis(model="XA10A")
        self.adaptor2 = Motors.AdaptorPlate(height=10e3, dimension=[233e3, 288e3])
        self.th1 = Motors.get_motors_with_model_for_axis(model="RA10A")
        self.th2 = Motors.get_motors_with_model_for_axis(model="RA10A")
        self.chi = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="z")
        self.x = Motors.get_motors_with_model_for_axis(model="XA07A")
        self.adaptor3 = Motors.AdaptorPlate(height=39e3, dimension=[70e3, 70e3])
        self.adaptor4 = Motors.AdaptorPlate(height=44e3, dimension=[70e3, 70e3])
        self.optics1 = channelCut1
        self.optics2 = channelCut2

        # Create the list of all components in this tower
        self.optics1.shift(displacement=self.adaptor3.top_mount_pos - crystal_loc1)
        self.optics2.shift(displacement=self.adaptor4.top_mount_pos - crystal_loc2)

        self.tower1 = [self.adaptor3, self.optics1]
        self.tower1 = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.tower1, motor_or_adaptor=self.x)
        self.tower1 = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.tower1, motor_or_adaptor=self.th1)

        self.tower2 = [self.adaptor4, self.optics2]
        self.tower2 = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.tower2, motor_or_adaptor=self.chi)
        self.tower2 = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.tower2, motor_or_adaptor=self.th2)

        # Adjust the relative position between the two towers with respect to the adaptor2
        displacement = np.array([0., 0., -89e3]) + self.adaptor2.top_mount_pos - self.tower1[0].bottom_mount_pos
        for item in self.tower1:
            # print(item)
            item.shift(displacement=displacement)

        displacement = np.array([0., 0., 89e3]) + self.adaptor2.top_mount_pos - self.tower2[0].bottom_mount_pos
        for item in self.tower2:
            item.shift(displacement=displacement)

        self.all_obj = [self.adaptor2, ] + self.tower1 + self.tower2
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.delay)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        # Define a holder that contains all the motor objects
        self.all_motors = [self.delay, self.th1, self.th2, self.chi, self.x]
        self.obj_to_plot = ([self.delay, self.th1, self.th2, self.chi, self.x] +
                            self.optics1.crystal_list + self.optics2.crystal_list)
        self.all_optics = self.optics1.crystal_list + self.optics2.crystal_list

    def delay_umv(self, target):
        motion_time, displacement = self.delay.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def delay_umvr(self, delta):
        self.delay_umv(target=self.delay.control_location * 1.0 + delta)

    def th1_umv(self, target):
        motion_time, rotMat = self.th1.user_move_abs(target=target)
        for item in self.tower1[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th1.rotation_center)
        return motion_time

    def th1_umvr(self, delta):
        self.th1_umv(target=self.th1.control_location * 1.0 + delta)

    def th2_umv(self, target):
        motion_time, rotMat = self.th2.user_move_abs(target=target)
        for item in self.tower2[1:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.th2.rotation_center)
        return motion_time

    def th2_umvr(self, delta):
        self.th2_umv(target=self.th2.control_location * 1.0 + delta)

    def chi_umv(self, target):
        motion_time, rotMat = self.chi.user_move_abs(target=target)
        for item in self.tower2[2:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.chi.rotation_center)
        return motion_time

    def chi_umvr(self, delta):
        self.chi_umv(target=self.chi.control_location * 1.0 + delta)

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.tower1[1:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)


class Grating_tower:
    """
    This is a temporary implementation for the TG project.
    For more general purpose usage, one needs to use something different.
    """

    def __init__(self,
                 grating_1,
                 grating_m1):
        # Create the instance of each motor and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=14e3, dimension=[70e3, 70e3])
        self.x = Motors.get_motors_with_model_for_axis(model="XA07A")
        self.adaptor2 = Motors.L_Bracket(height=159e3, dimension=[70e3, 70e3])
        self.y = Motors.get_motors_with_model_for_axis(model="XA07A", axis='y')
        self.pi = Motors.get_motors_with_model_for_axis(model='RA05A', axis='x')
        (self.roll, self.yaw) = Motors.get_motors_with_model_for_axis(model='SA05A-R2S01', axis='x')
        self.adaptor3 = Motors.AdaptorPlate(height=50e3, dimension=[50e3, 50e3])

        # Install adaptors and motors
        # Rotate the adaptor 3 to install it on the roll pi motor
        rot_mat = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
        self.adaptor3.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adaptor3.bottom_mount_pos))

        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=[self.adaptor3, ],
                                                                  motor_or_adaptor=self.yaw)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.pi)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        # Install the two gratings
        displacement = self.adaptor3.top_mount_pos - grating_1.surface_point
        grating_1.shift(displacement=displacement)
        displacement = self.adaptor3.top_mount_pos - grating_m1.surface_point
        grating_m1.shift(displacement=displacement)
        self.grating_1 = grating_1
        self.grating_m1 = grating_m1

        # Add the gratings to the grating tower
        self.all_obj += [self.grating_1, self.grating_m1]

        self.all_motors = [self.x, self.y, self.pi, self.roll, self.yaw]
        self.obj_to_plot = [self.x, self.y, self.pi, self.roll, self.yaw, self.grating_1]
        self.all_optics = [self.grating_1, self.grating_m1]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umvr(self, delta):
        self.y_umv(target=self.y.control_location * 1.0 + delta)

    def pi_umv(self, target):
        motion_time, rotMat = self.pi.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center)
        return motion_time

    def pi_umvr(self, delta):
        self.pi_umv(target=self.pi.control_location * 1.0 + delta)

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def roll_umvr(self, delta):
        self.roll_umv(target=self.roll.control_location * 1.0 + delta)

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[6:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time

    def yaw_umvr(self, delta):
        self.yaw_umv(target=self.yaw.control_location * 1.0 + delta)


class Mirror_tower1:
    """
    This is just a simple realization of the most commonly used crystal tower in the miniSD device.
    Even though initially, I was thinking that I should implement some function that
    are more general than this.
    In the end, I realized that it is beyond my current capability.
    Therefore, I guess it is easier for me to just get something more concrete and to give this
    to Khaled sooner.
    """

    def __init__(self, mirror):
        # Create the instance of each motors
        self.adapter1 = Motors.AdaptorPlate(height=20e3, dimension=[100e3, 200e3])
        self.z = Motors.get_motors_with_model_for_axis(model="XA10A-L101", axis='z')
        self.x = Motors.get_motors_with_model_for_axis(model="XA10A", axis='x')
        self.adapter2 = Motors.AdaptorPlate(height=10e3, dimension=[100e3, 250e3])
        self.roll = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=96e3, axis="z")
        self.adapter3 = Motors.L_Bracket(height=10e3, dimension=[70e3, 70e3])
        self.yaw = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="x")
        self.adapter4 = Motors.AdaptorPlate(height=70e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Install mirror to the adaptor 4
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0., 0., 1]))
        self.optics.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.optics.surface_point))
        self.adapter4.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adapter4.bottom_mount_pos))
        self.yaw.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yaw.bottom_mount_pos))

        displacement = self.adapter4.top_mount_pos + np.array([86.3e3 + 5e3, 0, 0]) - self.optics.surface_point
        self.optics.shift(displacement=displacement)

        self.all_obj = [self.adapter4, self.optics]
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.yaw)

        # Adjust the dimension of adaptor 3
        self.adapter3.top_mount_pos += np.array([0, 75.35e3, 0])
        self.adapter3.top_mount_pos[0] = 19e3
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter3)

        # Install other components
        # Adjust the dimension of adaptor 2
        self.adapter2.bottom_mount_pos[2] = -75e3
        self.adapter2.top_mount_pos[2] = 125e3 - 35e3
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter1)

        self.all_motors = [self.z, self.x, self.roll, self.yaw]
        self.obj_to_plot = [self.z, self.x, self.roll, self.yaw, self.optics]
        self.all_optics = [self.optics, ]

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umvr(self, delta):
        self.z_umv(target=self.z.control_location * 1.0 + delta)

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def roll_umvr(self, delta):
        self.roll_umv(target=self.roll.control_location * 1.0 + delta)

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time

    def yaw_umvr(self, delta):
        self.yaw_umv(target=self.yaw.control_location * 1.0 + delta)


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
                 mirror):
        # Create the instance of each motors
        self.adapter1 = Motors.AdaptorPlate(height=20e3, dimension=[100e3, 200e3])
        self.z = Motors.get_motors_with_model_for_axis(model="XA10A-L101", axis='z')
        self.x = Motors.get_motors_with_model_for_axis(model="XA10A", axis='x')
        self.adapter2 = Motors.AdaptorPlate(height=10e3, dimension=[100e3, 250e3])
        self.roll = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=96e3, axis="z")
        self.adapter3 = Motors.L_Bracket(height=10e3, dimension=[70e3, 70e3])
        self.yaw = Motors.get_motors_with_model_for_axis(model="SA07A", rot_center_height=70e3, axis="x")
        self.adapter4 = Motors.AdaptorPlate(height=70e3, dimension=[70e3, 70e3])
        self.optics = mirror

        # Install mirror to the adaptor 4
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0., 0., 1]))
        self.optics.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.optics.surface_point))
        self.adapter4.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adapter4.bottom_mount_pos))
        self.yaw.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yaw.bottom_mount_pos))

        displacement = self.adapter4.top_mount_pos + np.array([86.3e3 + 5e3, 0, 0]) - self.optics.surface_point
        self.optics.shift(displacement=displacement)

        self.all_obj = [self.adapter4, self.optics]
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.yaw)

        # Adjust the dimension of adaptor 3
        self.adapter3.top_mount_pos += np.array([0, 75.35e3, 0])
        self.adapter3.top_mount_pos[0] = 19e3
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter3)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)

        # print("test: current surface point", controller.optics.surface_point)
        # Everything above is copied from the mirror tower 1 class
        # Here I need to rotate the components to get the correct geometry
        # First rotate around the y rotation_axis
        rot_mat = np.array([[1.0, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1], ])
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.roll.bottom_mount_pos)
        # print("test: current surface point", controller.optics.surface_point)

        # Adjust the dimension of adaptor 2
        self.adapter2.bottom_mount_pos[2] = -75e3
        self.adapter2.top_mount_pos[2] = 125e3 - 35e3
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adapter1)

        self.all_motors = [self.z, self.x, self.roll, self.yaw]
        self.obj_to_plot = [self.z, self.x, self.roll, self.yaw, self.optics]
        self.all_optics = [self.optics, ]

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umvr(self, delta):
        self.z_umv(target=self.z.control_location * 1.0 + delta)

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[5:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def roll_umvr(self, delta):
        self.roll_umv(target=self.roll.control_location * 1.0 + delta)

    def yaw_umv(self, target):
        motion_time, rotMat = self.yaw.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.yaw.rotation_center)
        return motion_time

    def yaw_umvr(self, delta):
        self.yaw_umv(target=self.yaw.control_location * 1.0 + delta)


class Silicon_tower:
    def __init__(self, crystal):
        # Create the instance of each motors and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=22.7e3, dimension=[100e3, 100e3])
        self.y = Motors.get_motors_with_model_for_axis(model="ZA10A")
        self.z = Motors.get_motors_with_model_for_axis(model="XA10A", axis='z')
        self.x = Motors.get_motors_with_model_for_axis(model="XA10A")

        self.adaptor2 = Motors.AdaptorPlate(height=10e3, dimension=[75e3, 200e3])
        self.adaptor2.bottom_mount_pos[2] = 100e3 - 35e3
        self.adaptor2.top_mount_pos[2] = -(100e3 - 35e3)
        self.adaptor2.top_mount_pos[0] = 0.0

        self.adaptor3 = Motors.AdaptorPlate(height=245e3, dimension=[70e3, 70e3])
        tilt_angle3 = np.deg2rad(10)
        self.adaptor3.top_mount_dir = np.array([np.cos(np.deg2rad(tilt_angle3)),
                                                0, np.sin(np.deg2rad(tilt_angle3))])

        (self.roll, self.pi) = Motors.get_motors_with_model_for_axis(model='SA05A-R2S01', axis='y')
        self.adaptor4 = Motors.AdaptorPlate(height=50e3, dimension=[20e3, 20e3])
        self.optics = crystal

        # Install the crystal on the top of the first adaptor
        self.optics.shift(displacement=self.adaptor4.top_mount_pos - self.optics.surface_point)
        self.all_obj = [self.adaptor4, self.optics]
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.pi)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.roll)

        # Rotate around the x rotation_axis such that it matches the angle
        rot_mat = np.array([[np.cos(tilt_angle3), 0, np.sin(tilt_angle3)],
                            [0, 1, 0],
                            [-np.sin(tilt_angle3), 0, np.cos(tilt_angle3)]])
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.roll.bottom_mount_pos)

        # Install the setup on adaptor 3
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor3)

        # Rotate everything around the z rotation_axis by 180 deg
        rot_mat = np.eye(3)
        rot_mat[0, 0] = -1
        rot_mat[1, 1] = -1
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=self.adaptor3.bottom_mount_pos)

        # Install the other components
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        self.all_motors = [self.y, self.z, self.x, self.roll, self.pi]
        self.obj_to_plot = [self.y, self.z, self.x, self.roll, self.pi, self.optics]
        self.all_optics = [self.optics, ]

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umvr(self, delta):
        self.y_umv(target=self.y.control_location * 1.0 + delta)

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[3:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umvr(self, delta):
        self.z_umv(target=self.z.control_location * 1.0 + delta)

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def roll_umv(self, target):
        motion_time, rotMat = self.roll.user_move_abs(target=target)
        for item in self.all_obj[7:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.roll.rotation_center)
        return motion_time

    def roll_umvr(self, delta):
        self.roll_umv(target=self.roll.control_location * 1.0 + delta)

    def pi_umv(self, target):
        motion_time, rotMat = self.pi.user_move_abs(target=target)
        for item in self.all_obj[8:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=self.pi.rotation_center)
        return motion_time

    def pi_umvr(self, delta):
        self.pi_umv(target=self.pi.control_location * 1.0 + delta)


class TG_Sample_tower:
    def __init__(self, sample, yag_sample, yag1, yag2, yag3):

        # Create the instance of each motors and adaptors
        self.adaptor1 = Motors.AdaptorPlate(height=10e3, dimension=[100e3, 100e3])
        self.x = Motors.get_motors_with_model_for_axis(model="UTS100CC")
        self.adaptor2 = Motors.AdaptorPlate(height=24.3e3, dimension=[100e3, 100e3])
        self.y = Motors.get_motors_with_model_for_axis(model="ZA10A")
        self.z = Motors.get_motors_with_model_for_axis(model="XA10A", axis='z')
        self.adaptor3 = Motors.AdaptorPlate(height=30e3, dimension=[100e3, 100e3])
        self.adaptor4 = Motors.L_Bracket(height=35e3, dimension=[50e3, 50e3])
        self.th = Motors.get_motors_with_model_for_axis(model="RA05A", axis='y')
        self.adaptor5 = Motors.AdaptorPlate(height=72.7e3, dimension=[10e3, 10e3])

        self.sample = sample
        self.yag_sample = yag_sample
        self.yag1 = yag1
        self.yag2 = yag2
        self.yag3 = yag3

        # Currently the samples are pointing z rotation_axis, rotate them such that they are facing y rotation_axis
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0, 1, 0], dtype=np.float64))
        self.sample.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.sample.surface_point))
        self.yag_sample.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.yag_sample.surface_point))

        # Install sample and sample_yag on the adaptor 5
        displacement = np.copy(self.adaptor5.top_mount_pos - self.sample.surface_point)
        self.sample.shift(displacement=displacement)
        self.yag_sample.shift(displacement=displacement + np.array([0., 10e3, 0]))

        # Rotate around the z rotation_axis to make the component horizontal
        self.all_obj = [self.adaptor5, self.yag_sample, self.sample]
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(self.all_obj, self.th)
        rot_mat = util.get_rotmat_around_axis(angleRadian=np.pi / 2, axis=np.array([0, 0, 1], dtype=np.float64))
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.th.bottom_mount_pos))

        self.all_obj = Motors.install_motors_on_motor_or_adaptors(self.all_obj, self.adaptor4)

        rot_mat = util.get_rotmat_around_axis(angleRadian=np.deg2rad(-5), axis=np.array([1, 0, 0], dtype=np.float64))
        for item in self.all_obj:
            item.rotate_wrt_point(rot_mat=rot_mat, ref_point=np.copy(self.adaptor4.bottom_mount_pos))

        # Assemble the small sample tower to the big sample and yag tower
        displacement = np.array([0, -77.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos)
        for item in self.all_obj:
            item.shift(displacement=displacement)

        self.yag1.shift(displacement=np.array([35.7e3, 77.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.yag2.shift(displacement=np.array([35.7e3, 87.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.yag3.shift(displacement=np.array([45.7e3, 87.5e3, -70.35e3]) + np.copy(self.adaptor3.top_mount_pos))
        self.all_obj = [self.adaptor3, self.yag1, self.yag2, self.yag3] + self.all_obj

        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.z)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.y)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor2)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj, motor_or_adaptor=self.x)
        self.all_obj = Motors.install_motors_on_motor_or_adaptors(motor_tower=self.all_obj,
                                                                  motor_or_adaptor=self.adaptor1)

        self.all_motors = [self.x, self.y, self.z, self.th]
        self.obj_to_plot = [self.x, self.y, self.z, self.th, self.sample,
                            self.yag_sample, self.yag1, self.yag2, self.yag3]
        self.all_optics = [self.sample, self.yag_sample, self.yag1, self.yag2, self.yag3]

    def x_umv(self, target):
        motion_time, displacement = self.x.user_move_abs(target=target)
        for item in self.all_obj[2:]:
            item.shift(displacement=displacement)
        return motion_time

    def x_umvr(self, delta):
        self.x_umv(target=self.x.control_location * 1.0 + delta)

    def y_umv(self, target):
        motion_time, displacement = self.y.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def y_umvr(self, delta):
        self.y_umv(target=self.y.control_location * 1.0 + delta)

    def z_umv(self, target):
        motion_time, displacement = self.z.user_move_abs(target=target)
        for item in self.all_obj[4:]:
            item.shift(displacement=displacement)
        return motion_time

    def z_umvr(self, delta):
        self.z_umv(target=self.z.control_location * 1.0 + delta)

    def th_umv(self, target):
        # Shift all the motors and crystals with it
        motion_time, rotMat = self.th.user_move_abs(target=target)
        for item in self.all_obj[11:]:
            item.rotate_wrt_point(rot_mat=rotMat, ref_point=np.copy(self.th.rotation_center))
        return motion_time

    def th_umvr(self, delta):
        self.th_umv(target=self.th.control_location * 1.0 + delta)
