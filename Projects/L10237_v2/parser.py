import os
import numpy as np
import sys

sys.path.append("../../../../XRaySimulation")

from XRaySimulation import util


def parser(controller, motor_name, target, relative=True, XPPcontroller=False):
    """
    This is a temporary solution.

    The acceptable

    :param controller:
    :param commandline:
    :return:
    """
    motor_list = {'t1x', 't2x'}
    if motor_name == "t1x":
        if XPPcontroller:
            os.system("x.t1x.umvr({})".format(target))
        else:
            displacement = np.array([0, target, 0], )
            controller.cc1.shift(displacement)

    elif motor_name == "t1th":
        if XPPcontroller:
            os.system("x.t1th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.cc1.crystal_list[0].surface_point)
            controller.cc1.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "t2x":
        if XPPcontroller:
            os.system("x.t2x.umvr({})".format(target))
        else:
            displacement = np.array([0, target, 0], )
            controller.vcc1.shift(displacement)

    elif motor_name == "t2th":
        if XPPcontroller:
            os.system("x.t2th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.vcc1.crystal_list[0].surface_point)
            controller.vcc1.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "t3x":
        if XPPcontroller:
            os.system("x.t3x.umvr({})".format(target))
        else:
            displacement = np.array([0, target, 0], )
            controller.vcc2.shift(displacement)

    elif motor_name == "t3th":
        if XPPcontroller:
            os.system("x.t3th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.vcc2.crystal_list[0].surface_point)
            controller.vcc2.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "delay":
        if XPPcontroller:
            os.system("x.delay.umvr({})".format(target))
        else:
            displacement = np.array([0, target, 0], )
            controller.vcc3.shift(displacement)
            controller.vcc4.shift(displacement)

    elif motor_name == "t4th":
        if XPPcontroller:
            os.system("x.t4th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.vcc3.crystal_list[0].surface_point)
            controller.vcc3.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "t5th":
        if XPPcontroller:
            os.system("x.t5th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.vcc4.crystal_list[0].surface_point)
            controller.vcc4.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "t6x":
        if XPPcontroller:
            os.system("x.t6x.umvr({})".format(target))
        else:
            displacement = np.array([0, target, 0], )
            controller.cc2.shift(displacement)

    elif motor_name == "t6th":
        if XPPcontroller:
            os.system("x.t6th.umvr({})".format(target))
        else:
            rotMat = util.rot_mat_in_xz_plane(theta=target)
            ref_point = np.copy(controller.cc2.crystal_list[0].surface_point)
            controller.cc2.rotate_wrt_point(rot_mat=rotMat, ref_point=ref_point)

    elif motor_name == "show_cc":
        if XPPcontroller:
            os.system("x.show_cc()")
        else:
            controller.show_cc()

    elif motor_name == "show_vcc":
        if XPPcontroller:
            os.system("x.show_vcc()")
        else:
            controller.show_vcc()

    elif motor_name == "show_both":
        if XPPcontroller:
            os.system("x.show_both()")
        else:
            controller.show_both()

    elif motor_name == "show_neither":
        if XPPcontroller:
            os.system("x.show_neither()")
        else:
            controller.show_neither()

    else:
        print("The command is not defined. Please read the source code of this function.")
        return 1

    return 0
