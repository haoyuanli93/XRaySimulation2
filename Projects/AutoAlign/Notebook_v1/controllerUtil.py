import sys
import time

from XRaySimulation import RayTracing, Alignment

sys.path.append("../../../")

import numpy as np

from XRaySimulation import util

si220 = {'d': 1.9201 * 1e-4,
         "chi0": complex(-0.80575E-05, 0.10198E-06),
         "chih_sigma": complex(0.48909E-05, -0.98241E-07),
         "chihbar_sigma": complex(0.48909E-05, -0.98241E-07),
         "chih_pi": complex(0.40482E-05, -0.80452E-07),
         "chihbar_pi": complex(0.40482E-05, -0.80452E-07),
         }

dia111 = {'d': 2.0593 * 1e-4,
          "chi0": complex(-0.12067E-04, 0.82462E-08),
          "chih_sigma": complex(0.43910E-05, -0.57349E-08),
          "chihbar_sigma": complex(0.43910E-05, -0.57349E-08),
          "chih_pi": complex(0.37333E-05, -0.48247E-08),
          "chihbar_pi": complex(0.37333E-05, -0.48247E-08),
          }


def get_raytracing_trajectory(controller,
                              path="mono",
                              get_path_length='True',
                              virtual_sample_plane=None):
    if path == "cc":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.g2.grating_m1, ])
        trajectory, kout, pathlength = RayTracing.get_lightpath(device_list=defice_list,
                                                                kin=controller.gaussian_pulse.k0,
                                                                initial_point=controller.gaussian_pulse.x0,
                                                                final_plane_point=
                                                                np.copy(
                                                                    controller.sample.yag1.surface_point),
                                                                final_plane_normal=
                                                                np.copy(controller.sample.yag1.normal))

    elif path == "vcc":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_1]
                       + controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                       + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list
                       + [controller.g2.grating_1, ])

        trajectory, kout, pathlength = RayTracing.get_lightpath(device_list=defice_list,
                                                                kin=controller.gaussian_pulse.k0,
                                                                initial_point=controller.gaussian_pulse.x0,
                                                                final_plane_point=
                                                                np.copy(
                                                                    controller.sample.yag1.surface_point),
                                                                final_plane_normal=
                                                                np.copy(controller.sample.yag1.normal))

    elif path == "mono":
        defice_list = [controller.mono_t1.optics, controller.mono_t2.optics, ]

        trajectory, kout, pathlength = RayTracing.get_lightpath(device_list=defice_list,
                                                                kin=controller.gaussian_pulse.k0,
                                                                initial_point=controller.gaussian_pulse.x0,
                                                                final_plane_point=
                                                                np.copy(
                                                                    controller.sample.yag1.surface_point),
                                                                final_plane_normal=
                                                                np.copy(controller.sample.yag1.normal))

    else:
        print("Warning, the specified path option is not defined.")
        trajectory = 0
        kout = 0
        pathlength = 0

    if get_path_length:
        return trajectory, kout, pathlength
    else:
        return trajectory, kout


def align_xpp_mono(controller):
    # Get the geometry bragg angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=dia111['d'])

    # Align the first crystal
    (rot_mat1, fwhm1, kout1, angle_adjust1, angles1, reflectivity1
     ) = Alignment.align_crystal_around_axis(crystal=controller.mono_t1.optics,
                                             kin=controller.gaussian_pulse.k0,
                                             initial_angle=-bragg,
                                             rotation_axis=controller.mono_t1.th.rotation_axis,
                                             bandwidth_keV=0.5e-3,
                                             rot_crystal=False)

    # Move the crystal to the target path
    _ = controller.mono_t1.th_umv(target=angle_adjust1)
    # print("haha")

    # Align the second crystal
    kin = RayTracing.get_kout_single_device(device=controller.mono_t1.optics, kin=controller.gaussian_pulse.k0)
    (rot_mat2, fwhm2, kout2, angle_adjust2, angles2, reflectivity2
     ) = Alignment.align_crystal_around_axis(crystal=controller.mono_t2.optics,
                                             kin=kin,
                                             initial_angle=-bragg,
                                             rotation_axis=controller.mono_t2.th.rotation_axis,
                                             bandwidth_keV=0.5e-3,
                                             rot_crystal=False)
    _ = controller.mono_t2.th_umv(target=angle_adjust2)

    controller.mono_t1_rocking = [angles1 - angle_adjust1, reflectivity1]
    controller.mono_t2_rocking = [angles2 - angle_adjust2, reflectivity2]

    # Adjust the path of the XPP mono such that the exit X x-ray pulse is at the (0,0, ...)
    # on the second crystal
    trajectory, kout, _ = controller.get_raytracing_trajectory(path='mono')
    # Get the ideal location of the second crsytal
    dir = kout[-2] / np.linalg.norm(kout[-2])
    location = dir * (0 - controller.gaussian_pulse.x0[1]) / dir[1]
    # print(location)
    location += trajectory[-3]
    displacement = location - controller.mono_t2.optics.surface_point
    for item in controller.mono_t2.all_obj:
        item.shift(displacement=np.copy(displacement))

    return kout1, kout2


def align_miniSD_SASE(controller):
    # Get the kout after the XPP mono
    _, kout, _ = RayTracing.get_lightpath(
        device_list=[controller.mono_t1.optics, controller.mono_t2.optics],
        kin=controller.gaussian_pulse.k0,
        initial_point=controller.gaussian_pulse.x0,
        final_plane_point=np.array([0, 0, 10e6]),
        final_plane_normal=np.array([0, 0, -1]))
    kout = np.copy(kout[-1])

    # Get the geometry bragg angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=si220['d'])

    # ----------------------------------------
    # Align the CC branch
    kin = np.copy(kout + controller.g1.grating_m1.momentum_transfer)
    res1 = Alignment.align_crystal_around_axis(crystal=controller.t1.optics,
                                               kin=kin,
                                               initial_angle=bragg - controller.t1.th.user_getPosition(),
                                               rotation_axis=controller.t1.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)

    # Move the crystal to the target path
    _ = controller.t1.th_umv(target=res1[3] + controller.t1.th.user_getPosition())

    # Align the second CC
    device_list = controller.t1.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res6 = Alignment.align_crystal_around_axis(crystal=controller.t6.optics,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t6.th.user_getPosition(),
                                               rotation_axis=controller.t6.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    _ = controller.t6.th_umv(target=res6[3] + controller.t6.th.user_getPosition())

    # ----------------------------------------
    # Align the VCC branch
    kin = np.copy(kout + controller.g1.grating_1.momentum_transfer)
    res2 = Alignment.align_crystal_around_axis(crystal=controller.t2.optics,
                                               kin=kin,
                                               initial_angle=bragg - controller.t2.th.user_getPosition(),
                                               rotation_axis=controller.t2.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)

    # Move the crystal to the target path
    _ = controller.t2.th_umv(target=res2[3] + controller.t2.th.user_getPosition())

    # Align the second VCC
    device_list = controller.t2.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res3 = Alignment.align_crystal_around_axis(crystal=controller.t3.optics,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t3.th.user_getPosition(),
                                               rotation_axis=controller.t3.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    _ = controller.t3.th_umv(target=res3[3] + controller.t3.th.user_getPosition())

    # Align the third VCC
    device_list = controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res4 = Alignment.align_crystal_around_axis(crystal=controller.t45.optics1,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t45.th1.user_getPosition(),
                                               rotation_axis=controller.t45.th1.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    _ = controller.t45.th1_umv(target=res4[3] + controller.t45.th1.user_getPosition())

    # Align the 4th VCC
    device_list = (controller.t2.optics.crystal_list +
                   controller.t3.optics.crystal_list +
                   controller.t45.optics1.crystal_list)
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res5 = Alignment.align_crystal_around_axis(crystal=controller.t45.optics2,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t45.th2.user_getPosition(),
                                               rotation_axis=controller.t45.th2.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    _ = controller.t45.th2_umv(target=res5[3] + controller.t45.th2.user_getPosition())


def get_miniSD_rocking(controller):
    # Get the kout after the XPP mono
    _, kout, _ = RayTracing.get_lightpath(
        device_list=[controller.mono_t1.optics, controller.mono_t2.optics],
        kin=controller.gaussian_pulse.k0,
        initial_point=controller.gaussian_pulse.x0,
        final_plane_point=np.array([0, 0, 10e6]),
        final_plane_normal=np.array([0, 0, -1]))
    kout = np.copy(kout[-1])

    # Get the geometry bragg angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=si220['d'])

    # ----------------------------------------
    # Align the CC branch
    kin = np.copy(kout + controller.g1.grating_m1.momentum_transfer)
    res1 = Alignment.align_crystal_around_axis(crystal=controller.t1.optics,
                                               kin=kin,
                                               initial_angle=bragg - controller.t1.th.user_getPosition(),
                                               rotation_axis=controller.t1.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)

    # Move the crystal to the target path
    controller.t1_rocking[:] = [np.copy(res1[-2]), np.copy(res1[-1])]

    # Align the second CC
    device_list = controller.t1.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res6 = Alignment.align_crystal_around_axis(crystal=controller.t6.optics,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t6.th.user_getPosition(),
                                               rotation_axis=controller.t6.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    controller.t6_rocking[:] = [np.copy(res6[-2]), np.copy(res6[-1])]

    # ----------------------------------------
    # Align the VCC branch
    kin = np.copy(kout + controller.g1.grating_1.momentum_transfer)
    res2 = Alignment.align_crystal_around_axis(crystal=controller.t2.optics,
                                               kin=kin,
                                               initial_angle=bragg - controller.t2.th.user_getPosition(),
                                               rotation_axis=controller.t2.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)

    # Move the crystal to the target path
    controller.t2_rocking[:] = [np.copy(res2[-2]), np.copy(res2[-1])]

    # Align the second VCC
    device_list = controller.t2.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res3 = Alignment.align_crystal_around_axis(crystal=controller.t3.optics,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t3.th.user_getPosition(),
                                               rotation_axis=controller.t3.th.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    controller.t3_rocking[:] = [np.copy(res3[-2]), np.copy(res3[-1])]

    # Align the third VCC
    device_list = controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res4 = Alignment.align_crystal_around_axis(crystal=controller.t45.optics1,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t45.th1.user_getPosition(),
                                               rotation_axis=controller.t45.th1.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    controller.t4_rocking[:] = [np.copy(res4[-2]), np.copy(res4[-1])]

    # Align the 4th VCC
    device_list = (controller.t2.optics.crystal_list +
                   controller.t3.optics.crystal_list +
                   controller.t45.optics1.crystal_list)
    kin_new = RayTracing.get_kout_multi_device(device_list=device_list, kin=kin)[-1]
    res5 = Alignment.align_crystal_around_axis(crystal=controller.t45.optics2,
                                               kin=kin_new,
                                               initial_angle=bragg - controller.t45.th2.user_getPosition(),
                                               rotation_axis=controller.t45.th2.rotation_axis,
                                               bandwidth_keV=0.5e-3,
                                               rot_crystal=False)
    controller.t5_rocking[:] = [np.copy(res5[-2]), np.copy(res5[-1])]


def get_vcc_kout(controller):
    trajectory, kout, pathlength = get_raytracing_trajectory(controller=controller, path="vcc", )
    return kout[-1]


def get_diode_readout(pulse_energy, ratio, noise_level):
    randomSeed = int(time.time() * 1e6) % 65536
    np.random.seed(randomSeed)

    reading = pulse_energy * ratio + noise_level * np.random.rand(1)

    return reading


# -----------------------------------------
#    Visualization
# -----------------------------------------
def plot_motors(controller, ax, color='black', axis="xz"):
    if axis == "xz":
        for tower in controller.all_towers:
            for item in tower.all_motors:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c=color)
    elif axis == 'yz':
        for tower in controller.all_towers:
            for item in tower.all_motors:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 0] / 1000, c=color)
    elif axis == 'xy':
        for tower in controller.all_towers:
            for item in tower.all_motors:
                ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, c=color)


def plot_optics(controller, ax, color='black', axis="xz"):
    if axis == 'xz':
        for tower in controller.all_towers:
            for item in tower.all_optics:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 1] / 1000, c=color)
    elif axis == 'yz':
        for tower in controller.all_towers:
            for item in tower.all_optics:
                ax.plot(item.boundary[:, 2] / 1000, item.boundary[:, 0] / 1000, c=color)
    elif axis == 'xy':
        for tower in controller.all_towers:
            for item in tower.all_optics:
                ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, c=color)


def plot_mono_rocking(controller, ax_mono_t1, ax_mono_t2):
    ax_mono_t1.plot(np.rad2deg(controller.mono_t1_rocking[0]) * 1e3,
                    controller.mono_t1_rocking[1], c='b', label='mono t1')
    ax_mono_t1.set_xlim([- 5, 5])
    ax_mono_t1.set_xlabel("relative th (mdeg)")
    ax_mono_t1.set_ylabel("R")
    ax_mono_t1.set_title("mono T1")

    ax_mono_t2.plot(np.rad2deg(controller.mono_t2_rocking[0]) * 1e3,
                    controller.mono_t2_rocking[1], c='r', label='mono t2')
    ax_mono_t2.set_xlim([- 5, 5])
    ax_mono_t2.set_xlabel("relative th (mdeg)")
    ax_mono_t2.set_ylabel("R")
    ax_mono_t2.set_title("mono T2")


def plot_mono_optics(controller, ax, show_trajectory=False, xlim=None, ylim=None):
    controller.plot_motors(ax=ax, color='black')
    controller.plot_optics(ax=ax, color='blue')

    if show_trajectory:
        mono_traj, mono_kout, mono_pathlength = controller.get_raytracing_trajectory(path="mono")
        ax.plot(mono_traj[:, 2] / 1e3, mono_traj[:, 1] / 1e3, 'g')

    ax.set_aspect('equal')
    ax.set_title("Mono after alignment")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_xlim([-10e3 - 800, -10e3 + 100])
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim([- 600, 100])
    if ylim:
        ax.set_ylim(ylim)


def plot_miniSD_table(controller, ax, xlim=None, ylim=None, show_trajectory=False):
    if xlim is None:
        xlim = [-100, 1200]
    if ylim is None:
        ylim = [-100, 100]

    controller.plot_motors(ax=ax, color='black')
    controller.plot_optics(ax=ax, color='blue')

    ax.set_aspect('equal')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if show_trajectory:
        vcc_traj, vcc_kout, vcc_path = controller.get_raytracing_trajectory(path="vcc")
        cc_traj, cc_kout, cc_path = controller.get_raytracing_trajectory(path="cc")

        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3, 'g', label='vcc')
        ax.plot(cc_traj[:, 2] / 1e3, cc_traj[:, 1] / 1e3, 'r', label='cc')


def plot_miniSD_rocking(controller, ax_list):
    # Get the current rocking curve
    get_miniSD_rocking(controller=controller)
    print("Get the most updated rocking curve around current location.")

    # Start plotting
    record_to_plot = [controller.t1_rocking, controller.t2_rocking, controller.t3_rocking,
                      controller.t4_rocking, controller.t5_rocking, controller.t6_rocking]
    for idx in range(6):
        record = record_to_plot[idx]
        ax_list[idx].plot(np.rad2deg(record[0]) * 1000, record[1], label='t{}'.format(idx + 1))
        ax_list[idx].set_xlabel('relative th (mdeg)')
        ax_list[idx].legend()
        ax_list[idx].set_xlim([-5, 5])
