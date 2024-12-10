import sys
import time

import XRaySimulation.RayTracing
import XRaySimulation.RockingCurve

sys.path.append("../../../../XRaySimulation")

import numpy as np
from matplotlib import patches

from XRaySimulation import DeviceSimu, util

"""
This file contains functions that are used to be defined in the controller class.
I split it because it was too long.
"""
si220 = {'thickness': 1.9201 * 1e-4,
         "chi0": complex(-0.10169E-04, 0.16106E-06),
         "chih": complex(0.61786E-05, - 0.15508E-06),
         "chihbar": complex(0.61786E-05, -0.15508E-06),
         "chih_pi": complex(0.48374E-05, -0.11996E-06),
         "chihbar_pi": complex(0.48374E-05, -0.11996E-06),
         }

si111 = {'thickness': 3.1355 * 1e-4,
         "chi0": complex(-0.10169E-04, 0.16106E-06),
         "chih": complex(0.53693E-05, -0.11228E-06),
         "chihbar": complex(0.53693E-05, -0.11228E-06),
         "chih_pi": complex(0.49322E-05, -0.10272E-06),
         "chihbar_pi": complex(0.49322E-05, -0.10272E-06),
         }

dia111 = {'thickness': 2.0593 * 1e-4,
          "chi0": complex(-0.15217E-04, 0.13392E-07),
          "chih": complex(0.55417E-05, -0.93083E-08),
          "chihbar": complex(0.55417E-05, -0.93083E-08),
          "chih_pi": complex(0.44959E-05, - 0.74602E-08),
          "chihbar_pi": complex(0.44959E-05, -0.74602E-08),
          }


def get_raytracing_trajectory(controller, path="mono", get_path_length='True', virtual_sample_plane=None):
    if path == "cc":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.sample.yag1, ])
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=
                                                                               np.copy(controller.sample.yag1.normal))

    elif path == "cc sample":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.sample.sample, ])

        if virtual_sample_plane is None:
            virtual_sample_plane = np.copy(controller.sample.sample.normal)
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.sample.surface_point),
                                                                               final_plane_normal=virtual_sample_plane)

    elif path == "vcc":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_1]
                       + controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                       + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list
                       + [controller.sample.yag1, ])

        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=
                                                                               np.copy(controller.sample.yag1.normal))
    elif path == "probe m1 only":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_1]
                       + controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                       + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list
                       + [controller.m1.optics, controller.sample.yag1])

        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=
                                                                               np.copy(controller.sample.yag1.normal))
    elif path == "probe":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_1]
                       + controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                       + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list
                       + [controller.m1.optics, controller.si.optics, controller.sample.yag1])

        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=
                                                                               np.copy(controller.sample.yag1.normal))

    elif path == "pump a":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_m1, controller.m2a.optics, controller.sample.yag1])
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=np.copy(
                                                                                   controller.sample.yag1.normal))

    elif path == "pump a no mirror":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_m1, controller.sample.yag1])
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=np.copy(
                                                                                   controller.sample.yag1.normal))

    elif path == "pump b":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_1, controller.m2b.optics, controller.sample.yag1])
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=np.copy(
                                                                                   controller.sample.yag1.normal))

    elif path == "pump b no mirror":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_1, controller.sample.yag1])
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=np.copy(
                                                                                   controller.sample.yag1.surface_point),
                                                                               final_plane_normal=np.copy(
                                                                                   controller.sample.yag1.normal))

    elif path == "mono":
        defice_list = [controller.mono_t1.optics, controller.mono_t2.optics]
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=np.array([0, 0, -7e6]),
                                                                               final_plane_normal=np.array([0, 0, -1]))

    elif path == "probe sample":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_1]
                       + controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                       + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list
                       + [controller.m1.optics, controller.si.optics, controller.sample.sample])

        if virtual_sample_plane is None:
            virtual_sample_plane = np.copy(controller.sample.sample.normal)
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.sample.surface_point),
                                                                               final_plane_normal=virtual_sample_plane)

    elif path == "pump a sample":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_m1, controller.m2a.optics, controller.sample.sample])

        if virtual_sample_plane is None:
            virtual_sample_plane = np.copy(controller.sample.sample.normal)
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.sample.surface_point),
                                                                               final_plane_normal=virtual_sample_plane)

    elif path == "pump b sample":
        defice_list = ([controller.mono_t1.optics, controller.mono_t2.optics, controller.g1.grating_m1]
                       + controller.t1.optics.crystal_list + controller.t6.optics.crystal_list
                       + [controller.tg_g.grating_1, controller.m2b.optics, controller.sample.yag1])

        if virtual_sample_plane is None:
            virtual_sample_plane = np.copy(controller.sample.sample.normal)
        trajectory, kout, pathlength = XRaySimulation.RayTracing.get_lightpath(device_list=defice_list,
                                                                               kin=controller.gaussian_pulse.k0,
                                                                               initial_point=controller.gaussian_pulse.x0,
                                                                               final_plane_point=
                                                                               np.copy(
                                                                                   controller.sample.sample.surface_point),
                                                                               final_plane_normal=virtual_sample_plane)

    else:
        print("Warning, the specified path option is not defined.")
        trajectory = 0
        kout = 0
        pathlength = 0

    if get_path_length:
        return trajectory, kout, pathlength
    else:
        return trajectory, kout


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


def plot_mono_optics(controller, ax, show_trajectory=False):
    controller.plot_motors(ax=ax, color='black')
    controller.plot_optics(ax=ax, color='blue')

    if show_trajectory:
        mono_traj, mono_kout, mono_pathlength = controller.get_raytracing_trajectory(path="mono")
        ax.plot(mono_traj[:, 2] / 1e3, mono_traj[:, 1] / 1e3, 'g', label='vcc')

    ax.set_aspect('equal')
    ax.set_title("Mono after alignment")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_xlim([-10e3 - 800, -10e3 + 100])
    ax.set_ylim([- 600, 100])


def plot_miniSD_table(controller, ax, xlim=None, ylim=None, show_trajectory=False, show_device=True):
    if xlim is None:
        xlim = [-100, 1200]
    if ylim is None:
        ylim = [-100, 100]

    if show_device:
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


def plot_beam_on_yag(controller, ax):
    vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')

    vcc_spot = patches.Rectangle((vcc_traj[-1][1] / 1e3 - 0.75, vcc_traj[-1][0] / 1e3 - 0.75),
                                 width=1.5, height=1.5, fill=False, edgecolor='green', label='vcc')
    probe_m1_spot = patches.RegularPolygon(
        xy=(probe_m1_traj[-1][1] / 1e3, probe_m1_traj[-1][0] / 1e3),
        numVertices=3, radius=1., fill=False, edgecolor='black', label='m1')
    probe_spot = patches.Circle((probe_traj[-1][1] / 1e3, probe_traj[-1][0] / 1e3),
                                radius=0.5, fill=False, edgecolor='orange', label='probe')

    cc_spot = patches.Rectangle((pump_ref_traj[-1][1] / 1e3 - 0.75, pump_ref_traj[-1][0] / 1e3 - 0.75),
                                width=1.5, height=1.5, fill=False, edgecolor='pink', label='cc')
    pump_no_m1a_spot = patches.RegularPolygon(
        xy=(pump_a_no_mirror_traj[-1][1] / 1e3, pump_a_no_mirror_traj[-1][0] / 1e3),
        numVertices=3, radius=1., fill=False, edgecolor='pink', label='m2a')
    pump_no_m2b_spot = patches.RegularPolygon(
        xy=(pump_b_no_mirror_traj[-1][1] / 1e3, pump_b_no_mirror_traj[-1][0] / 1e3),
        numVertices=3, radius=1., fill=False, edgecolor='brown', label='m2b')
    pump_m1a_spot = patches.Circle((pump_a_traj[-1][1] / 1e3, pump_a_traj[-1][0] / 1e3),
                                   radius=0.5, fill=False, edgecolor='red', label='tg a')
    pump_m2b_spot = patches.Circle((pump_b_traj[-1][1] / 1e3, pump_b_traj[-1][0] / 1e3),
                                   radius=0.5, fill=False, edgecolor='purple', label='tg b')

    for item in controller.sample.all_optics:
        ax.plot(item.boundary[:, 1] / 1000, item.boundary[:, 0] / 1000, color='blue')

    ax.add_patch(vcc_spot)
    ax.add_patch(probe_m1_spot)
    ax.add_patch(probe_spot)

    ax.add_patch(cc_spot)
    ax.add_patch(pump_no_m1a_spot)
    ax.add_patch(pump_no_m2b_spot)
    ax.add_patch(pump_m1a_spot)
    ax.add_patch(pump_m2b_spot)

    ax.set_title("X-ray coming out of the screen")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc=(1, 0))


def plot_beam_on_sample_yag(controller, ax, aspect=None):
    # Calculate the interaction point
    probe_sample_traj, probe_kout, probe_path = controller.get_raytracing_trajectory(path="probe sample")
    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc sample")
    pump_a_sample_traj, pump_a_kout, pump_a_path = controller.get_raytracing_trajectory(path="pump a sample")
    pump_b_sample_traj, pump_b_kout, pump_b_path = controller.get_raytracing_trajectory(path="pump b sample")

    # Define the rotation matrix
    rot_mat = util.get_rotmat_around_axis(angleRadian=np.deg2rad(5), axis=np.array([1.0, 0, 0]))
    rot_center = np.copy(controller.sample.sample.surface_point)
    # print(rot_mat)

    # Define the object
    tmp = np.dot(probe_sample_traj - rot_center, rot_mat.T)
    # print(tmp)
    probe_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                radius=1, fill=False, edgecolor='green', label='probe')

    tmp = np.dot(pump_ref_traj - rot_center, rot_mat.T)
    pump_ref_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                   radius=0.75, fill=False, edgecolor='black', label='cc')

    tmp = np.dot(pump_a_sample_traj - rot_center, rot_mat.T)
    pump_a_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                 radius=0.5, fill=False, edgecolor='red', label='pump a')

    tmp = np.dot(pump_b_sample_traj - rot_center, rot_mat.T)
    pump_b_spot = patches.Circle((tmp[-1][2] / 1e3, tmp[-1][0] / 1e3),
                                 radius=0.5, fill=False, edgecolor='purple', label='pump b')

    ax.plot(np.dot(controller.sample.sample.boundary - rot_center, rot_mat.T)[:, 2] / 1e3,
            np.dot(controller.sample.sample.boundary - rot_center, rot_mat.T)[:, 0] / 1e3,
            color='purple',
            )
    ax.plot(np.dot(controller.sample.yag_sample.boundary - rot_center, rot_mat.T)[:, 2] / 1e3,
            np.dot(controller.sample.yag_sample.boundary - rot_center, rot_mat.T)[:, 0] / 1e3,
            color='blue', )
    ax.add_patch(probe_spot)
    ax.add_patch(pump_ref_spot)
    ax.add_patch(pump_a_spot)
    ax.add_patch(pump_b_spot)

    ax.set_title('Zyla 2')

    if aspect:
        ax.set_aspect(aspect)

    ax.set_xlabel("horizontal (mm)")
    ax.set_ylabel("vertical (mm)")
    ax.legend()


def plot_m1_traj(controller, ax, axis='yz', xlim=None, ylim=None):
    # Get the most updated trajectory
    vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    # probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    # pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    # pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    # pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    # pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')
    print("Perform ray tracing calculation at current motor position.")

    if xlim is None:
        xlim = [3900, 4100]
    if ylim is None:
        ylim = [-1, 1]

    if axis == 'yz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                color='g', label='vcc')
        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                color='r', label='cc')
        ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 0] / 1e3,
                color='g', linestyle='--', label='probe m1')
        controller.plot_optics(ax=ax, axis=axis, color='blue')

    elif axis == 'xz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                color='g', label='vcc')
        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                color='r', label='cc')
        ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 1] / 1e3,
                color='g', linstyle='--', label='probe m1')
        controller.plot_optics(ax=ax, axis=axis, color='blue')

    else:
        print("Please check the source code for the option for rotation_axis argument.")
        print("The current one \'{}\' is not defined".format(axis))

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("{} rotation_axis (mm)".format(axis[1]))
    ax.set_ylabel("{} rotation_axis (mm)".format(axis[0]))
    ax.set_title('Mirror 1')
    #ax.legend()


def plot_si_traj(controller, ax, axis='yz', xlim=None, ylim=None):
    # Get the most updated trajectory
    vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    # pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    # pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    # pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    # pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')
    print("Perform ray tracing calculation at current motor position.")

    if xlim is None:
        xlim = [probe_traj[-1, 2] / 1e3 - 50, probe_traj[-1, 2] / 1e3 + 5]
    if ylim is None:
        ylim = [probe_traj[-1, 0] / 1e3 - 15, probe_traj[-1, 0] / 1e3 + 20]

    if axis == 'yz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                color='g', label='vcc')
        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                color='r', label='cc')
        ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 0] / 1e3,
                color='g', linestyle='--', label='probe m1')
        ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 0] / 1e3,
                color='g', linestyle='dotted', label='probe')

        controller.plot_optics(ax=ax, axis=axis, color='blue')

    elif axis == 'xz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                color='g', label='vcc')
        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                color='r', label='cc')
        ax.plot(probe_m1_traj[:, 2] / 1e3, probe_m1_traj[:, 1] / 1e3,
                color='g', linstyle='--', label='probe m1')
        ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 1] / 1e3,
                color='g', linestyle='dotted', label='probe')

        controller.plot_optics(ax=ax, axis=axis, color='blue')

    else:
        print("Please check the source code for the option for rotation_axis argument.")
        print("The current one \'{}\' is not defined".format(axis))

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("{} rotation_axis (mm)".format(axis[1]))
    ax.set_ylabel("{} rotation_axis (mm)".format(axis[0]))
    ax.set_title('silicon')
    #ax.legend(loc=(1, 0))


def plot_tg_traj(controller, ax, axis='yz', xlim=None, ylim=None):
    # Get the most updated trajectory
    vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')
    print("Perform ray tracing calculation at current motor position.")

    if xlim is None:
        xlim = [7400, 7700]
    if ylim is None:
        ylim = [-5, 5]

    if axis == 'yz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 0] / 1e3,
                color='g', label='vcc')
        ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 0] / 1e3,
                color='g', linestyle='dotted', label='probe')

        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 0] / 1e3,
                color='r', label='cc')
        ax.plot(pump_a_no_mirror_traj[:, 2] / 1e3, pump_a_no_mirror_traj[:, 0] / 1e3,
                linestyle='--', color='r', label='pump a')
        ax.plot(pump_a_traj[:, 2] / 1e3, pump_a_traj[:, 0] / 1e3,
                linestyle='dotted', color='r', label='pump a')
        ax.plot(pump_b_no_mirror_traj[:, 2] / 1e3, pump_b_no_mirror_traj[:, 0] / 1e3,
                linestyle='--', color='r', label='pump b')
        ax.plot(pump_b_traj[:, 2] / 1e3, pump_b_traj[:, 0] / 1e3,
                linestyle='dotted', color='r', label='pump b')

        controller.plot_optics(ax=ax, axis=axis, color='blue')

    elif axis == 'xz':
        ax.plot(vcc_traj[:, 2] / 1e3, vcc_traj[:, 1] / 1e3,
                color='g', label='vcc')
        ax.plot(probe_traj[:, 2] / 1e3, probe_traj[:, 1] / 1e3,
                color='g', linestyle='dotted', label='probe')

        ax.plot(pump_ref_traj[:, 2] / 1e3, pump_ref_traj[:, 1] / 1e3,
                color='r', label='cc')
        ax.plot(pump_a_no_mirror_traj[:, 2] / 1e3, pump_a_no_mirror_traj[:, 1] / 1e3,
                linestyle='--', color='r', label='pump a')
        ax.plot(pump_a_traj[:, 2] / 1e3, pump_a_traj[:, 1] / 1e3,
                linestyle='dotted', color='r', label='pump a')
        ax.plot(pump_b_no_mirror_traj[:, 2] / 1e3, pump_b_no_mirror_traj[:, 1] / 1e3,
                linestyle='--', color='r', label='pump b')
        ax.plot(pump_b_traj[:, 2] / 1e3, pump_b_traj[:, 1] / 1e3,
                linestyle='dotted', color='r', label='pump b')

        controller.plot_optics(ax=ax, axis=axis, color='blue')

    else:
        print("Please check the source code for the option for rotation_axis argument.")
        print("The current one \'{}\' is not defined".format(axis))

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("{} rotation_axis (mm)".format(axis[1]))
    ax.set_ylabel("{} rotation_axis (mm)".format(axis[0]))
    ax.set_title('Sample')
    #ax.legend(loc=(1, 0))


def get_beam_position_on_yag(controller):
    vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')

    return {'vcc': vcc_traj[-1],
            'probe m1': probe_m1_traj[-1],
            'probe': probe_traj[-1],

            'cc': pump_ref_traj[-1],
            'pump a no mirror': pump_a_no_mirror_traj[-1],
            'pump b no mirror': pump_b_no_mirror_traj[-1],
            'pump a': pump_a_traj[-1],
            'pump b': pump_b_traj[-1],
            }


def get_beam_position_on_sample_yag(controller):
    probe_sample_traj, probe_kout, probe_path = controller.get_raytracing_trajectory(path="probe sample")
    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc sample")
    pump_a_sample_traj, pump_a_kout, pump_a_path = controller.get_raytracing_trajectory(path="pump a sample")
    pump_b_sample_traj, pump_b_kout, pump_b_path = controller.get_raytracing_trajectory(path="pump b sample")

    return {'probe': probe_sample_traj[-1],
            'cc': pump_ref_traj[-1],
            'pump a': pump_a_sample_traj[-1],
            'pump b': pump_b_sample_traj[-1],
            }


def get_sample_path_length(controller):
    probe_sample_traj, probe_kout, probe_path = controller.get_raytracing_trajectory(path="probe sample")
    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc sample")
    pump_a_sample_traj, pump_a_kout, pump_a_path = controller.get_raytracing_trajectory(path="pump a sample")
    pump_b_sample_traj, pump_b_kout, pump_b_path = controller.get_raytracing_trajectory(path="pump b sample")

    return {'probe': probe_path,
            'cc': pump_ref_path,
            'pump a': pump_a_path,
            'pump b': pump_b_path,
            }


def get_sample_kout(controller):
    probe_sample_traj, probe_kout, probe_path = controller.get_raytracing_trajectory(path="probe sample")
    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc sample")
    pump_a_sample_traj, pump_a_kout, pump_a_path = controller.get_raytracing_trajectory(path="pump a sample")
    pump_b_sample_traj, pump_b_kout, pump_b_path = controller.get_raytracing_trajectory(path="pump b sample")

    return {'probe': probe_kout,
            'cc': pump_ref_kout,
            'pump a': pump_a_kout,
            'pump b': pump_b_kout,
            }


def get_arrival_time(controller):
    probe_sample_traj, probe_kout, probe_path = controller.get_raytracing_trajectory(
        path="probe", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(
        path="cc", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
    pump_a_sample_traj, pump_a_kout, pump_a_path = controller.get_raytracing_trajectory(
        path="pump a", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))
    pump_b_sample_traj, pump_b_kout, pump_b_path = controller.get_raytracing_trajectory(
        path="pump b", virtual_sample_plane=np.array([0.0, 0.0, -1.0]))

    return {'probe': probe_path,
            'cc': pump_ref_path,
            'pump a': pump_a_path,
            'pump b': pump_b_path,
            }


def _get_diode(controller, spectrum_intensity):
    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['mono T2']))
    result = {
        "ipm2": get_diode_readout(pulse_energy=energy,
                                  ratio=controller.diode_ratio['ipm2'],
                                  noise_level=controller.diode_noise_level['ipm2'])}

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['g1 1st order']))
    result.update({
        "dg1": get_diode_readout(pulse_energy=energy,
                                 ratio=controller.diode_ratio['dg1'],
                                 noise_level=controller.diode_noise_level['dg1'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['cc1']))
    result.update({
        "d1": get_diode_readout(pulse_energy=energy,
                                ratio=controller.diode_ratio['d1'],
                                noise_level=controller.diode_noise_level['d1'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['vcc1']))
    result.update({
        "d2": get_diode_readout(pulse_energy=energy,
                                ratio=controller.diode_ratio['d2'],
                                noise_level=controller.diode_noise_level['d2'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['vcc2']))
    result.update({
        "d3": get_diode_readout(pulse_energy=energy,
                                ratio=controller.diode_ratio['d3'],
                                noise_level=controller.diode_noise_level['d3'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['vcc3']))
    result.update({
        "vcc3": get_diode_readout(pulse_energy=energy,
                                  ratio=controller.diode_ratio['d4'],
                                  noise_level=controller.diode_noise_level['d4'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['vcc4']))
    result.update({
        "vcc4": get_diode_readout(pulse_energy=energy,
                                  ratio=controller.diode_ratio['d5'],
                                  noise_level=controller.diode_noise_level['d5'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['cc2']))
    result.update({
        "cc2": get_diode_readout(pulse_energy=energy,
                                 ratio=controller.diode_ratio['d6'],
                                 noise_level=controller.diode_noise_level['d6'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['pump a']))
    result.update({
        "pump a": get_diode_readout(pulse_energy=energy,
                                    ratio=controller.diode_ratio['pump'],
                                    noise_level=controller.diode_noise_level['pump'])})

    energy = np.sum(np.multiply(spectrum_intensity, controller.crystal_efficiency['probe']))
    result.update({
        "si": get_diode_readout(pulse_energy=energy,
                                ratio=controller.diode_ratio['probe'],
                                noise_level=controller.diode_noise_level['probe'])})

    result.update({
        "d4": get_diode_readout(pulse_energy=0,
                                ratio=controller.diode_ratio['d4'], noise_level=controller.diode_noise_level['d4']),
        "d5": get_diode_readout(pulse_energy=0,
                                ratio=controller.diode_ratio['d5'], noise_level=controller.diode_noise_level['d5']),
        "d6": get_diode_readout(pulse_energy=0,
                                ratio=controller.diode_ratio['d6'], noise_level=controller.diode_noise_level['d6']),
        "pump": get_diode_readout(pulse_energy=0,
                                  ratio=controller.diode_ratio['pump'],
                                  noise_level=controller.diode_noise_level['pump']),
        "probe": get_diode_readout(pulse_energy=0,
                                   ratio=controller.diode_ratio['probe'],
                                   noise_level=controller.diode_noise_level['probe']),
    })
    if controller.cc_shutter:
        result['d6'] += result['cc2']
        result['probe'] += result['si']
    if controller.vcc_shutter:
        result['d4'] += result['vcc3']
        result['d5'] += result['vcc4']
        result['d6'] += result['vcc4']
        result['pump'] += result['pump a']

    return result


def get_diode(controller, spectrum_intensity, k_grid, gpu=False, force=False):
    # Step 1: check if the sase pulse is 1D
    if (len(spectrum_intensity.shape) == 1) and (not gpu):
        # Perform the 1D calculation
        if np.max(np.abs(controller.crystal_efficiency['k_vec'] - k_grid)) > 1e-6:
            print("The maximal difference between the k_vec of the SASE pulse "
                  "and the k_vec of the energy efficiency is larger than 1e-6.")
            print("Do not do the calculation unless setting force=True to force the calcluation.")
            if force:
                result = controller._get_diode(spectrum_intensity)
                return result
            else:
                return 1
        else:
            result = controller._get_diode(spectrum_intensity)
            return result
    else:
        print("Current the gpu support is not implemented yet with this module.")
        return 1


def get_zyla_1(controller, sigma_mat, i_probe, i_pump_a, i_pump_b, i_pump_ref, beam_list=None):
    """
    Get the current beam profile on the YAG screen looking through the zyla camera

    :return:
    """
    # Get the position on the YAG screen
    # vcc_traj, vcc_kout, vcc_pathlength = controller.get_raytracing_trajectory(path="vcc")
    # probe_m1_traj, probe_m1_kout, probe_m1_pathlength = controller.get_raytracing_trajectory(path="probe m1 only")
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    # pump_a_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a no mirror')
    pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    # pump_b_no_mirror_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b no mirror')
    pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')

    if beam_list is None:
        beam_list = ("probe", 'pump a', 'pump b')

    yag_image = np.zeros((controller.pixel_num_x, controller.pixel_num_y))
    # Get the pixel coordinate
    pixel_coor_x = controller.pixel_num_x + controller.sample.yag1.surface_point[1]
    pixel_coor_y = controller.pixel_num_x + controller.sample.yag1.surface_point[0]

    if 'probe' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = (probe_traj[-1] - controller.sample.yag1.surface_point)
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat,
                                                    beam_center=beam_center,
                                                    intensity=i_probe,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    if 'pump a' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = (pump_a_traj[-1] - controller.sample.yag1.surface_point)
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_a,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    if 'pump b' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = (pump_b_traj[-1] - controller.sample.yag1.surface_point)
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_b,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })
    if 'pump ref' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = (pump_ref_traj[-1] - controller.sample.yag1.surface_point)
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_ref,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    return yag_image


def get_zyla_2(controller, sigma_mat, i_probe, i_pump_a, i_pump_b, i_pump_ref, beam_list=None):
    """
    Get the current beam profile on the YAG screen looking through the zyla camera

    :return:
    """
    if beam_list is None:
        beam_list = ("probe", 'pump a', 'pump b')

    # Get the position on the YAG screen
    probe_traj, kout, probe_pathlength = controller.get_raytracing_trajectory(path="probe")

    pump_ref_traj, pump_ref_kout, pump_ref_path = controller.get_raytracing_trajectory(path="cc")
    pump_a_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump a')
    pump_b_traj, kout, pump_a_path = controller.get_raytracing_trajectory(path='pump b')

    rot_mat = util.get_rotmat_around_axis(angleRadian=np.deg2rad(5), axis=np.array([1.0, 0, 0]))

    (sigma_mat_yag,
     long,
     short,
     mag_factor) = get_beam_profile_on_yag_sample(rot_mat=rot_mat,
                                                  kin=kout[-1],
                                                  beam_size=np.sqrt(sigma_mat[0, 0]))

    yag_image = np.zeros((controller.pixel_num_x, controller.pixel_num_y))
    # Get the pixel coordinate
    pixel_coor_x = controller.pixel_num_x + controller.sample.yag1.surface_point[1]
    pixel_coor_y = controller.pixel_num_x + controller.sample.yag1.surface_point[0]

    if 'probe' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = np.dot(rot_mat, (probe_traj[-1] - controller.sample.yag1.surface_point))
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat_yag,
                                                    beam_center=beam_center,
                                                    intensity=i_probe,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    if 'pump a' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = np.dot(rot_mat, (pump_a_traj[-1] - controller.sample.yag1.surface_point))
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat_yag,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_a,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    if 'pump b' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = np.dot(rot_mat, (pump_b_traj[-1] - controller.sample.yag1.surface_point))
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat_yag,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_b,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })
    if 'pump ref' in beam_list:
        # Get the relative position of the X-ray with respect to the yag center
        position_relative = np.dot(rot_mat, (pump_ref_traj[-1] - controller.sample.yag1.surface_point))
        beam_center = np.array([position_relative[1], position_relative[0]], dtype=np.float64)

        yag_image += DeviceSimu.get_gaussian_on_yag(sigma_mat=sigma_mat_yag,
                                                    beam_center=beam_center,
                                                    intensity=i_pump_ref,
                                                    pixel_coor={'xCoor': pixel_coor_y,
                                                                'yCoor': pixel_coor_x, })

    return yag_image


def get_beam_profile_on_yag_sample(rot_mat, kin, beam_size):
    # Go the reference frame where the 0 is vertical direction, 2 is parallel to the yag horizontal edge
    # and 1 is normal to the yag surface
    kin_new = np.dot(rot_mat, kin)

    # Get the incident si111_angle with the wavevector and the YAG normal
    angle = np.arcsin(kin_new[1] / np.linalg.norm((kin_new)))

    # mag factor
    mag_factor = 1 / np.sin(angle)

    # Get the long rotation_axis
    long = np.array([kin_new[0], kin_new[2]])
    long /= np.linalg.norm(long)
    long *= mag_factor * beam_size

    short = np.array([kin_new[2], -kin_new[0]])
    short /= np.linalg.norm(short)
    short *= beam_size

    sigma_mat = np.outer(long, long) + np.outer(short, short)

    return sigma_mat, long, short, mag_factor


def align_xpp_mono(controller):
    # Get the geometry bragg si111_angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0,
                                 plane_distance=dia111['thickness'])

    # Step 1, move the mono1 th to the geometric path
    _ = controller.mono_t1.th_umv(target=-bragg)
    _ = controller.mono_t2.th_umv(target=-bragg)

    # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
    (angles1, reflect_sigma1,
     reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_around_axis(
        kin=controller.gaussian_pulse.k0,
        scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
        scan_number=10 ** 3,
        rotation_axis=controller.mono_t1.th.rotation_axis,
        crystal=controller.mono_t1.optics, )

    # Get the target bragg peak
    fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                              curve_values=np.square(np.abs(reflect_sigma1)),
                                              center=True,
                                              get_index=True)

    # Move the crystal to the target path
    _ = controller.mono_t1.th_umv(target=-bragg + angle_adjust)

    # Align the second crystal
    kin1 = np.copy(kout1[index])

    (angles2, reflect_sigma2,
     reflect_pi2, b_factor2, kout2) = XRaySimulation.RockingCurve.get_rocking_curve_around_axis(
        kin=kin1,
        scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
        scan_number=10 ** 3,
        rotation_axis=controller.mono_t2.th.rotation_axis,
        crystal=controller.mono_t2.optics, )

    # Get the target bragg peak
    fwhm2, angle_adjust2, index2 = util.get_fwhm(coordinate=angles2,
                                                 curve_values=np.square(np.abs(reflect_sigma2)),
                                                 center=True,
                                                 get_index=True)
    _ = controller.mono_t2.th_umv(target=-bragg + angle_adjust2)

    controller.mono_t1_rocking = [angles1 - angle_adjust, np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
    controller.mono_t2_rocking = [angles2 - angle_adjust2, np.square(np.abs(reflect_sigma2)) / np.abs(b_factor2)]

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

    return ((angles1 + angle_adjust, np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1), kout1),
            (angles2 + angle_adjust2, np.square(np.abs(reflect_sigma2)) / np.abs(b_factor2), kout2),)


def align_miniSD(controller):
    # Get the kout after the XPP mono
    _, kout, _ = XRaySimulation.RayTracing.get_lightpath(
        device_list=[controller.mono_t1.optics, controller.mono_t2.optics],
        kin=controller.gaussian_pulse.k0,
        initial_point=controller.gaussian_pulse.x0,
        final_plane_point=np.array([0, 0, 10e6]),
        final_plane_normal=np.array([0, 0, -1]))
    kout = kout[-1]

    # Get the geometry bragg si111_angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0,
                                 plane_distance=si220['thickness'])
    bragg_list = [bragg, bragg, bragg, bragg, bragg, bragg]

    # Step 1, move the mono1 th to the geometric path
    _ = controller.t1.th_umv(target=bragg_list[0])
    _ = controller.t2.th_umv(target=bragg_list[1])
    _ = controller.t3.th_umv(target=bragg_list[2])
    _ = controller.t45.th1_umv(target=bragg_list[3])
    _ = controller.t45.th2_umv(target=bragg_list[4])
    _ = controller.t6.th_umv(target=bragg_list[5])

    # Fine adjustment according to dynamical diffraction theory
    kin = np.copy(kout + controller.g1.grating_m1.momentum_transfer)
    combo = [[controller.t1, controller.t1_rocking, bragg_list[0]],
             [controller.t6, controller.t6_rocking, bragg_list[-1]], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].th.rotation_axis,
            channelcut=tower[0].optics, )

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)
        # Move the crystal to the target path
        _ = tower[0].th_umv(target=tower[2] + angle_adjust)

        # Record the current rocking curve
        tower[1][:] = [np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
        kin = np.copy(kout1[index])

    # Align vcc2 and vcc3
    # Fine adjustment according to dynamical diffraction theory
    kin = np.copy(kout + controller.g1.grating_1.momentum_transfer)
    combo = [[controller.t2, controller.t2_rocking, bragg_list[1]],
             [controller.t3, controller.t3_rocking, bragg_list[2]], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].th.rotation_axis,
            channelcut=tower[0].optics, )

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)
        # Move the crystal to the target path
        _ = tower[0].th_umv(target=tower[2] + angle_adjust)

        # Record the current rocking curve
        tower[1][:] = [np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
        kin = np.copy(kout1[index])

    # Align vcc4 and vcc5
    # Fine adjustment according to dynamical diffraction theory
    combo = [[controller.t45.th1, controller.t45.optics1, controller.t45.th1_umv, controller.t4_rocking, bragg_list[3]],
             [controller.t45.th2, controller.t45.optics2, controller.t45.th2_umv, controller.t5_rocking,
              bragg_list[4]], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].rotation_axis,
            channelcut=tower[1])

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)
        # Move the crystal to the target path
        _ = tower[2](target=tower[4] + angle_adjust)

        # Record the current rocking curve
        tower[3][:] = (np.copy(angles1 - angle_adjust), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1))
        kin = np.copy(kout1[index])


def get_miniSD_rocking(controller):
    # Get the kout after the XPP mono
    _, kout, _ = XRaySimulation.RayTracing.get_lightpath(
        device_list=[controller.mono_t1.optics, controller.mono_t2.optics],
        kin=controller.gaussian_pulse.k0,
        initial_point=controller.gaussian_pulse.x0,
        final_plane_point=np.array([0, 0, 10e6]),
        final_plane_normal=np.array([0, 0, -1]))
    kout = kout[-1]

    # Fine adjustment according to dynamical diffraction theory
    kin = np.copy(kout + controller.g1.grating_m1.momentum_transfer)
    combo = [[controller.t1, controller.t1_rocking],
             [controller.t6, controller.t6_rocking], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1, reflect_pi1, b_factor1, kout1
         ) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].th.rotation_axis, channelcut=tower[0].optics, )

        # Get the target bragg peak
        (fwhm, angle_adjust, index
         ) = util.get_fwhm(coordinate=angles1, curve_values=np.square(np.abs(reflect_sigma1)),
                           center=True, get_index=True)

        # Record the current rocking curve
        tower[1][:] = [np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
        kin = np.copy(kout1[index])

    # Align vcc2 and vcc3
    # Fine adjustment according to dynamical diffraction theory
    kin = np.copy(kout + controller.g1.grating_1.momentum_transfer)
    combo = [[controller.t2, controller.t2_rocking],
             [controller.t3, controller.t3_rocking], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].th.rotation_axis,
            channelcut=tower[0].optics, )

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)

        # Record the current rocking curve
        tower[1][:] = [np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1)]
        kin = np.copy(kout1[index])

    # Align vcc4 and vcc5
    # Fine adjustment according to dynamical diffraction theory
    combo = [[controller.t45.th1, controller.t45.optics1, controller.t45.th1_umv, controller.t4_rocking],
             [controller.t45.th2, controller.t45.optics2, controller.t45.th2_umv, controller.t5_rocking], ]
    for tower in combo:
        # Step 2, get the rocking curve around the motion rotation_axis for the two crystals.
        (angles1, reflect_sigma1,
         reflect_pi1, b_factor1, kout1) = XRaySimulation.RockingCurve.get_rocking_curve_channelcut_around_axis(
            kin=kin,
            scan_range=np.array([-np.deg2rad(0.1), np.deg2rad(0.1)]),
            scan_number=10 ** 3,
            rotation_axis=tower[0].rotation_axis,
            channelcut=tower[1])

        # Get the target bragg peak
        fwhm, angle_adjust, index = util.get_fwhm(coordinate=angles1,
                                                  curve_values=np.square(np.abs(reflect_sigma1)),
                                                  center=True,
                                                  get_index=True)

        # Record the current rocking curve
        tower[3][:] = (np.copy(angles1), np.square(np.abs(reflect_sigma1)) / np.abs(b_factor1))
        kin = np.copy(kout1[index])


def get_diode_readout(pulse_energy, ratio, noise_level):
    randomSeed = int(time.time() * 1e6) % 65536
    np.random.seed(randomSeed)

    reading = pulse_energy * ratio + noise_level * np.random.rand(1)

    return reading
