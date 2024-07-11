import sys
import time

from XRaySimulation import Efficiency, DeviceSimu, RayTracing, RockingCurve, Alignment

sys.path.append("../../../")

import numpy as np

from XRaySimulation import util

si220 = {'thickness': 1.9201 * 1e-4,
         "chi0": complex(-0.80575E-05, 0.10198E-06),
         "chih": complex(0.48909E-05, -0.98241E-07),
         "chihbar": complex(0.48909E-05, -0.98241E-07),
         "chih_pi": complex(0.40482E-05, -0.80452E-07),
         "chihbar_pi": complex(0.40482E-05, -0.80452E-07),
         }

dia111 = {'thickness': 2.0593 * 1e-4,
          "chi0": complex(-0.12067E-04, 0.82462E-08),
          "chih": complex(0.43910E-05, -0.57349E-08),
          "chihbar": complex(0.43910E-05, -0.57349E-08),
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
    # Get the geometry bragg si111_angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=dia111['thickness'])

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

    # Get the geometry bragg si111_angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=si220['thickness'])
    print(bragg)

    # Step 1, move the mono1 th to the geometric Bragg si111_angle
    # _ = controller.t1.th_umv(target=bragg)
    # _ = controller.t2.th_umv(target=bragg)
    # _ = controller.t3.th_umv(target=bragg)
    # _ = controller.t45.th1_umv(target=bragg)
    # _ = controller.t45.th2_umv(target=bragg)
    # _ = controller.t6.th_umv(target=bragg)

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

    # Get the geometry bragg si111_angle
    bragg = util.get_bragg_angle(wave_length=np.pi * 2 / controller.gaussian_pulse.klen0, plane_distance=si220['thickness'])
    print(bragg)

    # Step 1, move the mono1 th to the geometric Bragg si111_angle
    # _ = controller.t1.th_umv(target=bragg)
    # _ = controller.t2.th_umv(target=bragg)
    # _ = controller.t3.th_umv(target=bragg)
    # _ = controller.t45.th1_umv(target=bragg)
    # _ = controller.t45.th2_umv(target=bragg)
    # _ = controller.t6.th_umv(target=bragg)

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


def get_reflectivity(controller):
    energy_range = 8e-3
    num = 2000

    # Get the kout after the XPP mono
    _, kout, _ = RayTracing.get_lightpath(
        device_list=[controller.mono_t1.optics, controller.mono_t2.optics],
        kin=controller.gaussian_pulse.k0,
        initial_point=controller.gaussian_pulse.x0,
        final_plane_point=np.array([0, 0, 10e6]),
        final_plane_normal=np.array([0, 0, -1]))
    kout_mono = np.copy(kout[-1])
    kout_g1_cc = kout_mono + np.copy(controller.g1.grating_m1.momentum_transfer)

    # ----------------------------------------------
    # Get the CC branch efficiency curve
    klen = np.linalg.norm(kout_g1_cc)
    klen_array = np.linspace(start=-energy_range / 2., stop=energy_range / 2, num=num)
    klen_array = util.kev_to_wavevec_length(klen_array)
    klen_array += klen

    k_direction = kout_g1_cc / klen
    kin_array = np.zeros((num, 3))
    kin_array[:, :] = k_direction[np.newaxis, :]
    kin_array[:, :] *= klen_array[:, np.newaxis]

    # Get the device_list
    device_list = controller.t1.optics.crystal_list + controller.t6.optics.crystal_list

    (total_efficiency_holder,
     efficiency_holder,
     kout_holder) = Efficiency.get_crystal_reflectivity(device_list=device_list,
                                                        kin_list=kin_array)
    cc_efficiency = np.copy(total_efficiency_holder)
    cc1_efficiency = np.copy(efficiency_holder[0] * efficiency_holder[1])
    cc6_efficiency = np.copy(efficiency_holder[2] * efficiency_holder[3])

    # ----------------------------------------------
    # Get the VCC branch efficiency curve
    kout_g1_vcc = kout_mono + np.copy(controller.g1.grating_1.momentum_transfer)

    klen = np.linalg.norm(kout_g1_vcc)
    klen_array = np.linspace(start=-energy_range / 2., stop=energy_range / 2, num=num)
    klen_array = util.kev_to_wavevec_length(klen_array)
    klen_array += klen

    k_direction = kout_g1_vcc / klen
    kin_array = np.zeros((num, 3))
    kin_array[:, :] = k_direction[np.newaxis, :]
    kin_array[:, :] *= klen_array[:, np.newaxis]

    # Get the device_list
    device_list = (controller.t2.optics.crystal_list + controller.t3.optics.crystal_list
                   + controller.t45.optics1.crystal_list + controller.t45.optics2.crystal_list)

    (total_efficiency_holder,
     efficiency_holder,
     kout_holder) = Efficiency.get_crystal_reflectivity(device_list=device_list,
                                                        kin_list=kin_array)

    vcc_efficiency = np.copy(total_efficiency_holder)
    cc2_efficiency = np.copy(efficiency_holder[0] * efficiency_holder[1])
    cc3_efficiency = np.copy(efficiency_holder[2] * efficiency_holder[3])
    cc4_efficiency = np.copy(efficiency_holder[3] * efficiency_holder[4])
    cc5_efficiency = np.copy(efficiency_holder[5] * efficiency_holder[6])

    return {"cc": cc_efficiency,
            "vcc": vcc_efficiency,
            "cc1": cc1_efficiency,
            "cc2": cc2_efficiency,
            "cc3": cc3_efficiency,
            "cc4": cc4_efficiency,
            "cc5": cc5_efficiency,
            "cc6": cc6_efficiency,
            "energy": np.linspace(-energy_range, energy_range, num) + 11,
            }


def get_vcc_kout(controller):
    trajectory, kout, pathlength = get_raytracing_trajectory(controller=controller, path="vcc", )
    return kout[-1]


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
                result = _get_diode(controller=controller, spectrum_intensity=spectrum_intensity)
                return result
            else:
                return 1
        else:
            result = _get_diode(controller=controller, spectrum_intensity=spectrum_intensity)
            return result
    else:
        print("Current the gpu support is not implemented yet with this module.")
        return 1


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
