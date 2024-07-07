import numpy as np

from XRaySimulation import util, RockingCurve
from XRaySimulation.RayTracing import get_kout_single_device


def get_intensity_efficiency_sigma_polarization_single_device(device, kin):
    """
    Get the output intensity efficiency for the given wave vector
    assuming a monochromatic plane incident wave.

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        tmp = np.zeros((1, 3))
        tmp[0, :] = kin

        (reflect_s,
         reflect_p,
         b,
         kout_grid) = util.get_bragg_reflection_array(kin_grid=tmp,
                                                      d=device.thickness,
                                                      h=device.h,
                                                      n=device.normal,
                                                      chi0=device.chi0,
                                                      chih_sigma=device.chih_sigma,
                                                      chihbar_sigma=device.chihbar_sigma,
                                                      chih_pi=device.chih_pi,
                                                      chihbar_pi=device.chihbar_pi)

        efficiency = np.square(np.abs(reflect_s)) / np.abs(b)
        return efficiency

    if device.type == "Transmissive Grating":

        # Determine the grating order
        if device.order == 0:
            efficiency = util.get_square_grating_0th_transmission(kin=kin,
                                                                  height_vec=device.h,
                                                                  refractive_index=device.n,
                                                                  ab_ratio=device.ab_ratio,
                                                                  base=device.thick_vec)
        else:
            efficiency, _, _ = util.get_square_grating_transmission(kin=kin,
                                                                    height_vec=device.h,
                                                                    ab_ratio=device.ab_ratio,
                                                                    base=device.thick_vec,
                                                                    refractive_index=device.n,
                                                                    order=device.order,
                                                                    grating_k=device.momentum_transfer)
        # Translate to the intensity efficiency
        return np.square(np.abs(efficiency))

    if device.type == "Transmission Telescope for CPA":
        return np.square(np.abs(device.efficiency))


def get_intensity_efficiency_sigma_polarization(device_list, kin):
    """
    Get the reflectivity of this kin_array.
    Notice that this function is not particularly useful.
    It just aims to make the function lists complete.

    :param device_list:
    :param kin:
    :return:
    """
    efficiency_list = np.zeros(len(device_list), dtype=np.float64)

    # Variable for the kout
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    # Loop through all the devices
    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the efficiency
        efficiency_list[idx] = get_intensity_efficiency_sigma_polarization_single_device(device=device,
                                                                                         kin=kout_list[idx])
        # Get the output wave vector
        kout_list[idx + 1] = get_kout_single_device(device=device, kin=kout_list[idx])

    # Get the overall efficiency
    total_efficiency = np.prod(efficiency_list)

    return total_efficiency, efficiency_list, kout_list


def get_output_efficiency_curve(device_list, kin_list):
    """
    Get the reflectivity for each kin_array.

    :param kin_list:
    :param device_list:
    :return:
    """
    d_num = len(device_list)  # number of devices
    k_num = kin_list.shape[0]  # number of kin_array vectors

    efficiency_holder = np.zeros((k_num, d_num))
    kout_holder = np.zeros((k_num, d_num + 1, 3))
    total_efficiency_holder = np.zeros(k_num)

    # Loop through all the kin_array
    for idx in range(k_num):
        (total_efficiency,
         efficiency_tmp,
         kout_tmp) = get_intensity_efficiency_sigma_polarization(device_list=device_list,
                                                                 kin=kin_list[idx])

        efficiency_holder[idx, :] = efficiency_tmp[:]
        kout_holder[idx, :, :] = kout_tmp[:, :]
        total_efficiency_holder[idx] = total_efficiency

    return total_efficiency_holder, efficiency_holder, kout_holder


def get_crystal_reflectivity(device_list, kin_list):
    """
    Get the reflectivity for each kin_array.

    :param kin_list:
    :param device_list:
    :return:
    """
    d_num = len(device_list)  # number of devices
    k_num = kin_list.shape[0]  # number of kin_array vectors

    efficiency_holder = np.zeros((d_num, k_num))
    kout_holder = np.zeros((d_num + 1, k_num, 3))

    kout_holder[0, :, :] = np.copy(kin_list)[:, :]

    # Loop through all the kin_array
    for idx in range(d_num):
        (reflect_sigma, reflect_pi, b_factor, kout_array
         ) = RockingCurve.get_bragg_reflectivity_fix_crystal(kin=kout_holder[idx, :, :],
                                                             crystal=device_list[idx])

        efficiency_holder[idx, :] = np.square(np.abs(reflect_sigma)) / np.abs(b_factor)
        kout_holder[idx + 1, :, :] = kout_array[:, :]

    total_efficiency_holder = np.prod(efficiency_holder, axis=0)

    return total_efficiency_holder, efficiency_holder, kout_holder
