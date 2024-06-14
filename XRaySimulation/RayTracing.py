import numpy as np

from XRaySimulation import util


def get_kout_single_device(device, kin):
    """
    Get the output wave vector given the incident wave vector

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        kout = util.get_bragg_kout(kin=kin,
                                   h=device.h,
                                   normal=device.normal)
        return kout

    if device.type == "Transmissive Grating":
        kout = kin + device.momentum_transfer
        return kout

    if device.type == "Transmission Telescope for CPA":
        kout = util.get_telescope_kout(optical_axis=device.lens_axis,
                                       kin=kin)
        return kout

    if device.type == "Total Reflection Mirror":
        kout = util.get_mirror_kout(kin=kin,
                                    normal=device.normal, )
        return kout


def get_kout_multi_device(device_list, kin):
    """
    Get the output momentum vectors from each device.

    :param device_list:
    :param kin:
    :return:
    """

    # Create a variable for the kout list.
    # The reason to use is numpy array is that it's easy to determine the
    # total number of kouts generates and with numpy array, it might be more
    # efficient.
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the output wave vector
        kout_list[idx + 1] = get_kout_single_device(device=device,
                                                    kin=kout_list[idx])

    return kout_list


def get_lightpath(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

        if device.type == "Transmission Telescope for CPA":
            intersection_list.append(util.get_image_from_telescope_for_cpa(object_point=intersection_list[-1],
                                                                           lens_axis=device.lens_axis,
                                                                           lens_position=device.lens_position,
                                                                           focal_length=device.focal_length))
            # Find the path length
            # displacement = intersection_list[-1] - intersection_list[-2]
            # path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the output wave vector
            kout_list.append(util.get_telescope_kout(optical_axis=device.lens_axis,
                                                     kin=kout_list[-1]))
        if device.type == "Total Reflection Mirror":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(get_kout_single_device(device=device, kin=kout_list[-1]))

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   surface_point=final_plane_point,
                                                   normal=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

    return np.vstack(intersection_list), np.vstack(kout_list), path_length
