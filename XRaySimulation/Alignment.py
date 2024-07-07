import numpy as np

from XRaySimulation import util
from XRaySimulation import RockingCurve


def align_crystal_around_axis(crystal, kin, initial_angle, rotation_axis, bandwidth_keV=0.5e-3, fwhm_center=True,
                              polarization='s', rot_center=None, rot_crystal=True, iteration=3):
    if crystal.type == 'Crystal: Bragg Reflection':
        if rot_center is None:
            rot_center = np.copy(crystal.surface_point)
        rock_fun = RockingCurve.get_rocking_curve_bandwidth_sum
    elif crystal.type == "Channel cut with two surfaces":
        if rot_center is None:
            rot_center = np.copy(crystal.crystal_list[0].surface_point)
        rock_fun = RockingCurve.get_rocking_curve_channelcut_bandwidth_sum
    else:
        print("crystal has to be either 'Crystal: Bragg Reflection' or 'Channel cut with two surfaces' ")
        return 1

    # Get the wavevector array with the corresponding kin and bandwidth
    k_num = 100
    kin_len = np.linalg.norm(kin)
    kin_direction = kin / kin_len

    # Get the wave-vector array
    energy_array = np.linspace(start=-bandwidth_keV / 2., stop=bandwidth_keV / 2., num=k_num)
    klen_array = util.kev_to_wavevec_length(energy_array) + kin_len
    kin_array = np.zeros((k_num, 3))
    kin_array[:, :] = kin_direction[np.newaxis, :]
    kin_array[:, :] *= klen_array[:, np.newaxis]

    # print(kin_array.shape)

    # Search for the rocking curve around this
    scan_range = [initial_angle - np.deg2rad(2), initial_angle + np.deg2rad(2)]
    scan_num = 400
    for idx in range(iteration):
        (angles, reflect_s, reflect_p, b, kout_array
         ) = rock_fun(kin_array=kin_array, rotation_axis=rotation_axis, crystal=crystal,
                      scan_range=scan_range, scan_number=scan_num)

        if polarization == 's':
            # Find the best location
            reflectivity = np.mean(np.square(np.abs(reflect_s)) / np.abs(b), axis=0)
        elif polarization == 'p':
            reflectivity = np.mean(np.square(np.abs(reflect_p)) / np.abs(b), axis=0)
        else:
            print("polarization can only be s or p")
            return 1

        # Find the peak angle
        angle_idx = np.argmax(reflectivity)
        peak_angle = angles[angle_idx]

        # Update the searching range
        scan_range = [peak_angle - np.deg2rad(2 / 10 ** (idx * 2 + 1)),
                      peak_angle + np.deg2rad(2 / 10 ** (idx * 2 + 1))]
        # print(idx, kin_array.shape)

    if fwhm_center:
        # Get the FWHM center based on the last search
        fwhm, angle_adjust = util.get_fwhm(coordinate=angles,
                                           curve_values=reflectivity,
                                           center=True)
        peak_angle = angle_adjust
        angle_idx = np.argmin(np.abs(angles - angle_adjust))

    rot_mat = util.get_rotmat_around_axis(angleRadian=peak_angle, axis=rotation_axis)
    kout = kout_array[k_num // 2, angle_idx,]

    if rot_crystal:
        crystal.rotate_wrt_point(rot_mat=rot_mat,
                                 ref_point=rot_center)

    return rot_mat, fwhm, kout, angle_adjust, angles, reflectivity


def align_grating_normal_direction(grating, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, grating.normal) / np.linalg.norm(axis) / np.linalg.norm(grating.normal)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, grating.normal)

    if np.dot(new_h, axis) < 0:
        rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    grating.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=grating.surface_point)


def align_telescope_optical_axis(telescope, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, telescope.lens_axis) / np.linalg.norm(axis) / np.linalg.norm(telescope.lens_axis)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, telescope.lens_axis)

    if np.dot(new_h, axis) < 0:
        rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    telescope.rotate_wrt_point(rot_mat=rot_mat,
                               ref_point=telescope.lens_point)
