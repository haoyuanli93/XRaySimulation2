import numpy as np

from XRaySimulation import util


# ------------------------------------------------
#    Get reflectivity
# ------------------------------------------------
def get_bragg_reflectivity_fix_crystal(kin, crystal):
    """
    Calculate the reflectivity with a fixed crystal.
    
    :param kin: wave vector array.  Numpy array of shape (normal, 3)
    :param crystal:
    :return:
    """

    chi_dict = crystal.chi_dict
    thickness = crystal.thickness
    crystal_h = crystal.h
    normal = crystal.normal

    # Extract the parameter
    chi0 = chi_dict["chi0"]
    chih_sigma = chi_dict["chih"]
    chihbar_sigma = chi_dict["chih"]
    chih_pi = chi_dict["chih_pi"]
    chihbar_pi = chi_dict["chih_pi"]

    # ----------------------------------------------
    #    Get reflected wave-vectors
    # ----------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin, axis=-1)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin, normal)
    dot_kh = np.dot(kin, crystal_h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + np.dot(crystal_h, normal), klen_grid)

    b_factor = np.divide(gamma_0, gamma_h).astype(np.complex128)
    alpha = np.divide(2 * dot_kh + np.sum(np.square(crystal_h)), np.square(klen_grid))

    # Get momentum tranfer
    delta = np.multiply(klen_grid, -gamma_h - np.sqrt(gamma_h ** 2 - alpha))

    # Get the output wave-vector
    kout = np.copy(kin) + crystal_h[np.newaxis, :] + delta[:, np.newaxis] * normal[np.newaxis, :]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    alpha = alpha.astype(np.complex128)
    # Get alpha tidle
    alpha_tidle = (alpha * b_factor + chi0 * (complex(1., 0) - b_factor)) / complex(2., 0)

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** complex(2.0, 0) + b_factor * (chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = complex(1., 0) - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2., 0) - numerator)

    reflect_sigma = chih_sigma * b_factor * numerator / denominator

    # ------------------------------------------------------------
    # Step 3: Get the reflectivity for pi polarization
    # ------------------------------------------------------------
    # Notice that the polarization factor has been incorporated into the chi_pi

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + b_factor * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag
    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)
    reflect_pi = b_factor * chih_pi * numerator / denominator

    return reflect_sigma, reflect_pi, b_factor, kout


def get_bragg_reflectivity_per_entry(kin, thickness, crystal_h, normal, chi_dict):
    """
    Calculate the reflectivity for each element. 
    
    :param kin: wave vector array.  Numpy array of shape (normal, 3)
    :param thickness: float
    :param crystal_h: Numpy array of shape (normal, 3)
    :param normal: numpy array of shape (normal, 3)
    :param chi_dict: The dictionary for parameters of electric susceptability.
    :return:
    """

    # Extract the parameter
    chi0 = chi_dict["chi0"]
    chih_sigma = chi_dict["chih"]
    chihbar_sigma = chi_dict["chih"]
    chih_pi = chi_dict["chih_pi"]
    chihbar_pi = chi_dict["chih_pi"]

    # ----------------------------------------------
    #    Get reflected wave-vectors
    # ----------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin, axis=-1)

    # Get gamma and alpha and b
    dot_kn = np.sum(np.multiply(kin, normal), axis=-1)
    dot_kh = np.sum(np.multiply(kin, crystal_h), axis=-1)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + np.sum(np.multiply(crystal_h, normal), axis=-1), klen_grid)

    b_factor = np.divide(gamma_0, gamma_h).astype(np.complex128)
    alpha = np.divide(2 * dot_kh + np.sum(np.square(crystal_h), axis=-1), np.square(klen_grid))

    # Get momentum tranfer
    delta = np.multiply(klen_grid, -gamma_h - np.sqrt(gamma_h ** 2 - alpha))

    # Get the output wave-vector
    kout = np.copy(kin) + crystal_h + delta[:, np.newaxis] * normal

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    alpha = alpha.astype(np.complex128)
    # Get alpha tidle
    alpha_tidle = (alpha * b_factor + chi0 * (complex(1., 0) - b_factor)) / complex(2., 0)

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** complex(2.0, 0) + b_factor * (chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = complex(1., 0) - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2., 0) - numerator)

    reflect_sigma = chih_sigma * b_factor * numerator / denominator

    # ------------------------------------------------------------
    # Step 3: Get the reflectivity for pi polarization
    # ------------------------------------------------------------
    # Notice that the polarization factor has been incorporated into the chi_pi

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + b_factor * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag
    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)
    reflect_pi = b_factor * chih_pi * numerator / denominator

    return reflect_sigma, reflect_pi, b_factor, kout


# ------------------------------------------------
#    Get rocking curve
# ------------------------------------------------
def get_rocking_curve_around_axis(kin,
                                  scan_range,
                                  scan_number,
                                  rotation_axis,
                                  crystal, ):
    """

    :param kin:
    :param scan_range:
    :param scan_number:
    :param crystal:
    :return:
    """
    h_initial = np.copy(crystal.h)
    normal_initial = np.copy(crystal.normal)
    thickness = crystal.thickness
    chi_dict = crystal.chi_dict

    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning si111_angle
    angles = np.linspace(start=scan_range[0], stop=scan_range[1], num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.get_rotmat_around_axis(angleRadian=angles[idx], axis=rotation_axis)
        h_array[idx] = rot_mat.dot(h_initial)
        normal_array[idx] = rot_mat.dot(normal_initial)

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros_like(h_array, dtype=np.float64)
    kin_grid[:, 0] = kin[0]
    kin_grid[:, 1] = kin[1]
    kin_grid[:, 2] = kin[2]

    (reflect_sigma,
     reflect_pi,
     b_factor,
     kout) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                              thickness=thickness,
                                              crystal_h=h_array,
                                              normal=normal_array,
                                              chi_dict=chi_dict)

    return angles, reflect_sigma, reflect_pi, b_factor, kout


def get_rocking_curve_channelcut_around_axis(kin,
                                             rotation_axis,
                                             channelcut,
                                             scan_range,
                                             scan_number,
                                             ):
    """

    :param kin:
    :param channelcut:
    :param rotation_axis:
    :param scan_range:
    :param scan_number:
    :return:
    """

    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array_1 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_1 = np.zeros((scan_number, 3), dtype=np.float64)

    h_array_2 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_2 = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning si111_angle
    angles = np.linspace(start=scan_range[0], stop=scan_range[1], num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.get_rotmat_around_axis(angleRadian=angles[idx], axis=rotation_axis)
        # print(rot_mat)
        # print(np.linalg.det(rot_mat))

        h_array_1[idx] = rot_mat.dot(channelcut.crystal_list[0].h)
        normal_array_1[idx] = rot_mat.dot(channelcut.crystal_list[0].normal)

        h_array_2[idx] = rot_mat.dot(channelcut.crystal_list[1].h)
        normal_array_2[idx] = rot_mat.dot(channelcut.crystal_list[1].normal)

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros_like(h_array_1, dtype=np.float64)
    kin_grid[:, 0] = kin[0]
    kin_grid[:, 1] = kin[1]
    kin_grid[:, 2] = kin[2]

    (reflect_sigma_1,
     reflect_pi_1,
     b_factor_1,
     kout_1) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                                thickness=channelcut.crystal_list[0].thickness,
                                                crystal_h=h_array_1,
                                                normal=normal_array_1,
                                                chi_dict=channelcut.crystal_list[0].chi_dict)

    (reflect_sigma_2,
     reflect_pi_2,
     b_factor_2,
     kout_2) = get_bragg_reflectivity_per_entry(kin=kout_1,
                                                thickness=channelcut.crystal_list[1].thickness,
                                                crystal_h=h_array_2,
                                                normal=normal_array_2,
                                                chi_dict=channelcut.crystal_list[1].chi_dict)

    return (angles,
            reflect_sigma_1 * reflect_sigma_2,
            reflect_pi_1 * reflect_pi_2,
            b_factor_1 * b_factor_2,
            kout_2)


def get_rocking_curve_bandwidth_sum(kin_array,
                                    rotation_axis,
                                    crystal,
                                    scan_range,
                                    scan_number, ):
    k_num = kin_array.shape[0]
    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array_1 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_1 = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning si111_angle
    angles = np.linspace(start=scan_range[0], stop=scan_range[1], num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.get_rotmat_around_axis(angleRadian=angles[idx], axis=rotation_axis)

        h_array_1[idx] = rot_mat.dot(crystal.h)
        normal_array_1[idx] = rot_mat.dot(crystal.normal)

    # -------------------------------------------------------------
    #   Generate the 2D arrays for the k_vec and BraggG for the scan
    # -------------------------------------------------------------
    h_array_1_2d = np.zeros((k_num, scan_number, 3))
    normal_array_1_2d = np.zeros((k_num, scan_number, 3))


    h_array_1_2d[:, :, :] = h_array_1[np.newaxis, :, :]
    normal_array_1_2d[:, :, :] = normal_array_1[np.newaxis, :, :]


    h_array_new = np.reshape(h_array_1_2d, newshape=(k_num * scan_number, 3))
    normal_array_new = np.reshape(normal_array_1_2d, newshape=(k_num * scan_number, 3))

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros((k_num, scan_number, 3))
    kin_grid[:, :, :] = kin_array[:, np.newaxis, :]
    kin_grid = np.reshape(kin_grid, (k_num * scan_number, 3))

    # Maybe it is still faster if we choose to trade time with memory
    (reflect_sigma_1,
     reflect_pi_1,
     b_factor_1,
     kout_1) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                                thickness=crystal.thickness,
                                                crystal_h=h_array_new,
                                                normal=normal_array_new,
                                                chi_dict=crystal.chi_dict)

    return (angles,
            np.reshape(reflect_sigma_1, (k_num, scan_number)),
            np.reshape(reflect_pi_1, (k_num, scan_number)),
            np.reshape(b_factor_1, (k_num, scan_number)),
            np.reshape(kout_1, (k_num, scan_number, 3)))


def get_rocking_curve_channelcut_bandwidth_sum(kin_array,
                                               rotation_axis,
                                               crystal,
                                               scan_range,
                                               scan_number, ):
    """
    Calculate the rocking curve for a series of k_vec vectors

    :param kin_array:
    :param crystal
    :param scan_range:
    :param scan_number:
    :return:
    """
    k_num = kin_array.shape[0]
    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array_1 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_1 = np.zeros((scan_number, 3), dtype=np.float64)

    h_array_2 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_2 = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning si111_angle
    angles = np.linspace(start=scan_range[0], stop=scan_range[1], num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.get_rotmat_around_axis(angleRadian=angles[idx], axis=rotation_axis)

        h_array_1[idx] = rot_mat.dot(crystal.crystal_list[0].h)
        normal_array_1[idx] = rot_mat.dot(crystal.crystal_list[0].normal)

        h_array_2[idx] = rot_mat.dot(crystal.crystal_list[1].h)
        normal_array_2[idx] = rot_mat.dot(crystal.crystal_list[1].normal)

    # -------------------------------------------------------------
    #   Generate the 2D arrays for the k_vec and BraggG for the scan
    # -------------------------------------------------------------
    h_array_1_2d = np.zeros((k_num, scan_number, 3))
    h_array_2_2d = np.zeros((k_num, scan_number, 3))
    normal_array_1_2d = np.zeros((k_num, scan_number, 3))
    normal_array_2_2d = np.zeros((k_num, scan_number, 3))

    h_array_1_2d[:, :, :] = h_array_1[np.newaxis, :, :]
    h_array_2_2d[:, :, :] = h_array_2[np.newaxis, :, :]
    normal_array_1_2d[:, :, :] = normal_array_1[np.newaxis, :, :]
    normal_array_2_2d[:, :, :] = normal_array_2[np.newaxis, :, :]

    h_array_1 = np.reshape(h_array_1_2d, newshape=(k_num * scan_number, 3))
    h_array_2 = np.reshape(h_array_2_2d, newshape=(k_num * scan_number, 3))
    normal_array_1 = np.reshape(normal_array_1_2d, newshape=(k_num * scan_number, 3))
    normal_array_2 = np.reshape(normal_array_2_2d, newshape=(k_num * scan_number, 3))

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros((k_num, scan_number, 3))
    kin_grid[:, :, :] = kin_array[:, np.newaxis, :]
    kin_grid = np.reshape(kin_grid, (k_num * scan_number, 3))
    
    # Maybe it is still faster if we choose to trade time with memory
    (reflect_sigma_1,
     reflect_pi_1,
     b_factor_1,
     kout_1) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                                thickness=crystal.crystal_list[0].thickness,
                                                crystal_h=h_array_1,
                                                normal=normal_array_1,
                                                chi_dict=crystal.crystal_list[0].chi_dict)

    (reflect_sigma_2,
     reflect_pi_2,
     b_factor_2,
     kout_2) = get_bragg_reflectivity_per_entry(kin=kout_1,
                                                thickness=crystal.crystal_list[1].thickness,
                                                crystal_h=h_array_2,
                                                normal=normal_array_2,
                                                chi_dict=crystal.crystal_list[1].chi_dict)

    return (angles,
            np.reshape(reflect_sigma_1 * reflect_sigma_2, (k_num, scan_number)),
            np.reshape(reflect_pi_1 * reflect_pi_2, (k_num, scan_number)),
            np.reshape(b_factor_1 * b_factor_2, (k_num, scan_number)),
            np.reshape(kout_2, (k_num, scan_number, 3))
            )
