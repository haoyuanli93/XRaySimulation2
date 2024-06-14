import numpy as np
from scipy import interpolate
from skimage.restoration import unwrap_phase

from XRaySimulation import util


def add_propagate_phase(kx, ky, kz, distance, spectrum):
    """

    :param kx:
    :param ky:
    :param kz:
    :param distance:
    :param spectrum:
    :return:
    """
    nx = kx.shape[0]
    ny = ky.shape[0]
    nz = kz.shape[0]

    # Get time
    t = distance / util.c

    # Get frequency
    omega = np.zeros((nx, ny, nz))
    omega += np.square(kx[:, np.newaxis, np.newaxis])
    omega += np.square(ky[np.newaxis, :, np.newaxis])
    omega += np.square(kz[np.newaxis, np.newaxis, :])
    omega = np.sqrt(omega) * util.c

    # get phase, to save memory, I'll just use omega
    omega *= t
    omega -= kz[np.newaxis, np.newaxis, :] * distance

    # Get the phase
    np.multiply(spectrum,
                np.exp(1.j * omega),
                out=spectrum,
                dtype=np.complex128)


def add_lens_transmission_function(x, y, kz, fx, fy, xy_kz_field, n=complex(1.0, 0)):
    """

    :param x:
    :param y:
    :param kz:
    :param fx:
    :param fy:
    :param xy_kz_field:
    :param n:
    :return:
    """
    phaseX = np.exp(complex(- n.imag / (1. - n.real), -1) * np.outer(np.square(x), kz / 2. / fx))
    phaseY = np.exp(complex(- n.imag / (1. - n.real), -1) * np.outer(np.square(y), kz / 2. / fy))

    # Add the transmission function to the electric field along each direction
    np.multiply(xy_kz_field, phaseX[:, np.newaxis, :], out=xy_kz_field)
    np.multiply(xy_kz_field, phaseY[np.newaxis, :, :], out=xy_kz_field)


def get_flat_wavevector_array(kx, ky, kz):
    nx = kx.shape[0]
    ny = ky.shape[0]
    nz = kz.shape[0]

    kVecArray = np.zeros((nx, ny, nz, 3))
    kVecArray[:, :, :, 0] = kx[:, np.newaxis, np.newaxis]
    kVecArray[:, :, :, 1] = ky[np.newaxis, :, np.newaxis]
    kVecArray[:, :, :, 2] = kz[np.newaxis, np.newaxis, :]

    return np.reshape(kVecArray, (nx * ny * nz, 3))


def get_interpolated_eField(kvec_array, coor_dict, efield_array, k0, mode, coor_info_new=None, affine_mat=None):
    # Because almost for sure we have the screen facing towards the Z direction.
    # I only consider the interpolation that generate the electric field in the x,y,t or x,y,z coordinate
    # the same coordiante as the inital pulse.
    # For the theory, find it in Haoyuan Li's thesis

    # The purpose of this interpolation is multiple
    # 1. Demonstrate the pulse front tilt issue after asymmetric channel-cut crystals
    #        for this purpose, the interpolation is within the xpp x xpp z plane, or the y-z plane in this simulation
    # 2. Get the accurate electric field and use that to calculate the TG fringe
    #        for this purpose, the interpolation is within the xpp x xpp z plane, or the y-z plane in this simulation
    # 3. Get the probe pulse electric field and see its spatial overlap with the TG fringe
    #        for this purpose, the interpolation is within the xpp y xpp z plane, or the x-z plane in this simulation

    # Below, I try to implement two kinds of interpolation
    # one is the 2D interpolation. The interpolation dimension is the same as that
    # explained above. The other one is the 3D interpolation.
    # The 3D interpolation is more time-consuming and more accurate.
    # Ideally, in one simulation, I would need to compare the two cases and
    # choose the correct one to implement.

    # This function is so fundamental, I believe I need to create a basic function
    # for this purpose.

    # 2024-04-04 implement the 3D interpolation with low efficiency first
    # Even though the calculation is less efficient, it is more universal and maybe more compatible with the simulation
    if mode == "xyz 3D":
        # print("Interpolate the electric field such that after the intpolation")
        # print("the three axes of the array are parallel to that of the x,y,z axes.")
        # For VCC pulse, we want to interpolate within the yz plane, or the xpp x - xpp z plane.

        oldShape = np.array(efield_array.shape)

        # Step 1: Get the interpolation matrix
        # Get the spatial grid according to my note
        u1 = (kvec_array[oldShape[0] // 2 + 1, oldShape[1] // 2, oldShape[2] // 2]
              - kvec_array[oldShape[0] // 2 - 1, oldShape[1] // 2, oldShape[2] // 2]) / 2.
        u2 = (kvec_array[oldShape[0] // 2, oldShape[1] // 2 + 1, oldShape[2] // 2]
              - kvec_array[oldShape[0] // 2, oldShape[1] // 2 - 1, oldShape[2] // 2]) / 2.
        u3 = (kvec_array[oldShape[0] // 2, oldShape[1] // 2, oldShape[2] // 2 + 1]
              - kvec_array[oldShape[0] // 2, oldShape[1] // 2, oldShape[2] // 2 - 1]) / 2.

        u_mat = np.zeros((3, 3))
        u_mat[0] = u1
        u_mat[1] = u2
        u_mat[2] = u3
        u_mat[np.abs(u_mat) < 1e-10] = 0
        u_mat /= 2 * np.pi
        u_mat_inv = np.linalg.inv(u_mat)

        # Get the position grid for interpolation
        if coor_info_new:
            nx = coor_info_new['nx']
            ny = coor_info_new['ny']
            nz = coor_info_new['nz']
            dx = coor_info_new['dx']
            dy = coor_info_new['dy']
            dz = coor_info_new['dz']


        else:
            # Otherwise, calculate the new coordinate by analyzing the current situation.
            # step 1: Get the boundary of old space in the new coordinate system
            corners = np.array([[coor_dict['xCoor'][0], coor_dict['yCoor'][0], coor_dict['zCoor'][0]],
                                [coor_dict['xCoor'][0], coor_dict['yCoor'][0], coor_dict['zCoor'][-1]],
                                [coor_dict['xCoor'][0], coor_dict['yCoor'][-1], coor_dict['zCoor'][0]],
                                [coor_dict['xCoor'][0], coor_dict['yCoor'][-1], coor_dict['zCoor'][-1]],
                                [coor_dict['xCoor'][-1], coor_dict['yCoor'][0], coor_dict['zCoor'][0]],
                                [coor_dict['xCoor'][-1], coor_dict['yCoor'][0], coor_dict['zCoor'][-1]],
                                [coor_dict['xCoor'][-1], coor_dict['yCoor'][-1], coor_dict['zCoor'][0]],
                                [coor_dict['xCoor'][-1], coor_dict['yCoor'][-1], coor_dict['zCoor'][-1]],
                                ])
            new_corners = np.dot(u_mat_inv, corners.T).T

            # Step 2: Use the original resolution. Get the new pixel number
            dx = coor_dict['xCoor'][1] - coor_dict['xCoor'][0]
            dy = coor_dict['yCoor'][1] - coor_dict['yCoor'][0]
            dz = coor_dict['zCoor'][1] - coor_dict['zCoor'][0]

            (nx, ny, nz) = (np.max(new_corners, axis=0) - np.min(new_corners, axis=0)) / np.array([dx, dy, dz])
            nx = int(nx)
            ny = int(ny)
            nz = int(nz)

        # ---------------------------------------------------
        # Get the new coordinate system
        # ---------------------------------------------------
        (xCoor, yCoor, zCoor, tCoor,
         kxCoor, kyCoor, kzCoor,
         ExCoor, EyCoor, EzCoor) = util.get_coordinate(nx=nx, ny=ny, nz=nz,
                                                       dx=dx, dy=dy, dz=dz,
                                                       k0=np.linalg.norm(k0))

        new_coor_dict = {'xCoor': xCoor, 'yCoor': yCoor, 'zCoor': zCoor, 'tCoor': tCoor,
                         'kxCoor': kxCoor, 'kyCoor': kyCoor, 'kzCoor': kzCoor,
                         'ExCoor': ExCoor, 'EyCoor': EyCoor, 'EzCoor': EzCoor, }

        new_position_grid = np.zeros((nx, ny, nz, 3))
        new_position_grid[:, :, :, 0] = xCoor[:, np.newaxis, np.newaxis]
        new_position_grid[:, :, :, 1] = yCoor[np.newaxis, :, np.newaxis]
        new_position_grid[:, :, :, 2] = zCoor[np.newaxis, np.newaxis, :]

        new_position_grid_for_interpolation = np.reshape(new_position_grid, (nx * ny * nz, 3))
        del new_position_grid
        new_position_grid_for_interpolation = np.dot(u_mat, new_position_grid_for_interpolation.T).T

        field_fit_mag = interpolate.interpn(points=(np.arange(-oldShape[0] // 2, oldShape[0] // 2) / oldShape[0],
                                                    np.arange(-oldShape[1] // 2, oldShape[1] // 2) / oldShape[1],
                                                    np.arange(-oldShape[2] // 2, oldShape[2] // 2) / oldShape[2],),
                                            values=np.abs(efield_array),
                                            xi=new_position_grid_for_interpolation,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=0.)

        field_fit_phase = interpolate.interpn(points=(np.arange(-oldShape[0] // 2, oldShape[0] // 2) / oldShape[0],
                                                      np.arange(-oldShape[1] // 2, oldShape[1] // 2) / oldShape[1],
                                                      np.arange(-oldShape[2] // 2, oldShape[2] // 2) / oldShape[2],),
                                              values=unwrap_phase(np.angle(efield_array)),
                                              xi=new_position_grid_for_interpolation,
                                              method='linear',
                                              bounds_error=False,
                                              fill_value=0.)

        field_fit = field_fit_mag * np.exp(1.j * field_fit_phase)
        field_fit = np.reshape(field_fit, (nx, ny, nz))
        return field_fit, new_coor_dict

    elif mode == "beam frame 3D":
        pass
    elif (mode == "vcc") or (mode == "yz"):
        pass
        print("No interpolation is implemented. This option is not implemented yet.")
    elif mode == "TG pump":
        print("No interpolation is implemented. This option is not implemented yet.")
        pass
    elif mode == "TG probe":
        print("No interpolation is implemented. This option is not implemented yet.")
        pass
    else:
        print("No interpolation is applied. Currently this function cannot handle a general interpolation request.")
        print("Please check the source code for this function to understand the current capability boundary.")


def get_intensity_on_YAG(intensity, intensity_coor, intensity_loc, pixel_coor):
    nx, ny = (pixel_coor['xCoor'].shape[0], pixel_coor['yCoor'].shape[0])
    new_position_grid = np.zeros((nx, ny, 2))
    new_position_grid[:, :, 0] = pixel_coor['xCoor'][:, np.newaxis]
    new_position_grid[:, :, 1] = pixel_coor['yCoor'][np.newaxis, :]
    new_position_grid_for_interpolation = np.reshape(new_position_grid, (nx * ny, 2))
    del new_position_grid

    yag_image = interpolate.interpn(points=(intensity_coor['xCoor'] + intensity_loc[0],
                                            intensity_coor['yCoor'] + intensity_loc[1],),
                                    values=intensity,
                                    xi=new_position_grid_for_interpolation,
                                    method='nearest',
                                    bounds_error=False,
                                    fill_value=0.)
    return yag_image


def get_gaussian_on_yag(sigma_mat, beam_center, intensity, pixel_coor):
    nx, ny = (pixel_coor['xCoor'].shape[0], pixel_coor['yCoor'].shape[0])

    new_position_grid = np.zeros((nx, ny, 2))
    new_position_grid[:, :, 0] = pixel_coor['xCoor'][:, np.newaxis]
    new_position_grid[:, :, 1] = pixel_coor['yCoor'][np.newaxis, :]
    new_position_grid_for_interpolation = np.reshape(new_position_grid, (nx * ny, 2))

    # Get the intensity field
    term1 = new_position_grid[:, :] - beam_center[np.newaxis, :]
    term2 = np.dot(term1, np.linalg.inv(sigma_mat).T)
    term = - np.sum(np.multiply(term1, term2), axis=-1) / 2.

    # Get the Gaussian intensity
    scaling = intensity / np.pi / 2. / np.sqrt(np.linalg.det(sigma_mat))
    term *= scaling

    return np.reshape(term, newshape=(nx, ny))
