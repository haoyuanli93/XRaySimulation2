import numpy as np
import time

from XRaySimulation import util
from datetime import datetime

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi
two_pi = 2 * pi


class saseSource:

    def __init__(self,
                 nx=32, ny=32, nz=1024,
                 dx=4, dy=4, dz=0.1, Ec=9.8,
                 mean_pulse_energy_uJ=600,  # uJ.  10 uJ / 0.5eV * 30 eV = 600 uJ
                 pulse_energy_sigma_uJ=100,  # SASE energy fluctuation. Not the fluctuation after the xpp mono
                 n_gaussian=500,
                 mode_size_x=200,
                 mode_size_y=200,
                 mode_size_z=0.15 * util.c,
                 mode_center_spread_x=20,
                 mode_center_spread_y=20,
                 mode_center_spread_z=20 * util.c,
                 x0=None):

        self.mean_pulse_energy_uJ = mean_pulse_energy_uJ
        self.pulse_energy_sigma_uJ = pulse_energy_sigma_uJ

        self.n_gaussian = n_gaussian
        self.modeSizeX = mode_size_x
        self.modeSizeY = mode_size_y
        self.modeSizeZ = mode_size_z
        self.modeCenterSpreadX = mode_center_spread_x
        self.modeCenterSpreadY = mode_center_spread_y
        self.modeCenterSpreadZ = mode_center_spread_z
        self.x0 = x0

        self.nx = nx  # For y rotation_axis in XPP frame
        self.ny = ny  # For x rotation_axis in XPP frame
        self.nz = nz  # For z rotation_axis in XPP frame

        self.dx = dx
        self.dy = dy
        self.dz = dz  # The total pulse duration is 50 fs in this simulation

        wave_vec_len = util.kev_to_wavevec_length(energy=Ec)

        self.wave_vec_len = wave_vec_len
        self.wave_vec = np.array([0., 0., wave_vec_len], dtype=np.float64)

        # Get kin_array grid
        (xCoor, yCoor, zCoor, tCoor, kxCoor, kyCoor, kzCoor,
         ExCoor, EyCoor, EzCoor) = util.get_coordinate(
            nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, k0=wave_vec_len)

        self.xCoor = xCoor
        self.yCoor = yCoor
        self.zCoor = zCoor
        self.tCoor = tCoor
        self.kxCoor = kxCoor
        self.kyCoor = kyCoor
        self.kzCoor = kzCoor
        self.ExCoor = ExCoor
        self.EyCoor = EyCoor
        self.EzCoor = EzCoor

        self.coor_dict = {'xCoor': xCoor, 'yCoor': yCoor, 'zCoor': zCoor, 'tCoor': tCoor,
                          'kxCoor': kxCoor, 'kyCoor': kyCoor, 'kzCoor': kzCoor,
                          'ExCoor': ExCoor, 'EyCoor': EyCoor, 'EzCoor': EzCoor, }

    def get_kin_grid(self):
        kinGrid = np.zeros((self.nx, self.ny, self.nz, 3))
        kinGrid[:, :, :, 0] = self.kxCoor[:, np.newaxis, np.newaxis]
        kinGrid[:, :, :, 1] = self.kyCoor[np.newaxis, :, np.newaxis]
        kinGrid[:, :, :, 2] = self.kzCoor[np.newaxis, np.newaxis, :]

        return kinGrid

    def get_sase_1d(self, randomSeed=None):

        if randomSeed is None:
            np.random.seed(datetime.now().timestamp())

        sase_field = get_gaussian_mode_sum_1d(nz=self.nz,
                                              dz=self.dz,
                                              nGaussian=self.n_gaussian,
                                              modeSizeZ=self.modeSizeZ,
                                              modeCenterSpreadZ=self.modeCenterSpreadZ,
                                              k0=self.wave_vec_len,
                                              randomSeed=randomSeed)

        # target energy
        pulse_energy = min(5.0, np.random.normal(loc=self.mean_pulse_energy_uJ,
                                                 scale=self.pulse_energy_sigma_uJ))

        # Apply scaling to obtain the corresponding pulse energy
        norm = np.linalg.norm(np.abs(sase_field))
        scaling = np.sqrt(pulse_energy) / norm
        scaling = complex(scaling, 0)

        return sase_field * scaling, pulse_energy

    def get_sase_3d(self, randomSeed=None):
        if randomSeed is None:
            np.random.seed(datetime.now().timestamp())
        sase_field = getGaussianModeSum(nx=self.nx,
                                        ny=self.ny,
                                        nz=self.nz,
                                        dx=self.dx,
                                        dy=self.dy,
                                        dz=self.dz,
                                        nGaussian=self.n_gaussian,
                                        modeSizeX=self.modeSizeX,
                                        modeSizeY=self.modeSizeY,
                                        modeSizeZ=self.modeSizeZ,
                                        modeCenterSpreadX=self.modeCenterSpreadX,
                                        modeCenterSpreadY=self.modeCenterSpreadY,
                                        modeCenterSpreadZ=self.modeCenterSpreadZ,
                                        k0=self.wave_vec_len,
                                        randomSeed=randomSeed)

        print("The electric field is not normalized properly. Do not use this function.")

        return sase_field


class GaussianPulse3D:
    def __init__(self):
        self.klen0 = 100.
        self.x0 = np.zeros(3, dtype=np.float64)
        self.k0 = np.zeros(3, dtype=np.float64)
        self.n = np.zeros(3, dtype=np.float64)

        self.omega0 = 20000.  # PHz

        # Basically, this mean that initially, we are in the frame where
        # different components of the pulse decouple. Then we rotate
        # back in to the lab frame.
        self.sigma_x = 0.1  # fs
        self.sigma_y = 33.  # fs
        self.sigma_z = 33.  # fs

        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))

        # Intensity. Add a coefficient for overall intensity.
        self.scaling = 1.

        # Polarization
        self.polar = np.array([1., 0., 0.], dtype=np.complex128)

    def set_pulse_properties(self, central_energy, polar, sigma_x, sigma_y, sigma_z, x0):
        """
        Set the pulse properties assuming that the pulse is propagating along
        the positive z direction.
        :param central_energy:
        :param polar:
        :param sigma_x: The unit is fs. However, in the function, it's converted into um.
        :param sigma_y: The unit is fs. However, in the function, it's converted into um.
        :param sigma_z: The unit is fs. However, in the function, it's converted into um.
        :param x0:
        :return:
        """
        # Get the corresponding wave vector
        self.klen0 = util.kev_to_wavevec_length(energy=central_energy)

        self.polar = np.array(np.reshape(polar, (3,)),
                              dtype=np.complex128)

        self.k0 = np.array([0., 0., self.klen0])
        self.n = self.k0 / np.linalg.norm(self.k0)
        self.omega0 = self.klen0 * util.c
        self.x0 = x0

        # Initialize the sigma matrix
        self.set_sigma_mat(sigma_x=sigma_x,
                           sigma_y=sigma_y,
                           sigma_z=sigma_z)

        # Normalize the pulse such that the incident total energy is 1 au
        # Then in this case, if one instead calculate the square L2 norm of the spectrum, then
        # the value is 8 * pi ** 3
        self.scaling = 2. * np.sqrt(2) * np.power(np.pi, 0.75) * np.sqrt(sigma_x *
                                                                         sigma_y *
                                                                         sigma_z *
                                                                         (util.c ** 3))

    def set_sigma_mat(self, sigma_x, sigma_y, sigma_z):
        """
        Notice that this function assumes that the pulse propagates long the z direction.

        :param sigma_x:
        :param sigma_y:
        :param sigma_z:
        :return:
        """

        self.sigma_x = sigma_x  # sigma_t
        self.sigma_y = sigma_y  # sigma_t  # fs
        self.sigma_z = sigma_z  # fs
        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))
        self.sigma_mat *= util.c ** 2

    def shift(self, displacement):
        """

        :param displacement:
        :return:
        """
        self.x0 += displacement

    def rotate(self, rot_mat):
        """
        Rotate the pulse with respect to the origin

        :param rot_mat:
        :return:
        """
        self.x0 = np.dot(rot_mat, self.x0)
        self.k0 = np.dot(rot_mat, self.k0)
        self.polar = np.dot(rot_mat, self.polar)

        self.sigma_mat = np.dot(np.dot(rot_mat, self.sigma_mat), rot_mat.T)

    def rotate_wrt_point(self, rot_mat, ref_point):
        """
        This is a function designed
        :param rot_mat:
        :param ref_point:
        :return:
        """
        # Step 1: shift with respect to that point
        self.shift(displacement=-ref_point)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=ref_point)


# --------------------------------------------------------------
#          Get spectrum
# --------------------------------------------------------------
def get_gaussian_pulse_spectrum(k_grid, sigma_mat, scaling, k0):
    # Get the momentum difference
    dk = k0[np.newaxis, :] - k_grid

    # Get the quadratic term
    quad_term = - (dk[:, 0] * sigma_mat[0, 0] * dk[:, 0] + dk[:, 0] * sigma_mat[0, 1] * dk[:, 1] +
                   dk[:, 0] * sigma_mat[0, 2] * dk[:, 2] +
                   dk[:, 1] * sigma_mat[1, 0] * dk[:, 0] + dk[:, 1] * sigma_mat[1, 1] * dk[:, 1] +
                   dk[:, 1] * sigma_mat[1, 2] * dk[:, 2] +
                   dk[:, 2] * sigma_mat[2, 0] * dk[:, 0] + dk[:, 2] * sigma_mat[2, 1] * dk[:, 1] +
                   dk[:, 2] * sigma_mat[2, 2] * dk[:, 2]) / 2.

    # if quad_term >= -200:
    magnitude = scaling * (np.exp(quad_term) + 0.j)
    return magnitude


def get_square_pulse_spectrum(k_grid, k0, a_val, b_val, c_val, scaling):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j
    spectrum *= scaling

    return spectrum


def get_square_pulse_spectrum_smooth(k_grid, k0, a_val, b_val, c_val, scaling, sigma):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j

    spectrum *= scaling

    # Add the Gaussian filter
    tmp = - (dk[:, 0] ** 2 + dk[:, 1] ** 2 + dk[:, 2] ** 2) * sigma ** 2 / 2.
    gaussian = np.exp(tmp)

    return np.multiply(spectrum, gaussian)


# Unit is fs, um, and keV
def getGaussianModeSum(nx=128, ny=128, nz=1024,
                       dx=4, dy=4, dz=0.1,
                       nGaussian=1000,
                       modeSizeX=200,
                       modeSizeY=200,
                       modeSizeZ=0.1 * 299792458. * 1e-9,  # Pulse coherence lenght of 100 as to um
                       modeCenterSpreadX=10,   # you can play with this prameter a bit. Anything between 1 um to 20 um is possible physically.
                       modeCenterSpreadY=10,  # you can play with this prameter a bit. Anything between 1 um to 20 um is possible physically.
                       modeCenterSpreadZ=10 * 299792458. * 1e-9,  # 10fs pulse duration to um
                       k0=12.4 / 10 * 1e4,
                       randomSeed=41):
    """

    :param nx:  number of pixels along x rotation_axis (horizontal)
    :param ny:  number of pixels along y rotation_axis (vertical pointing to the roof)
    :param nz:  Number of pixel along the z rotation_axis (beam propagation direection)
    :param dx:   Pixel size in um
    :param dy:   Pixel size in um
    :param dz:   pixel size in um
    :param nGaussian:   Number of Gaussian mode to add to. Anything larger than 1000 is okay visually.
    :param modeSizeX:   Size of the Gausssian mode along the x rotation_axis
    :param modeSizeY:   Size of the Gausssian mode along the y rotation_axis
    :param modeSizeZ:   Size of the Gausssian mode along the z rotation_axis
    :param modeCenterSpreadX:   The FWHM if the spread of the cetner of the mode along X rotation_axis
    :param modeCenterSpreadY:   The FWHM if the spread of the cetner of the mode along y rotation_axis
    :param modeCenterSpreadZ:   The FWHM if the spread of the cetner of the mode along z rotation_axis
    :param k0:     Center wave-vector = 12.4 / E_keV * 1e4   (i.e. angular wave-number of the carrier frequency in um^-1)
    :param randomSeed:  Random seed.
    :return:
    """

    # Generate a series of electric field mode
    np.random.seed(randomSeed)
    modeCenter = np.random.rand(nGaussian, 3) - 0.5
    modeCenter[:, 0] *= modeCenterSpreadX
    modeCenter[:, 1] *= modeCenterSpreadY
    modeCenter[:, 2] *= modeCenterSpreadZ

    modeMagnitude = np.random.rand(nGaussian) + 0.1
    modePhaseCenter = np.random.rand(nGaussian) * np.pi * 2

    # Electric Field
    eField = np.zeros((nx, ny, nz), dtype=np.complex128)

    for modeIdx in range(nGaussian):
        # Create the mode
        modeField = np.ones((nx, ny, nz), dtype=np.float64)

        modeField *= np.exp(
            - np.square(np.arange(- nx // 2, nx - nx // 2) * dx - modeCenter[modeIdx, 0])
            / 2. / modeSizeX ** 2)[:, np.newaxis, np.newaxis]

        modeField *= np.exp(
            - np.square(np.arange(- ny // 2, ny - ny // 2) * dy - modeCenter[modeIdx, 1])
            / 2. / modeSizeY ** 2)[np.newaxis, :, np.newaxis]

        modeField *= np.exp(
            - np.square(np.arange(- nz // 2, nz - nz // 2) * dz - modeCenter[modeIdx, 2])
            / 2. / modeSizeZ ** 2)[np.newaxis, np.newaxis, :]

        modeField *= modeMagnitude[modeIdx] / modeSizeX / modeSizeY / modeSizeZ / np.power(np.pi * 2, 1.5)

        # Create the phase
        modePhase = np.arange(nz) * dz * k0 + modePhaseCenter[modeIdx]

        # Add the mode to the electric field
        eField.real += modeField * np.cos(modePhase)[np.newaxis, np.newaxis, :]
        eField.imag += modeField * np.sin(modePhase)[np.newaxis, np.newaxis, :]

    # Remove the overall carry frequency
    eField *= np.exp(-1.j * np.arange(nz) * dz * k0)[np.newaxis, np.newaxis, :]

    return eField


def get_gaussian_mode_sum_1d(nz, dz, nGaussian=50, modeSizeZ=0.9, modeCenterSpreadZ=1.5, k0=100, randomSeed=41):
    """

    :param nz:
    :param dz:
    :param nGaussian:
    :param modeSizeZ:
    :param modeCenterSpreadZ:
    :param k0:
    :param randomSeed:
    :return:
    """

    # Generate a series of electric field mode
    np.random.seed(randomSeed)
    modeCenter = np.random.rand(nGaussian) - 0.5
    modeCenter[:] *= modeCenterSpreadZ

    modeMagnitude = np.random.rand(nGaussian) + 0.1
    modePhaseCenter = np.random.rand(nGaussian) * np.pi * 2

    # Electric Field
    eField = np.zeros(nz, dtype=np.complex128)
    coor = np.linspace(start=-nz * dz / 2, stop=nz * dz / 2, num=nz)
    for modeIdx in range(nGaussian):
        modeField = (np.exp(- np.square(coor - modeCenter[modeIdx]) / 2. / modeSizeZ ** 2)
                     * modeMagnitude[modeIdx] / modeSizeZ / np.sqrt(np.pi))

        # Create the phase
        modePhase = np.arange(nz) * dz * k0 + modePhaseCenter[modeIdx]

        # Add the mode to the electric field
        eField.real += modeField * np.cos(modePhase)
        eField.imag += modeField * np.sin(modePhase)

    # Remove the overall carry frequency
    eField *= np.exp(-1.j * np.arange(nz) * dz * k0)

    return eField


# ---------------------------------------------
#    For the data science project
# ---------------------------------------------
def get_1D_GaussianPulse_array(pulseNum=120,
                               nk=300,
                               dk_keV=1e-5,
                               nGaussian=10,
                               modeSize_keV=0.1e-3,
                               modeCenterSpread_keV=0.5e-3,
                               pulseEnergyCenter_uJ=3,
                               pulseEnergySigma_uJ=1,  # Follow a logNormal Distribution
                               ):
    # Get a random seed based on current time
    randomSeed = int(time.time() * 1e6) % 65536
    np.random.seed(randomSeed)

    # Get the pulse spectrum holder
    spectrumHolder = np.zeros((pulseNum, nk), dtype=np.complex128)

    # Get Randomly generate some Gaussian function centers
    centers = (np.random.rand(pulseNum, nGaussian) - 0.5) * modeCenterSpread_keV
    widths = (np.random.rand(pulseNum, nGaussian) + 0.5) * modeSize_keV
    magnitude = (np.random.rand(pulseNum, nGaussian) + 1e-2) / np.sqrt(widths * dk_keV)
    phase = np.exp(1.j * (np.random.rand(pulseNum, nGaussian) * np.pi * 2))

    k_range = nk * dk_keV
    k_array_keV = np.linspace(-k_range / 2, k_range / 2, nk)

    # Loop through the mode number to get the pulse
    for modeIdx in range(nGaussian):
        tmp = - np.square(k_array_keV[np.newaxis, :] - centers[:, modeIdx, np.newaxis]) / 2
        tmp /= np.square(widths[:, modeIdx, np.newaxis])
        tmp = np.exp(tmp)
        spectrumHolder[:, :] += (magnitude[:, modeIdx, np.newaxis] * phase[:, modeIdx, np.newaxis]) * tmp

    pulseEnergy = np.random.lognormal(mean=pulseEnergyCenter_uJ, sigma=pulseEnergySigma_uJ, size=pulseNum)
    energy_normalization = np.sqrt(np.sum(np.square(np.abs(spectrumHolder)), axis=-1) * dk_keV)
    energy_normalization = (np.sqrt(pulseEnergy) / energy_normalization).astype(np.complex128)

    spectrumHolder = spectrumHolder * energy_normalization[:, np.newaxis]

    k_vec = np.zeros((nk, 3))
    k_vec[:, 2] = util.kev_to_wavevec_length(k_array_keV)

    return spectrumHolder, pulseEnergy, k_array_keV, k_vec
