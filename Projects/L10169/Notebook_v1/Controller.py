import sys

sys.path.append("../../../")

import numpy as np

from XRaySimulation import Crystal, util, Pulse
from XRaySimulation.Machine import Motors

# The following modules are loaded as a temporary solution
import MotorStack
import controllerUtil

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

g1_period = 1  # um
g2_period = 1  # um


class XppController:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __init__(self, photon_kev=9.6, gpu=False, gpuModule=False):
        fwhm = 200  # um

        # Define a reference pulse for the alignment

        self.gaussian_pulse = Pulse.GaussianPulse3D()
        self.gaussian_pulse.set_pulse_properties(central_energy=photon_kev,
                                                 polar=[1., 0., 0.],
                                                 sigma_x=fwhm / 2. / np.sqrt(np.log(2)) / util.c,
                                                 sigma_y=fwhm / 2. / np.sqrt(np.log(2)) / util.c,
                                                 sigma_z=9.,
                                                 x0=np.array([0., -500e3, -30e6]))
        self.wavelength = np.pi * 2 / util.kev_to_wavevec_length(energy=photon_kev)

        # Step 1 Create all the optics and motors
        motors, optics = assemble_motors_and_optics()

        # Step 2 Create properties associate with each component
        self._motor_stacks = motors
        self._optics = optics

        self.mono_t1 = motors['mono t1']
        self.mono_t2 = motors['mono t2']

        self.t1 = motors['t1']
        self.t2 = motors['t2']
        self.t3 = motors['t3']
        self.t45 = motors['t45']
        self.t6 = motors['t6']
        self.g1 = motors['g1']
        self.g2 = motors['g2']
        self.sample = motors['sample']

        self.all_towers = [self.mono_t1, self.mono_t2,
                           self.t1, self.t2, self.t3, self.t45, self.t6,
                           self.g1, self.g2,
                           self.sample, ]
        # Install the crystal towers on the breadboard
        self.breadboard1 = Motors.Breadboard(hole_num_x=23, hole_num_z=55, gauge='metric')
        # controller.breadboard2 = Motors.Breadboard(hole_num_x=17, hole_num_z=17, gauge='metric')
        self.breadboard3 = Motors.Breadboard(hole_num_x=18, hole_num_z=18, gauge='metric')
        self.breadboard4 = Motors.Breadboard(hole_num_x=18, hole_num_z=18, gauge='metric')

        # Get the xpp mono
        install_xpp_mono(controller=self)

        # Install the motion stack
        insatll_motionStack_and_optics(controller=self)

        # Add the shutter
        self.cc_shutter = True
        self.vcc_shutter = True

        # Step 5 Add diodes
        # In the simulation, the normalization of the electric field is such that
        # np.sum(np.square(np.abs(e_field))) = pulse energy in uJ
        # therefore, this ratio is defined such that, for example, for ipm2, 10uJ => a value of 8000
        self.diode_ratio = {'ipm2': 8000 / 10, 'dg1': 3 / 10, 'd1': 3 / 10, 'd2': 3 / 10, 'd3': 3 / 10, 'd4': 3 / 10,
                            'd5': 3 / 10, 'd6': 3 / 10, 'pump': 3 / 10, 'probe': 3 / 10, }

        self.diode_noise_level = {'ipm2': 100, 'dg1': 0.01, 'd1': 0.01, 'd2': 0.01, 'd3': 0.01, 'd4': 0.01,
                                  'd5': 0.01, 'd6': 0.01, 'pump': 0.01, 'probe': 0.01, }

        # Step 6 Add cameras
        self.pixel_num_x = 2048
        self.pixel_num_y = 2048

        self.pixel_coor_x = np.linspace(-self.pixel_num_x * 6.5 / 3, self.pixel_num_x * 6.5 / 3, self.pixel_num_x)
        self.pixel_coor_y = np.linspace(-self.pixel_num_y * 6.5 / 3, self.pixel_num_y * 6.5 / 3, self.pixel_num_x)

        # -------------------------------------------------------------------
        #   Information of the diode

        # -------------------------------------------------------------------
        #      Keep record of the history or property of the setup
        self.record = []
        self.mono_t1_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.mono_t2_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]

        self.t1_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t2_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t3_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t4_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t5_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]
        self.t6_rocking = [np.zeros(10 ** 4), np.zeros(10 ** 4)]

        # Save miniSD transmission function for a specified incident k vector
        # notice that I only save this information for the 1D case.
        # Saving this information for the 3D case is too expensive for the current situation.
        self.crystal_efficiency = None
        # -------------------------------------------------------------------

        # ------------------------------------------------------------------
        #   Load the external gpu module
        if gpu:
            self.gpuModule = gpuModule

    def align_xpp_mono(self):
        controllerUtil.align_xpp_mono(controller=self)

    def align_miniSD(self):
        controllerUtil.align_miniSD_SASE(controller=self)

    def get_raytracing_trajectory(self, path="mono", get_path_length='True', virtual_sample_plane=None):
        return controllerUtil.get_raytracing_trajectory(controller=self,
                                                        path=path,
                                                        get_path_length=get_path_length,
                                                        virtual_sample_plane=virtual_sample_plane)

    def get_vcc_kout(self):
        return controllerUtil.get_vcc_kout(controller=self)

    def plot_motors(self, ax, color='black', axis="xz"):
        controllerUtil.plot_motors(controller=self, ax=ax, color=color, axis=axis)

    def plot_optics(self, ax, color='black', axis="xz"):
        controllerUtil.plot_optics(controller=self, ax=ax, color=color, axis=axis)

    def plot_mono_rocking(self, ax_mono_t1, ax_mono_t2):
        controllerUtil.plot_mono_rocking(controller=self, ax_mono_t1=ax_mono_t1, ax_mono_t2=ax_mono_t2)

    def plot_mono_optics(self, ax, show_trajectory=False, xlim=None, ylim=None):
        controllerUtil.plot_mono_optics(controller=self, ax=ax, show_trajectory=show_trajectory, xlim=xlim, ylim=ylim)

    def plot_miniSD_table(self, ax, xlim=None, ylim=None, show_trajectory=False):
        controllerUtil.plot_miniSD_table(controller=self, ax=ax, xlim=xlim, ylim=ylim, show_trajectory=show_trajectory)

    def plot_miniSD_rocking(self, ax_list):
        controllerUtil.plot_miniSD_rocking(controller=self, ax_list=ax_list)

    def get_diode(self, spectrum_intensity, k_grid, gpu=False, force=False):
        return controllerUtil.get_diode(controller=self, spectrum_intensity=spectrum_intensity,
                                        k_grid=k_grid, gpu=gpu, force=force)

    def get_reflectivity(self):
        return controllerUtil.get_reflectivity(controller=self)

    def show_cc(self):
        self.cc_shutter = True
        self.vcc_shutter = False

    def show_vcc(self):
        self.vcc_shutter = True
        self.cc_shutter = False

    def show_both(self):
        self.vcc_shutter = True
        self.cc_shutter = True

    def show_neither(self):
        self.vcc_shutter = False
        self.cc_shutter = False


def get_optics():
    # Define gratings
    g1_cc = Crystal.RectangleGrating(a=g1_period / 2.,
                                     b=g1_period / 2.,
                                     direction=np.array([0, 1., 0.], dtype=np.float64),
                                     surface_point=np.zeros(3),
                                     order=1.)

    g1_vcc = Crystal.RectangleGrating(a=g1_period / 2.,
                                      b=g1_period / 2.,
                                      direction=np.array([0., 1., 0.], dtype=np.float64),
                                      surface_point=np.zeros(3),
                                      order=-1.)

    g2_cc = Crystal.RectangleGrating(a=g2_period / 2.,
                                     b=g2_period / 2.,
                                     direction=np.array([0., 1., 0.], dtype=np.float64),
                                     surface_point=np.zeros(3),
                                     order=-1.)
    g2_vcc = Crystal.RectangleGrating(a=g2_period / 2.,
                                      b=g2_period / 2.,
                                      direction=np.array([0., 1., 0.], dtype=np.float64),
                                      surface_point=np.zeros(3),
                                      order=1.)

    # ------------------------------------------
    #   Get crystal for XPP mono
    # ------------------------------------------
    mono_miscut = [np.deg2rad(0.0), np.deg2rad(0.0)]
    mono_diamond = [Crystal.CrystalBlock3D(h=np.array([0., 2. * np.pi / dia111['thickness'], 0.]),
                                           normal=np.array(
                                               [0., -np.cos(mono_miscut[x]), np.sin(mono_miscut[x])]),
                                           surface_point=np.zeros(3, dtype=np.float64),
                                           thickness=10e3,
                                           chi_dict=dia111,
                                           edge_length=20e3) for x in range(2)]
    mono_diamond[1].rotate_wrt_point(rot_mat=np.array([[1, 0, 0],
                                                       [0, -1, 0],
                                                       [0, 0, -1]], dtype=np.float64),
                                     ref_point=np.copy(mono_diamond[1].surface_point))

    # ------------------------------------------------
    #    Get VCC
    # Define Bragg crystals
    vcc_channel_cut_config = ["lower left", 'upper left', 'upper left', 'lower left']
    vcc_channel_cut_angles = np.deg2rad(np.array([[0, -5], [5., 0], [0, 5], [-5, 0]]))
    vcc_channel_cut_edge_length_list = np.array([[50e3, 65.25e3],
                                                 [65.25e3, 50e3, ],
                                                 [50e3, 65.25e3],
                                                 [65.25e3, 50e3, ],
                                                 ])
    vcc_channel_cuts = [Crystal.ChannelCut(crystal_type="Silicon",
                                           miller_index="220",
                                           thickness_list=np.array([1e4, 1e4]),
                                           gap=13.595e3,
                                           surface_center_offset=32.5e3,
                                           edge_length_list=vcc_channel_cut_edge_length_list[_x],
                                           asymmetry_angle_list=vcc_channel_cut_angles[_x],
                                           first_surface_loc=vcc_channel_cut_config[_x],
                                           source=None,
                                           crystal_property=si220)
                        for _x in range(4)]
    # Shift the crystal such that the rotation center is at 0
    vcc_channel_cuts[1].shift(displacement=np.copy(vcc_channel_cuts[1].crystal_list[1].surface_point))
    vcc_channel_cuts[3].shift(displacement=np.copy(vcc_channel_cuts[3].crystal_list[1].surface_point))

    # --------------------------------------------------
    #   Get CC
    cc_channel_cut_config = ["upper left", 'lower left', ]
    cc_channel_cut_angles = np.deg2rad(np.array([[0, 0], [0, 0]]))
    cc_channel_cut_edge_length_list = np.array([[40e3, 100e3],
                                                [120e3, 15e3]])
    cc_channel_cut_center_offset = [30e3, 52.5e3]
    cc_channel_cut_gap = [25.15e3, 25.8e3]

    cc_channel_cuts = [Crystal.ChannelCut(crystal_type="Silicon",
                                          miller_index="220",
                                          thickness_list=np.array([1e4, 1e4]),
                                          gap=cc_channel_cut_gap[_x],
                                          surface_center_offset=cc_channel_cut_center_offset[_x],
                                          edge_length_list=cc_channel_cut_edge_length_list[_x],
                                          asymmetry_angle_list=cc_channel_cut_angles[_x],
                                          first_surface_loc=cc_channel_cut_config[_x],
                                          source=None,
                                          crystal_property=si220)
                       for _x in range(2)]
    cc_channel_cuts[1].shift(displacement=np.copy(vcc_channel_cuts[1].crystal_list[1].surface_point))

    # Create the YAG crystals
    #  Later, I'll install the YAG camera. However, at this moment, I would like to use a
    # simple implementation of the yag crystal as a place-holder to make the simulation work.
    sample = Crystal.YAG()
    yag_sample = Crystal.YAG()
    yag1 = Crystal.YAG()
    yag2 = Crystal.YAG()
    yag3 = Crystal.YAG()

    optics_dict = {"g1 cc": g1_cc,
                   "g1 vcc": g1_vcc,
                   "g2 cc": g2_cc,
                   "g2 vcc": g2_vcc,
                   "cc1": cc_channel_cuts[0],
                   "cc2": cc_channel_cuts[1],
                   "vcc1": vcc_channel_cuts[0],
                   "vcc2": vcc_channel_cuts[1],
                   "vcc3": vcc_channel_cuts[2],
                   "vcc4": vcc_channel_cuts[3],
                   "yag sample": yag_sample,
                   "yag1": yag1,
                   "yag2": yag2,
                   "yag3": yag3,
                   "sample": sample,
                   "xpp mono": mono_diamond,
                   }
    return optics_dict


def assemble_motors_and_optics():
    # Get all the optics
    optics_all = get_optics()

    # Get the XPP mono
    monoT1 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['xpp mono'][0],
                                                   crystal_loc=np.copy(optics_all['xpp mono'][0].surface_point, ))

    monoT2 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['xpp mono'][1],
                                                   crystal_loc=np.copy(optics_all['xpp mono'][1].surface_point, ))
    # Get all the motors
    t1 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['cc1'],
                                               crystal_loc=np.copy(optics_all['cc1'].crystal_list[0].surface_point, ))
    t6 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['cc2'],
                                               crystal_loc=np.copy(optics_all['cc2'].crystal_list[1].surface_point, ))

    # For the VCC branch
    t2 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['vcc3'],
                                               crystal_loc=np.copy(optics_all['vcc3'].crystal_list[0].surface_point, ))
    t3 = MotorStack.CrystalTower_x_y_theta_chi(crystal=optics_all['vcc4'],
                                               crystal_loc=np.copy(optics_all['vcc4'].crystal_list[1].surface_point, ))
    t45 = MotorStack.CrystalTower_miniSD_Scan(channelCut1=optics_all['vcc1'],
                                              crystal_loc1=np.copy(optics_all['vcc1'].crystal_list[0].surface_point, ),
                                              channelCut2=optics_all['vcc2'],
                                              crystal_loc2=np.copy(optics_all['vcc2'].crystal_list[1].surface_point, ),
                                              )

    # Change the positive direction of the motor
    t3.th.set_positive(motion='negative')
    t45.th1.set_positive(motion='negative')
    t6.th.set_positive(motion='negative')

    # Get the grating tower
    g1 = MotorStack.Grating_tower(grating_1=optics_all['g1 cc'],
                                  grating_m1=optics_all['g1 vcc'],
                                  )
    g2 = MotorStack.Grating_tower(grating_1=optics_all['g2 cc'],
                                  grating_m1=optics_all['g2 vcc'], )

    # Get the sample tower
    sample = MotorStack.TG_Sample_tower(sample=optics_all['sample'],
                                        yag_sample=optics_all['yag sample'],
                                        yag1=optics_all['yag1'],
                                        yag2=optics_all['yag2'],
                                        yag3=optics_all['yag3']
                                        )

    motor_stacks = {'t1': t1,
                    't2': t2,
                    't3': t3,
                    't45': t45,
                    't6': t6,
                    'g1': g1,
                    'g2': g2,
                    'sample': sample,
                    'mono t1': monoT1,
                    'mono t2': monoT2}

    return motor_stacks, optics_all


def install_xpp_mono(controller):
    # Insatll the XPP mono
    bragg = util.get_bragg_angle(wave_length=controller.wavelength, plane_distance=dia111['thickness'])
    # Assume that the gap size is 50 cm, then the z offset is gap / np.tan(2 * bragg)
    gap = 500e3
    z_offset = gap / np.tan(2 * bragg)

    # Shift the pulse and mono tower 1
    displacement = np.array([0, -gap, -z_offset], dtype=np.float64)
    for item in controller.mono_t1.all_obj:
        item.shift(displacement=displacement)

    # Shift the installation path of the xpp mono
    displacement = np.array([0, 0, -10e6], dtype=np.float64)
    for item in controller.mono_t1.all_obj:
        item.shift(displacement=displacement)
    for item in controller.mono_t2.all_obj:
        item.shift(displacement=displacement)


def insatll_motionStack_and_optics(controller):
    # Install SD table
    controller.breadboard1.shift(displacement=np.array([-220e3 - 12.7e3, -225e3, 0, ]))
    Motors.install_motors_on_breadboard(motor_stack=controller.t1.all_obj, breadboard=controller.breadboard1,
                                        diag_hole_idx1=(7, 0), diag_hole_idx2=(11, 5))
    Motors.install_motors_on_breadboard(motor_stack=controller.t2.all_obj, breadboard=controller.breadboard1,
                                        diag_hole_idx1=(7, 8), diag_hole_idx2=(11, 14))
    Motors.install_motors_on_breadboard(motor_stack=controller.t3.all_obj, breadboard=controller.breadboard1,
                                        diag_hole_idx1=(7, 16), diag_hole_idx2=(11, 22))
    Motors.install_motors_on_breadboard(motor_stack=controller.t45.all_obj, breadboard=controller.breadboard1,
                                        diag_hole_idx1=(7, 27), diag_hole_idx2=(11, 37))
    Motors.install_motors_on_breadboard(motor_stack=controller.t6.all_obj, breadboard=controller.breadboard1,
                                        diag_hole_idx1=(7, 42), diag_hole_idx2=(11, 47))

    # Install sample table
    controller.breadboard3.shift(displacement=np.array([-254e3, -212.5e3, 7728e3]))
    controller.breadboard4.shift(displacement=np.array([-254e3, -212.5e3, 7728e3 + 457.2e3]))
    Motors.install_motors_on_breadboard(motor_stack=controller.sample.all_obj, breadboard=controller.breadboard4,
                                        diag_hole_idx1=(4, 0), diag_hole_idx2=(7, 5))

    displacement = np.array([50e3, 0.0, 0.0])
    for item in controller.sample.all_obj:
        item.shift(displacement=displacement)

    # Install the gratings
    # Assume that there is no need to align the gratings
    displacement = np.array([0.0, 0.0, -1.9e6]) - controller.g1.grating_1.surface_point
    for item in controller.g1.all_obj:
        item.shift(displacement=displacement)

    displacement = np.array([0.0, 0.0, 3.8e6]) - controller.g2.grating_1.surface_point
    for item in controller.g2.all_obj:
        item.shift(displacement=displacement)
