import sys

sys.path.append("../../../../XRaySimulation")

import numpy as np

from XRaySimulation import Crystal, util, Pulse
from XRaySimulation.Machine import Motors

# The following modules are loaded as a temporary solution
import MotorStack
import controllerUtil

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

g1_period = 1  # um
g2_period = 1  # um
tg_g_period = 1  # um


class XppController_TG:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __init__(self, photon_kev=9.8, gpu=False, gpuModule=False):
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
        self.tg_g = motors['tg g']
        self.m1 = motors['m1']
        self.m2a = motors['m2a']
        self.m2b = motors['m2b']
        self.si = motors['si']
        self.sample = motors['sample']

        self.all_towers = [self.mono_t1, self.mono_t2,
                           self.t1, self.t2, self.t3, self.t45, self.t6,
                           self.g1, self.tg_g,
                           self.m1, self.m2a, self.m2b, self.si, self.sample, ]
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
        controllerUtil.align_miniSD(controller=self)

    def get_raytracing_trajectory(self, path="mono", get_path_length='True', virtual_sample_plane=None):
        return controllerUtil.get_raytracing_trajectory(controller=self,
                                                        path=path,
                                                        get_path_length=get_path_length,
                                                        virtual_sample_plane=virtual_sample_plane)

    def plot_motors(self, ax, color='black', axis="xz"):
        controllerUtil.plot_motors(controller=self, ax=ax, color=color, axis=axis)

    def plot_optics(self, ax, color='black', axis="xz"):
        controllerUtil.plot_optics(controller=self, ax=ax, color=color, axis=axis)

    def plot_mono_rocking(self, ax_mono_t1, ax_mono_t2):
        controllerUtil.plot_mono_rocking(controller=self, ax_mono_t1=ax_mono_t1, ax_mono_t2=ax_mono_t2)

    def plot_mono_optics(self, ax, show_trajectory=False):
        controllerUtil.plot_mono_optics(controller=self, ax=ax, show_trajectory=show_trajectory)

    def plot_miniSD_table(self, ax, xlim=None, ylim=None, show_trajectory=False, show_device=True):
        controllerUtil.plot_miniSD_table(controller=self, ax=ax, xlim=xlim, ylim=ylim,
                                         show_trajectory=show_trajectory,
                                         show_device=show_device)

    def plot_miniSD_rocking(self, ax_list):
        controllerUtil.plot_miniSD_rocking(controller=self, ax_list=ax_list)

    def plot_beam_on_yag(self, ax):
        controllerUtil.plot_beam_on_yag(controller=self, ax=ax)

    def plot_beam_on_sample_yag(self, ax, aspect=None):
        controllerUtil.plot_beam_on_sample_yag(controller=self, ax=ax, aspect=aspect)

    def plot_m1_traj(self, ax, axis='yz', xlim=None, ylim=None):
        controllerUtil.plot_m1_traj(controller=self, ax=ax, axis=axis, xlim=xlim, ylim=ylim)

    def plot_si_traj(self, ax, axis='yz', xlim=None, ylim=None):
        controllerUtil.plot_si_traj(controller=self, ax=ax, axis=axis, xlim=xlim, ylim=ylim)

    def plot_tg_traj(self, ax, axis='yz', xlim=None, ylim=None):
        controllerUtil.plot_tg_traj(controller=self, ax=ax, axis=axis, xlim=xlim, ylim=ylim)

    def get_beam_position_on_yag(self):
        return controllerUtil.get_beam_position_on_yag(controller=self)

    def get_beam_position_on_sample_yag(self):
        return controllerUtil.get_beam_position_on_sample_yag(controller=self)

    def get_sample_path_length(self):
        return controllerUtil.get_sample_path_length(controller=self)

    def get_arrival_time(self):
        return controllerUtil.get_arrival_time(controller=self)

    def get_diode(self, spectrum_intensity, k_grid, gpu=False, force=False):
        return controllerUtil.get_diode(controller=self, spectrum_intensity=spectrum_intensity,
                                        k_grid=k_grid, gpu=gpu, force=force)

    def get_zyla_1(self, sigma_mat, i_probe, i_pump_a, i_pump_b, i_pump_ref, beam_list=None):
        return controllerUtil.get_zyla_1(controller=self, sigma_mat=sigma_mat, i_probe=i_probe, i_pump_a=i_pump_a,
                                         i_pump_b=i_pump_b, i_pump_ref=i_pump_ref, beam_list=beam_list, )

    def get_zyla_2(self, sigma_mat, i_probe, i_pump_a, i_pump_b, i_pump_ref, beam_list=None):
        return controllerUtil.get_zyla_2(controller=self, sigma_mat=sigma_mat, i_probe=i_probe, i_pump_a=i_pump_a,
                                         i_pump_b=i_pump_b, i_pump_ref=i_pump_ref, beam_list=beam_list, )

    def get_sample_kout(self):
        return controllerUtil.get_sample_kout(controller=self)

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

    tg_g_a = Crystal.RectangleGrating(a=tg_g_period / 2.,
                                      b=tg_g_period / 2.,
                                      surface_point=np.zeros(3),
                                      order=1.)

    tg_g_b = Crystal.RectangleGrating(a=tg_g_period / 2.,
                                      b=tg_g_period / 2.,
                                      surface_point=np.zeros(3),
                                      order=-1.)

    # Define total reflection mirrors
    tg_mirror_pump_a = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0, 0]))
    tg_mirror_pump_b = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0.0, 0]))
    tg_mirror_probe = Crystal.TotalReflectionMirror(surface_point=np.zeros(3), normal=np.array([-1.0, 0, 0]))

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

    # Get the silicon 111 for the TG probe
    tg_si111 = Crystal.CrystalBlock3D(h=np.array([np.pi * 2 / si111['thickness'], 0, 0], dtype=np.float64),
                                      normal=np.array([-1., 0, 0.]),
                                      surface_point=np.zeros(3),
                                      thickness=1e4,
                                      chi_dict=si111,
                                      edge_length=2e4, )
    tg_si111.boundary = np.array([[0, -10e3, -10e3, ],
                                  [0, -10e3, 10e3, ],
                                  [0, 10e3, 10e3, ],
                                  [0, 10e3, -10e3, ],
                                  [0, -10e3, -10e3, ], ])

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
                   "tg g a": tg_g_a,
                   "tg g b": tg_g_b,
                   "tg mirror pump a": tg_mirror_pump_a,
                   "tg mirror pump b": tg_mirror_pump_b,
                   "tg mirror probe": tg_mirror_probe,
                   "tg si111": tg_si111,
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

    tg_g = MotorStack.Grating_tower(grating_1=optics_all['tg g a'],
                                    grating_m1=optics_all['tg g b'], )

    # Get the Mirror tower
    m1 = MotorStack.Tower_x_y_pi(mirror=optics_all['tg mirror probe'], )
    m2a = MotorStack.Mirror_tower1(mirror=optics_all['tg mirror pump a'])
    m2b = MotorStack.Mirror_tower2(mirror=optics_all['tg mirror pump b'])

    m2b.yaw.set_positive(motion='negative')

    # Get the silicon tower
    si = MotorStack.Silicon_tower(crystal=optics_all['tg si111'], )

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
                    'tg g': tg_g,
                    'm1': m1,
                    'm2a': m2a,
                    'm2b': m2b,
                    "si": si,
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

    # Install mirror1
    displacement = np.array([0e3, 0.0, 4e6]) - controller.m1.optics.surface_point
    for item in controller.m1.all_obj:
        item.shift(displacement=displacement)

    # Install sample table
    controller.breadboard3.shift(displacement=np.array([-254e3, -212.5e3, 7728e3]))
    Motors.install_motors_on_breadboard(motor_stack=controller.m2a.all_obj, breadboard=controller.breadboard3,
                                        diag_hole_idx1=(0, 0), diag_hole_idx2=(3, 12))
    Motors.install_motors_on_breadboard(motor_stack=controller.m2b.all_obj, breadboard=controller.breadboard3,
                                        diag_hole_idx1=(14, 0), diag_hole_idx2=(17, 12))

    controller.breadboard4.shift(displacement=np.array([-254e3, -212.5e3, 7728e3 + 457.2e3]))
    Motors.install_motors_on_breadboard(motor_stack=controller.si.all_obj, breadboard=controller.breadboard4,
                                        diag_hole_idx1=(5, 2), diag_hole_idx2=(10, 6))
    Motors.install_motors_on_breadboard(motor_stack=controller.sample.all_obj, breadboard=controller.breadboard4,
                                        diag_hole_idx1=(4, 0), diag_hole_idx2=(7, 5))

    displacement = np.array([50e3, 0.0, 0.0])
    for item in controller.sample.all_obj:
        item.shift(displacement=displacement)

    # print("test", controller.si.optics.surface_point)
    displacement = np.array([412.7e3 + 60e3, 25e3, 12.5e3])
    for item in controller.si.all_obj:
        item.shift(displacement=displacement)

    # Install the gratings
    # Assume that there is no need to align the gratings
    displacement = np.array([0.0, 0.0, -1.9e6]) - controller.g1.grating_1.surface_point
    for item in controller.g1.all_obj:
        item.shift(displacement=displacement)

    displacement = np.array([0.0, 0.0, 3.8e6]) - controller.g2.grating_1.surface_point
    for item in controller.g2.all_obj:
        item.shift(displacement=displacement)

    displacement = np.array([0.0, 0.0, 1.5e6]) - controller.tg_g.grating_1.surface_point
    for item in controller.tg_g.all_obj:
        item.shift(displacement=displacement)
