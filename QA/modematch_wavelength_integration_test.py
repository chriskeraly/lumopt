
""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.figures_of_merit.modematch import ModeMatch

class ModeMatchWavelengthIntegrationTest(TestCase):
    """ Unit test for class ModeMatch. Checks that the integrals of the figure of merit and its gradient with respect to wavelength are working correctly. """

    def test_single_wavelength_fom_integral(self):
        """ Test FOM integral for single wavelength case: result should be input FOM for backward compatibility. """
        exact_fom = 0.4896
        fom = ModeMatch.fom_wavelength_integral(T_fwd_vs_wavelength = np.array([exact_fom]),
                                                wavelengths = np.array([1300e-9]),
                                                target_T_fwd = lambda wl: 0.5 * np.ones(wl.size),
                                                norm_p = 1) # unused
        self.assertAlmostEqual(fom, exact_fom, 15)

    def test_fom_integral_norm_p1(self):
        """ Test FOM integral with norm p = 1. """
        wl_points = 5
        fom = ModeMatch.fom_wavelength_integral(T_fwd_vs_wavelength = np.ones(wl_points), 
                                                wavelengths = np.linspace(1300e-9, 1800e-9, wl_points), 
                                                target_T_fwd = lambda wl: np.power(np.sin(np.pi * (wl - wl.min()) / (wl.max() - wl.min())), 2),
                                                norm_p = 1)
        exact_fom = 0.0
        self.assertAlmostEqual(fom, exact_fom, 15)
    
    def test_fom_integral_norm_p2(self):
        """ Test FOM integral with norm p = 2. """
        wl_points = 5000
        fom = ModeMatch.fom_wavelength_integral(T_fwd_vs_wavelength = 0.5 * np.ones(wl_points), 
                                                wavelengths = np.linspace(1.0e-9, 1.0e-8, wl_points), 
                                                target_T_fwd = lambda wl: np.exp(-1.0 * (wl - wl.min()) / (wl.max() - wl.min())), 
                                                norm_p = 2)
        exact_fom = 0.5 * np.exp(-1) * (np.sqrt(2.0 * (np.exp(2.0) - 1.0)) - np.sqrt(4.0 * np.exp(1.0) - np.exp(2.0) - 2.0))
        self.assertAlmostEqual(fom, exact_fom, 7)

    def test_single_wavelength_gradient_integral(self):
        """ Test FOM gradient integral for single wavelength case. """
        fom_grad = ModeMatch.fom_gradient_wavelength_integral_impl(T_fwd_vs_wavelength = np.array([0.2851]),
                                                                   T_fwd_partial_derivs_vs_wl =  np.array([2.0]),
                                                                   target_T_fwd_vs_wavelength = np.array([1.0]),
                                                                   wl = np.array([1800e-9]),
                                                                   norm_p = 1)
        exact_fom_grad = -1.0 * np.sign(0.2851 - 1.0) * 2.0
        self.assertAlmostEqual(fom_grad[0], exact_fom_grad, 15)

    def test_fom_gradient_integral_p1(self):
        """ Test FOM gradient integral with norm p = 1. """
        wl_points = 3
        wavelengths = np.linspace(1300e-9, 1800e-9, wl_points)
        target_T_fwd = lambda wl: np.linspace(0.0, 1.0, wl.size)
        fom_grad = ModeMatch.fom_gradient_wavelength_integral_impl(T_fwd_vs_wavelength = 0.25 * np.ones(wl_points),
                                                                   T_fwd_partial_derivs_vs_wl = np.ones((wl_points,1)),
                                                                   target_T_fwd_vs_wavelength = target_T_fwd(wavelengths),
                                                                   wl = wavelengths,
                                                                   norm_p = 1)
        self.assertAlmostEqual(fom_grad[0], 0.5, 15)

    def test_fom_gradient_integral_p2(self):
        """ Test FOM gradient integral with norm p = 2. """
        wl_points = 5000
        wavelengths = np.linspace(1.0e-9, 1.0e-8, wl_points)
        target_T_fwd = lambda wl: np.exp(-1.0 * (wl - wl.min()) / (wl.max() - wl.min()))
        fom_grad = ModeMatch.fom_gradient_wavelength_integral_impl(T_fwd_vs_wavelength = 0.5 * np.ones(wl_points),
                                                                   T_fwd_partial_derivs_vs_wl = np.ones((wl_points,1)),
                                                                   target_T_fwd_vs_wavelength = target_T_fwd(wavelengths),
                                                                   wl = wavelengths,
                                                                   norm_p = 2)
        exact_fom_grad = (np.exp(1.0) - 2.0) / np.sqrt(4.0 * np.exp(1.0) - np.exp(2.0) - 2.0)
        self.assertAlmostEqual(fom_grad[0], exact_fom_grad, 7)

if __name__ == "__main__":
    run([__file__])