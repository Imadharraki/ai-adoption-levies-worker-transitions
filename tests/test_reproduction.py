"""Regression tests for the adoption and welfare calculations."""

import unittest

import numpy as np
from scipy.integrate import quad

from reproduce import (
    DELTA, ECONOMIES, HORIZON, KAPPA, LEVY_0, LEVY_1, LOSS,
    PREPARATION, SLOW_HAZARD, FAST_HAZARD, adoption_share,
    common_hazard_effect, displacement_kernel, grid_bounds,
    identified_welfare, optimal_effort, private_change,
    steady_welfare,
)


class WelfareTests(unittest.TestCase):
    def test_stationary_values(self):
        expected = {'A': 0.0028712404109672, 'B': -0.0025045601768369}
        for key, value in expected.items():
            economy = ECONOMIES[key]
            a0 = optimal_effort(economy.productivity, LEVY_0)
            a1 = optimal_effort(economy.productivity, LEVY_1)
            self.assertAlmostEqual(float(economy.masses @ adoption_share(a0)), 0.5, places=8)
            self.assertAlmostEqual(float(economy.masses @ adoption_share(a1)), 0.4, places=8)
            self.assertAlmostEqual(steady_welfare(economy, LEVY_1) - steady_welfare(economy, LEVY_0),
                                   value, places=9)

    def test_private_and_transport(self):
        economy = ECONOMIES['B']
        hazard = np.array([SLOW_HAZARD, FAST_HAZARD])
        marginal = np.array([economy.masses[1], 1.0 - economy.masses[1]])
        lo, hi, low_pi, high_pi = identified_welfare(economy, HORIZON, hazard, marginal)
        self.assertAlmostEqual(private_change(economy, HORIZON), -0.005461733573965264, places=9)
        self.assertAlmostEqual(lo, -0.0012530449938666, places=9)
        self.assertAlmostEqual(hi, 0.0051471105526967, places=9)
        np.testing.assert_allclose(low_pi.sum(axis=0), marginal, atol=1e-9)
        np.testing.assert_allclose(high_pi.sum(axis=1), economy.masses, atol=1e-9)

    def test_kernel_against_quadrature(self):
        economy = ECONOMIES['B']
        a0 = optimal_effort(economy.productivity, LEVY_0)
        a1 = optimal_effort(economy.productivity, LEVY_1)
        difference = adoption_share(a0) - adoption_share(a1)
        hazard = np.array([0.45, 1.0, 1.8, 1.2 + a1[1]])
        target = displacement_kernel(economy, HORIZON, hazard)
        for i in range(len(a1)):
            for j, mu in enumerate(hazard):
                lam = DELTA + a1[i]
                integral = quad(
                    lambda t: difference[i] * (DELTA + a1[i] * np.exp(-lam * t))
                              * (-np.expm1(-mu * (HORIZON - t))) / mu,
                    0.0, HORIZON, epsabs=1e-12,
                )[0] * LOSS / HORIZON
                self.assertAlmostEqual(integral, target[i, j], places=10)

    def test_grid_brackets_exact_welfare(self):
        for economy in ECONOMIES.values():
            lower, upper = grid_bounds(economy, np.linspace(0, 0.18, 9))
            exact = steady_welfare(economy, LEVY_1) - steady_welfare(economy, LEVY_0)
            self.assertLessEqual(lower, exact + 1e-10)
            self.assertGreaterEqual(upper, exact - 1e-10)

    def test_critical_horizon(self):
        economy = ECONOMIES['B']
        self.assertGreater(common_hazard_effect(economy, 5.0), 0.0)
        self.assertLess(common_hazard_effect(economy, 10.0), 0.0)
        self.assertAlmostEqual(common_hazard_effect(economy, 7.252271), 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
