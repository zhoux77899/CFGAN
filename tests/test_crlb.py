"""
This module contains comprehensive unit tests for the `CRLB` class in the `cfgan.math.crlb` module.

Tests cover:
- Initialization with numpy and torch inputs
- Padding operations (pad method)
- Maximum value operations (maximum method)
- Square root operations (sqrt method)
- Gradient calculations (grad method)
- Forward pass calculations (forward method with lateral/axial/radial modes)
- Numerical consistency between numpy and torch implementations
"""

import unittest

import numpy as np
import torch

from cfgan.math.crlb import CRLB


class TestCRLBInitialization(unittest.TestCase):
    """Test CRLB class initialization."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        self.eps = 1e-6

    def _create_psf_numpy(self):
        """Create a test PSF image as numpy array."""
        np.random.seed(42)
        psf = np.random.rand(*self.psf_shape).astype(np.float64)
        return psf / psf.sum()

    def _create_psf_torch(self):
        """Create a test PSF image as torch tensor."""
        torch.manual_seed(42)
        psf = torch.rand(*self.psf_shape, dtype=torch.float64)
        return psf / psf.sum()

    def test_init_numpy(self):
        """Test initialization with numpy.ndarray."""
        psf = self._create_psf_numpy()
        crlb = CRLB(psf, self.delta, self.eps)

        self.assertIsInstance(crlb.psf_imgs, np.ndarray)
        self.assertEqual(crlb.psf_imgs.shape, self.psf_shape)
        self.assertAlmostEqual(crlb.psf_imgs.sum().item(), self.psf_shape[0], places=5)
        self.assertEqual(crlb.dx, 0.1)
        self.assertEqual(crlb.dz, 0.5)
        self.assertEqual(crlb.eps, 1e-6)

    def test_init_torch(self):
        """Test initialization with torch.Tensor."""
        psf = self._create_psf_torch()
        crlb = CRLB(psf, self.delta, self.eps)

        self.assertIsInstance(crlb.psf_imgs, torch.Tensor)
        self.assertEqual(crlb.psf_imgs.shape, self.psf_shape)
        self.assertAlmostEqual(crlb.psf_imgs.sum().item(), self.psf_shape[0], places=5)
        self.assertEqual(crlb.dx, 0.1)
        self.assertEqual(crlb.dz, 0.5)
        self.assertEqual(crlb.eps, 1e-6)

    def test_init_invalid_dim_2d(self):
        """Test that 2D array raises AssertionError."""
        psf_2d = np.random.rand(256, 256).astype(np.float64)
        with self.assertRaises(AssertionError):
            CRLB(psf_2d, self.delta)

    def test_init_invalid_dim_4d(self):
        """Test that 4D array raises AssertionError."""
        psf_4d = np.random.rand(10, 200, 256, 256).astype(np.float64)
        with self.assertRaises(AssertionError):
            CRLB(psf_4d, self.delta)

    def test_init_invalid_delta_length(self):
        """Test that invalid delta length raises AssertionError."""
        psf = self._create_psf_numpy()
        with self.assertRaises(AssertionError):
            CRLB(psf, (0.1, 0.5, 0.1))  # type: ignore

    def test_init_default_eps(self):
        """Test initialization with default eps value."""
        psf = self._create_psf_numpy()
        crlb = CRLB(psf, self.delta)
        self.assertEqual(crlb.eps, 1e-6)


class TestCRLBPad(unittest.TestCase):
    """Test CRLB padding functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_pad_numpy(self):
        """Test padding with numpy array."""
        grad = (self.psf_numpy[:, :, 2:] - self.psf_numpy[:, :, :-2]) / (2 * 0.1)
        padding = (1, 1, 0, 0, 0, 0)
        padded = self.crlb_numpy.pad(grad, padding)

        self.assertIsInstance(padded, np.ndarray)
        expected_shape = (200, 256, 256)
        self.assertEqual(padded.shape, expected_shape)

    def test_pad_torch(self):
        """Test padding with torch tensor."""
        grad = (self.psf_torch[:, :, 2:] - self.psf_torch[:, :, :-2]) / (2 * 0.1)
        padding = (1, 1, 0, 0, 0, 0)
        padded = self.crlb_torch.pad(grad, padding)

        self.assertIsInstance(padded, torch.Tensor)
        expected_shape = (200, 256, 256)
        self.assertEqual(padded.shape, expected_shape)

    def test_pad_consistency(self):
        """Test that numpy and torch padding results are numerically consistent."""
        grad_np = (self.psf_numpy[:, :, 2:] - self.psf_numpy[:, :, :-2]) / (2 * 0.1)
        grad_torch = (self.psf_torch[:, :, 2:] - self.psf_torch[:, :, :-2]) / (2 * 0.1)
        padding = (1, 1, 0, 0, 0, 0)

        padded_np = self.crlb_numpy.pad(grad_np, padding)
        padded_torch = self.crlb_torch.pad(grad_torch, padding)

        np.testing.assert_allclose(
            padded_np,
            padded_torch.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="pad: numpy/torch padding results inconsistent",
        )

    def test_pad_invalid_length(self):
        """Test that invalid padding length raises AssertionError."""
        grad = self.psf_numpy[:, :, 2:] - self.psf_numpy[:, :, :-2]
        with self.assertRaises(AssertionError):
            self.crlb_numpy.pad(grad, (1, 1, 0, 0))

    def test_pad_unsupported_type(self):
        """Test that unsupported type raises NotImplementedError."""
        grad_list = [1, 2, 3]
        padding = (1, 1, 0, 0, 0, 0)
        with self.assertRaises(NotImplementedError):
            self.crlb_numpy.pad(grad_list, padding)  # type: ignore

    def test_pad_all_directions(self):
        """Test padding in all directions."""
        padding_x = (1, 1, 0, 0, 0, 0)
        padding_y = (0, 0, 1, 1, 0, 0)
        padding_z = (0, 0, 0, 0, 1, 1)

        grad_x = (self.psf_numpy[:, :, 2:] - self.psf_numpy[:, :, :-2]) / (2 * 0.1)
        grad_y = (self.psf_numpy[:, 2:, :] - self.psf_numpy[:, :-2, :]) / (2 * 0.1)
        grad_z = (self.psf_numpy[2:, :, :] - self.psf_numpy[:-2, :, :]) / (2 * 0.5)

        padded_x = self.crlb_numpy.pad(grad_x, padding_x)
        padded_y = self.crlb_numpy.pad(grad_y, padding_y)
        padded_z = self.crlb_numpy.pad(grad_z, padding_z)

        self.assertEqual(padded_x.shape, (200, 256, 256))
        self.assertEqual(padded_y.shape, (200, 256, 256))
        self.assertEqual(padded_z.shape, (200, 256, 256))


class TestCRLBMaximum(unittest.TestCase):
    """Test CRLB maximum value functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_maximum_numpy(self):
        """Test maximum operation with numpy array."""
        x = np.array([0.5e-6, 1e-6, 2e-6, 0.5e-6])
        result = self.crlb_numpy.maximum(x)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result[0], 1e-6)
        self.assertEqual(result[1], 1e-6)
        self.assertEqual(result[2], 2e-6)
        self.assertEqual(result[3], 1e-6)

    def test_maximum_torch(self):
        """Test maximum operation with torch tensor."""
        x = torch.tensor([0.5e-6, 1e-6, 2e-6, 0.5e-6], dtype=torch.float64)
        result = self.crlb_torch.maximum(x)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result[0].item(), 1e-6)
        self.assertEqual(result[1].item(), 1e-6)
        self.assertEqual(result[2].item(), 2e-6)
        self.assertEqual(result[3].item(), 1e-6)

    def test_maximum_consistency(self):
        """Test that numpy and torch maximum results are numerically consistent."""
        x_np = np.array([0.5e-6, 1e-6, 2e-6, 5e-6, 0.5e-6], dtype=np.float64)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        result_np = self.crlb_numpy.maximum(x_np)
        result_torch = self.crlb_torch.maximum(x_torch)

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="maximum: numpy/torch results inconsistent"
        )

    def test_maximum_unsupported_type(self):
        """Test that unsupported type raises NotImplementedError."""
        x_list = [0.5e-6, 1e-6, 2e-6]
        with self.assertRaises(NotImplementedError):
            self.crlb_numpy.maximum(x_list)  # type: ignore

    def test_maximum_all_above_eps(self):
        """Test maximum when all values are above eps."""
        x = np.array([1e-5, 2e-5, 3e-5], dtype=np.float64)
        result = self.crlb_numpy.maximum(x)

        np.testing.assert_array_equal(result, x)

    def test_maximum_all_below_eps(self):
        """Test maximum when all values are below eps."""
        x = np.array([1e-7, 5e-8, 2e-9], dtype=np.float64)
        result = self.crlb_numpy.maximum(x)

        self.assertTrue(np.all(result >= 1e-6))


class TestCRLBSqrt(unittest.TestCase):
    """Test CRLB square root functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_sqrt_numpy(self):
        """Test sqrt operation with numpy array."""
        x = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        result = self.crlb_numpy.sqrt(x)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0, 5.0])

    def test_sqrt_torch(self):
        """Test sqrt operation with torch tensor."""
        x = torch.tensor([4.0, 9.0, 16.0, 25.0], dtype=torch.float64)
        result = self.crlb_torch.sqrt(x)

        self.assertIsInstance(result, torch.Tensor)
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_sqrt_consistency(self):
        """Test that numpy and torch sqrt results are numerically consistent."""
        x_np = np.array([0.01, 0.04, 0.09, 0.16, 0.25], dtype=np.float64)
        x_torch = torch.tensor(x_np, dtype=torch.float64)

        result_np = self.crlb_numpy.sqrt(x_np)
        result_torch = self.crlb_torch.sqrt(x_torch)

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="sqrt: numpy/torch results inconsistent"
        )

    def test_sqrt_unsupported_type(self):
        """Test that unsupported type raises NotImplementedError."""
        x_list = [4.0, 9.0, 16.0]
        with self.assertRaises(NotImplementedError):
            self.crlb_numpy.sqrt(x_list)  # type: ignore

    def test_sqrt_zero(self):
        """Test sqrt of zero."""
        x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        result = self.crlb_numpy.sqrt(x)

        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])


class TestCRLBGrad(unittest.TestCase):
    """Test CRLB gradient calculation functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_grad_x_numpy(self):
        """Test gradient calculation in x direction with numpy."""
        grad = self.crlb_numpy.grad("x")

        self.assertIsInstance(grad, np.ndarray)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_x_torch(self):
        """Test gradient calculation in x direction with torch."""
        grad = self.crlb_torch.grad("x")

        self.assertIsInstance(grad, torch.Tensor)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_y_numpy(self):
        """Test gradient calculation in y direction with numpy."""
        grad = self.crlb_numpy.grad("y")

        self.assertIsInstance(grad, np.ndarray)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_y_torch(self):
        """Test gradient calculation in y direction with torch."""
        grad = self.crlb_torch.grad("y")

        self.assertIsInstance(grad, torch.Tensor)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_z_numpy(self):
        """Test gradient calculation in z direction with numpy."""
        grad = self.crlb_numpy.grad("z")

        self.assertIsInstance(grad, np.ndarray)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_z_torch(self):
        """Test gradient calculation in z direction with torch."""
        grad = self.crlb_torch.grad("z")

        self.assertIsInstance(grad, torch.Tensor)
        self.assertEqual(grad.shape, self.psf_shape)

    def test_grad_consistency_x(self):
        """Test that numpy and torch gradient x results are numerically consistent."""
        grad_np = self.crlb_numpy.grad("x")
        grad_torch = self.crlb_torch.grad("x")

        np.testing.assert_allclose(
            grad_np, grad_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="grad(x): numpy/torch results inconsistent"
        )

    def test_grad_consistency_y(self):
        """Test that numpy and torch gradient y results are numerically consistent."""
        grad_np = self.crlb_numpy.grad("y")
        grad_torch = self.crlb_torch.grad("y")

        np.testing.assert_allclose(
            grad_np, grad_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="grad(y): numpy/torch results inconsistent"
        )

    def test_grad_consistency_z(self):
        """Test that numpy and torch gradient z results are numerically consistent."""
        grad_np = self.crlb_numpy.grad("z")
        grad_torch = self.crlb_torch.grad("z")

        np.testing.assert_allclose(
            grad_np, grad_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="grad(z): numpy/torch results inconsistent"
        )

    def test_grad_invalid_direction(self):
        """Test that invalid direction raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.crlb_numpy.grad("invalid")


class TestCRLBForward(unittest.TestCase):
    """Test CRLB forward pass functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        self.total_photons = 1000
        self.background_photons = 10
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_forward_lateral_numpy(self):
        """Test forward pass with lateral mode using numpy."""
        result = self.crlb_numpy.forward(self.total_photons, self.background_photons, "lateral")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(np.all(result >= 0))

    def test_forward_lateral_torch(self):
        """Test forward pass with lateral mode using torch."""
        result = self.crlb_torch.forward(self.total_photons, self.background_photons, "lateral")

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(torch.all(result >= 0))

    def test_forward_axial_numpy(self):
        """Test forward pass with axial mode using numpy."""
        result = self.crlb_numpy.forward(self.total_photons, self.background_photons, "axial")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(np.all(result >= 0))

    def test_forward_axial_torch(self):
        """Test forward pass with axial mode using torch."""
        result = self.crlb_torch.forward(self.total_photons, self.background_photons, "axial")

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(torch.all(result >= 0))

    def test_forward_radial_numpy(self):
        """Test forward pass with radial mode using numpy."""
        result = self.crlb_numpy.forward(self.total_photons, self.background_photons, "radial")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(np.all(result >= 0))

    def test_forward_radial_torch(self):
        """Test forward pass with radial mode using torch."""
        result = self.crlb_torch.forward(self.total_photons, self.background_photons, "radial")

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(torch.all(result >= 0))

    def test_forward_mode_consistency_lateral(self):
        """Test that lateral mode returns numerically consistent results."""
        result_np = self.crlb_numpy.forward(self.total_photons, self.background_photons, "lateral")
        result_torch = self.crlb_torch.forward(self.total_photons, self.background_photons, "lateral")

        np.testing.assert_allclose(
            result_np,
            result_torch.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="forward(lateral): numpy/torch results inconsistent",
        )

    def test_forward_mode_consistency_axial(self):
        """Test that axial mode returns numerically consistent results."""
        result_np = self.crlb_numpy.forward(self.total_photons, self.background_photons, "axial")
        result_torch = self.crlb_torch.forward(self.total_photons, self.background_photons, "axial")

        np.testing.assert_allclose(
            result_np,
            result_torch.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="forward(axial): numpy/torch results inconsistent",
        )

    def test_forward_mode_consistency_radial(self):
        """Test that radial mode returns numerically consistent results."""
        result_np = self.crlb_numpy.forward(self.total_photons, self.background_photons, "radial")
        result_torch = self.crlb_torch.forward(self.total_photons, self.background_photons, "radial")

        np.testing.assert_allclose(
            result_np,
            result_torch.numpy(),
            rtol=1e-5,
            atol=1e-8,
            err_msg="forward(radial): numpy/torch results inconsistent",
        )

    def test_forward_invalid_mode(self):
        """Test that invalid mode raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.crlb_numpy.forward(self.total_photons, self.background_photons, "invalid")

    def test_forward_lateral_radial_relationship(self):
        """Test that radial result is greater than or equal to lateral result."""
        lateral = self.crlb_numpy.forward(self.total_photons, self.background_photons, "lateral")
        radial = self.crlb_numpy.forward(self.total_photons, self.background_photons, "radial")

        self.assertTrue(np.all(radial >= lateral))

    def test_forward_special_values(self):
        """Test handling of special values (NaN, Inf)."""
        psf_with_nan = self.psf_numpy.copy()
        psf_with_nan[0, 0, 0] = 0.0
        psf_with_nan = psf_with_nan / psf_with_nan.sum()

        crlb = CRLB(psf_with_nan, self.delta)
        result = crlb.forward(self.total_photons, self.background_photons, "lateral")

        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestCRLBCall(unittest.TestCase):
    """Test CRLB __call__ method functionality."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        self.total_photons = 1000
        self.background_photons = 10
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_call_numpy(self):
        """Test __call__ method with numpy input."""
        result_call = self.crlb_numpy(self.total_photons, self.background_photons, "lateral")
        result_forward = self.crlb_numpy.forward(self.total_photons, self.background_photons, "lateral")

        np.testing.assert_array_equal(result_call, result_forward)

    def test_call_torch(self):
        """Test __call__ method with torch input."""
        result_call = self.crlb_torch(self.total_photons, self.background_photons, "lateral")
        result_forward = self.crlb_torch.forward(self.total_photons, self.background_photons, "lateral")

        torch.testing.assert_close(result_call, result_forward)

    def test_call_all_modes(self):
        """Test __call__ method with all valid modes."""
        modes = ["lateral", "axial", "radial"]

        for mode in modes:
            result_call = self.crlb_numpy(self.total_photons, self.background_photons, mode)
            result_forward = self.crlb_numpy.forward(self.total_photons, self.background_photons, mode)
            np.testing.assert_array_equal(
                result_call, result_forward, err_msg=f"__call__ inconsistency for mode '{mode}'"
            )


class TestCRLBNumericalEdgeCases(unittest.TestCase):
    """Test numerical edge cases and boundary conditions."""

    def setUp(self):
        self.psf_shape = (200, 256, 256)
        self.delta = (0.1, 0.5)
        np.random.seed(42)
        self.psf_data = np.random.rand(*self.psf_shape).astype(np.float64)
        self.psf_data = self.psf_data / self.psf_data.sum()
        self.psf_numpy = self.psf_data.copy()
        self.psf_torch = torch.from_numpy(self.psf_data.copy())
        self.crlb_numpy = CRLB(self.psf_numpy, self.delta)
        self.crlb_torch = CRLB(self.psf_torch, self.delta)

    def test_low_photon_count(self):
        """Test with very low photon count."""
        total_photons = 1
        background_photons = 0

        result_np = self.crlb_numpy.forward(total_photons, background_photons, "lateral")
        result_torch = self.crlb_torch.forward(total_photons, background_photons, "lateral")

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="low photon count: numpy/torch inconsistent"
        )

    def test_high_background(self):
        """Test with high background photons."""
        total_photons = 1000
        background_photons = 10000

        result_np = self.crlb_numpy.forward(total_photons, background_photons, "axial")
        result_torch = self.crlb_torch.forward(total_photons, background_photons, "axial")

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="high background: numpy/torch inconsistent"
        )

    def test_small_delta(self):
        """Test with very small delta values."""
        crlb_small = CRLB(self.psf_numpy, (0.001, 0.001))
        crlb_small_torch = CRLB(self.psf_torch, (0.001, 0.001))

        result_np = crlb_small.forward(1000, 10, "radial")
        result_torch = crlb_small_torch.forward(1000, 10, "radial")

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="small delta: numpy/torch inconsistent"
        )

    def test_large_delta(self):
        """Test with large delta values."""
        crlb_large = CRLB(self.psf_numpy, (10.0, 50.0))
        crlb_large_torch = CRLB(self.psf_torch, (10.0, 50.0))

        result_np = crlb_large.forward(1000, 10, "radial")
        result_torch = crlb_large_torch.forward(1000, 10, "radial")

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="large delta: numpy/torch inconsistent"
        )

    def test_unity_psf(self):
        """Test with uniform PSF (all values equal)."""
        psf_uniform = np.ones(self.psf_shape, dtype=np.float64)
        psf_uniform = psf_uniform / psf_uniform.sum()
        psf_uniform_torch = torch.ones(self.psf_shape, dtype=torch.float64)
        psf_uniform_torch = psf_uniform_torch / psf_uniform_torch.sum()

        crlb_uniform = CRLB(psf_uniform, self.delta)
        crlb_uniform_torch = CRLB(psf_uniform_torch, self.delta)

        result_np = crlb_uniform.forward(1000, 10, "radial")
        result_torch = crlb_uniform_torch.forward(1000, 10, "radial")

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), rtol=1e-5, atol=1e-8, err_msg="uniform PSF: numpy/torch inconsistent"
        )

    def test_deterministic_results(self):
        """Test that same input produces same output."""
        result1 = self.crlb_numpy.forward(1000, 10, "lateral")
        result2 = self.crlb_numpy.forward(1000, 10, "lateral")

        np.testing.assert_array_equal(result1, result2)
