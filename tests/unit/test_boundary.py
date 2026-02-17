"""
Unit tests for boundary module.

Tests for PeriodicBoundaryCondition, OpenBoundaryCondition,
and MixedBoundaryCondition classes.
"""
import numpy as np
import pytest

from pymd.boundary import (
    BoundaryCondition,
    MixedBoundaryCondition,
    OpenBoundaryCondition,
    PeriodicBoundaryCondition,
)


class TestPeriodicBoundaryCondition:
    """Tests for PeriodicBoundaryCondition."""

    @pytest.fixture
    def pbc(self) -> PeriodicBoundaryCondition:
        """Create a periodic boundary condition instance."""
        return PeriodicBoundaryCondition()

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Create a standard cubic box."""
        return np.array([10.0, 10.0, 10.0])

    def test_minimum_image_no_wrap(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test minimum image for vectors within half-box."""
        vector = np.array([[2.0, 3.0, 4.0]])
        result = pbc.apply_minimum_image(vector, box)
        np.testing.assert_array_almost_equal(result, vector)

    def test_minimum_image_wrap_positive(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test minimum image wraps large positive vectors."""
        # 8.0 is more than half of 10, should wrap to -2.0
        vector = np.array([[8.0, 0.0, 0.0]])
        result = pbc.apply_minimum_image(vector, box)
        expected = np.array([[-2.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_minimum_image_wrap_negative(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test minimum image wraps large negative vectors."""
        # -7.0 is less than -half of 10, should wrap to 3.0
        vector = np.array([[-7.0, 0.0, 0.0]])
        result = pbc.apply_minimum_image(vector, box)
        expected = np.array([[3.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_minimum_image_exactly_half(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test edge case at exactly half box size."""
        # At exactly 5.0, round(0.5) = 0 for even, 1 for odd (banker's rounding)
        # Result depends on numpy's round behavior
        vector = np.array([[5.0, 0.0, 0.0]])
        result = pbc.apply_minimum_image(vector, box)
        # Should be in range [-5, 5]
        assert np.abs(result[0, 0]) <= 5.0

    def test_minimum_image_batch(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test minimum image on batch of vectors."""
        vectors = np.array([
            [2.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [-7.0, 0.0, 0.0],
            [0.0, 8.0, -6.0],
        ])
        result = pbc.apply_minimum_image(vectors, box)

        # All results should be in range [-5, 5]
        assert np.all(np.abs(result) <= 5.0)

    def test_wrap_positions_in_box(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test wrapping positions already in box."""
        positions = np.array([[2.0, 3.0, 4.0]])
        result = pbc.wrap_positions(positions, box)
        np.testing.assert_array_almost_equal(result, positions)

    def test_wrap_positions_positive(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test wrapping positions outside box (positive)."""
        positions = np.array([[15.0, 22.0, 37.0]])
        result = pbc.wrap_positions(positions, box)
        expected = np.array([[5.0, 2.0, 7.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_wrap_positions_negative(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test wrapping negative positions."""
        positions = np.array([[-3.0, -15.0, -7.0]])
        result = pbc.wrap_positions(positions, box)
        expected = np.array([[7.0, 5.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_wrap_positions_result_in_range(
        self, pbc: PeriodicBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test that wrapped positions are always in [0, box)."""
        # Random positions far outside box
        positions = np.random.randn(100, 3) * 100
        result = pbc.wrap_positions(positions, box)

        assert np.all(result >= 0)
        assert np.all(result < box)

    def test_get_name(self, pbc: PeriodicBoundaryCondition) -> None:
        """Test boundary condition name."""
        assert pbc.get_name() == "Periodic"


class TestOpenBoundaryCondition:
    """Tests for OpenBoundaryCondition."""

    @pytest.fixture
    def obc(self) -> OpenBoundaryCondition:
        """Create an open boundary condition instance."""
        return OpenBoundaryCondition()

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Create a standard box (ignored by OBC)."""
        return np.array([10.0, 10.0, 10.0])

    def test_minimum_image_no_change(
        self, obc: OpenBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test that minimum image makes no changes."""
        vector = np.array([[100.0, -50.0, 200.0]])
        result = obc.apply_minimum_image(vector, box)
        np.testing.assert_array_almost_equal(result, vector)

    def test_wrap_positions_no_change(
        self, obc: OpenBoundaryCondition, box: np.ndarray
    ) -> None:
        """Test that wrapping makes no changes."""
        positions = np.array([[100.0, -50.0, 200.0]])
        result = obc.wrap_positions(positions, box)
        np.testing.assert_array_almost_equal(result, positions)

    def test_get_name(self, obc: OpenBoundaryCondition) -> None:
        """Test boundary condition name."""
        assert obc.get_name() == "Open"


class TestMixedBoundaryCondition:
    """Tests for MixedBoundaryCondition."""

    @pytest.fixture
    def box(self) -> np.ndarray:
        """Create a standard box."""
        return np.array([10.0, 10.0, 10.0])

    def test_xy_periodic(self, box: np.ndarray) -> None:
        """Test XY periodic, Z open (slab geometry)."""
        bc = MixedBoundaryCondition((True, True, False))

        # X should wrap, Z should not
        vector = np.array([[8.0, 0.0, 30.0]])
        result = bc.apply_minimum_image(vector, box)

        assert pytest.approx(result[0, 0]) == -2.0  # X wrapped
        assert pytest.approx(result[0, 2]) == 30.0  # Z unchanged

    def test_z_periodic(self, box: np.ndarray) -> None:
        """Test Z periodic, XY open (nanowire geometry)."""
        bc = MixedBoundaryCondition((False, False, True))

        vector = np.array([[30.0, 20.0, 8.0]])
        result = bc.apply_minimum_image(vector, box)

        assert pytest.approx(result[0, 0]) == 30.0  # X unchanged
        assert pytest.approx(result[0, 1]) == 20.0  # Y unchanged
        assert pytest.approx(result[0, 2]) == -2.0  # Z wrapped

    def test_wrap_xy_only(self, box: np.ndarray) -> None:
        """Test wrapping only in XY dimensions."""
        bc = MixedBoundaryCondition((True, True, False))

        positions = np.array([[15.0, -5.0, 25.0]])
        result = bc.wrap_positions(positions, box)

        assert pytest.approx(result[0, 0]) == 5.0   # X wrapped
        assert pytest.approx(result[0, 1]) == 5.0   # Y wrapped
        assert pytest.approx(result[0, 2]) == 25.0  # Z unchanged

    def test_no_periodic(self, box: np.ndarray) -> None:
        """Test all open (equivalent to OpenBC)."""
        bc = MixedBoundaryCondition((False, False, False))

        vector = np.array([[30.0, 30.0, 30.0]])
        result = bc.apply_minimum_image(vector, box)
        np.testing.assert_array_almost_equal(result, vector)

    def test_all_periodic(self, box: np.ndarray) -> None:
        """Test all periodic (equivalent to PeriodicBC)."""
        bc = MixedBoundaryCondition((True, True, True))

        vector = np.array([[8.0, -7.0, 6.0]])
        result = bc.apply_minimum_image(vector, box)

        expected = np.array([[-2.0, 3.0, -4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_name_xy_periodic(self) -> None:
        """Test name for XY periodic."""
        bc = MixedBoundaryCondition((True, True, False))
        assert "XY" in bc.get_name()

    def test_get_name_z_periodic(self) -> None:
        """Test name for Z periodic."""
        bc = MixedBoundaryCondition((False, False, True))
        assert "Z" in bc.get_name()

    def test_get_name_none_periodic(self) -> None:
        """Test name for no periodic."""
        bc = MixedBoundaryCondition((False, False, False))
        assert "none" in bc.get_name()

    def test_invalid_dims(self) -> None:
        """Test that invalid periodic_dims raises error."""
        with pytest.raises(ValueError):
            MixedBoundaryCondition((True, True))  # Only 2 dimensions


class TestBoundaryConditionInterface:
    """Test that all boundary conditions implement the interface."""

    @pytest.mark.parametrize("bc_class,args", [
        (PeriodicBoundaryCondition, ()),
        (OpenBoundaryCondition, ()),
        (MixedBoundaryCondition, ((True, True, False),)),
    ])
    def test_is_base_class_instance(
        self, bc_class: type, args: tuple
    ) -> None:
        """Test all BCs inherit from BoundaryCondition."""
        bc = bc_class(*args)
        assert isinstance(bc, BoundaryCondition)

    @pytest.mark.parametrize("bc_class,args", [
        (PeriodicBoundaryCondition, ()),
        (OpenBoundaryCondition, ()),
        (MixedBoundaryCondition, ((True, True, False),)),
    ])
    def test_has_required_methods(
        self, bc_class: type, args: tuple
    ) -> None:
        """Test all BCs have required methods."""
        bc = bc_class(*args)
        assert hasattr(bc, "apply_minimum_image")
        assert hasattr(bc, "wrap_positions")
        assert hasattr(bc, "get_name")
        assert callable(bc.apply_minimum_image)
        assert callable(bc.wrap_positions)
        assert callable(bc.get_name)
