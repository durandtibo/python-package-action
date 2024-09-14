from __future__ import annotations

from coola.testing import numpy_available


@numpy_available
def test_numpy() -> None:
    import numpy as np  # local import because it is an optional dependency

    assert np.array_equal(np.ones((2, 3)) + np.ones((2, 3)), np.full((2, 3), 2.0))
