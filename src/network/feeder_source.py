from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    from typing import Protocol, runtime_checkable
except ImportError:                          # Python < 3.8 fallback
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from src.network.synthetic_feeder import FeederData, generate_synthetic_feeder
from src.network.layout_feeder import generate_layout_feeder
from src.network.gis_feeder import GISFeederSource


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FeederSource(Protocol):
    """
    Structural interface for feeder generators.

    Any class with a `build() -> FeederData` method satisfies this protocol.
    No inheritance required.
    """

    def build(self) -> FeederData:
        """Return a fully-populated FeederData."""
        ...


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------

class SyntheticFeederSource:
    """
    Wraps generate_synthetic_feeder() (MST-based random radial feeder).

    All keyword arguments are forwarded verbatim to generate_synthetic_feeder.
    See src/network/synthetic_feeder.py for full parameter documentation.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def build(self) -> FeederData:
        return generate_synthetic_feeder(**self._kwargs)


class LayoutFeederSource:
    """
    Wraps generate_layout_feeder() (comb / herringbone topology).

    All keyword arguments are forwarded verbatim to generate_layout_feeder.
    See src/network/layout_feeder.py for full parameter documentation.

    The `trunk_waypoints` keyword argument is the GIS integration hook:
    pass a list of (x_km, y_km) tuples extracted from a real street centreline
    to override the default straight-spine placement.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def build(self) -> FeederData:
        return generate_layout_feeder(**self._kwargs)


# Re-export GISFeederSource so callers can import from this module
__all__ = [
    "FeederSource",
    "SyntheticFeederSource",
    "LayoutFeederSource",
    "GISFeederSource",
    "make_feeder",
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_feeder(source_type: str = "synthetic", **kwargs: Any) -> FeederData:
    """
    Convenience factory: build a FeederData from a named source type.

    Parameters
    ----------
    source_type : str
        One of ``"synthetic"``, ``"layout"``, or ``"gis"``.
    **kwargs
        Forwarded to the corresponding source class constructor.

    Returns
    -------
    FeederData

    Raises
    ------
    ValueError
        If source_type is not one of the recognised values.
    RuntimeError
        If source_type == "gis" and the MIP solver returns infeasible.

    Examples
    --------
    >>> feeder = make_feeder("synthetic", n_buses=33, seed=0)
    >>> feeder = make_feeder("layout", n_trunk=8, lateral_depth=3)
    >>> feeder = make_feeder("gis",
    ...     filepath="data/examples/simple_lv_feeder.geojson",
    ...     root_coord=(-0.10000, 51.50000))
    """
    if source_type == "synthetic":
        return SyntheticFeederSource(**kwargs).build()
    elif source_type == "layout":
        return LayoutFeederSource(**kwargs).build()
    elif source_type == "gis":
        return GISFeederSource(**kwargs).build()   # raises NotImplementedError
    else:
        raise ValueError(
            f"Unknown source_type {source_type!r}. "
            "Choose from: 'synthetic', 'layout', 'gis'."
        )
