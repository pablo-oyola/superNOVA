"""Desktop front ends.

Importing this package needs PySide6, which the solver does not; nothing else
in ``supernova`` imports it. Launch with ``python -m supernova gui``.
"""

from supernova.gui.continuum import ContinuumWindow, main

__all__ = ["ContinuumWindow", "main"]
