"""SLEPc eigensolvers for the assembled pencil.

Two backends, chosen by the adiabatic index: the quadratic polynomial problem
``(Tminus1 + s*T0 + s**2*T1) x = 0``, and -- for exact ``gamma_s == 0``, where
the ``1/s`` term vanishes -- the generalized linear problem
``(-T0) x = s*T1 x``.
"""
