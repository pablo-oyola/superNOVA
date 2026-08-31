"""Assembly of the discretised ideal-MHD operator.

The pipeline is: equilibrium profiles become point-local blocks
(:mod:`~supernova.operators.blocks`), the blocks become the split local tensors
``M0``/``M1``/``Mminus1`` (:mod:`~supernova.operators.local`), those are
projected onto the compatible mixed radial spaces
(:mod:`~supernova.operators.mixed`, :mod:`~supernova.operators.project`), and
the result is reduced by the boundary conditions
(:mod:`~supernova.operators.boundary`).
"""
