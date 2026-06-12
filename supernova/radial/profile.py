"""
Simple implementation of a radial profile class that will represent a radial function(s).

The idea is a given function to sustain completely the function {F_nm(r)}_{For all n, m}.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from typing import List, Optional, Any
from .base import RadialFE
from .._logger import get_logger
from ._harmonics_kernels import decompose2d, decompose3d
from pycocos import MagneticCoordinates

logger = get_logger(__name__)

def ProfileFactoryPsiThetaZeta(
    *args,
    ntormax: Optional[int] = None,
    mpolmax: Optional[int] = None
) -> List["Profile"]:
    """
    Factory function to generate many Profiles at once, assuming the
    input field is defined on the (Psi, Theta, Zeta) grid.

    Input arguments (*args) are expected to be different datarrays or datasets,
    defined on the same (Psi, Theta, Zeta) grid, and the factory will return a list of Profile
    instances, one for each input field. 
    The factory will batch the process for all the input profiles: decomposing them all
    into (n, m) harmonics and then calling the Profile constructor that will build the Profile instances.
    """
    if ntormax is not None and (not isinstance(ntormax, int) or ntormax < 0):
        raise ValueError("ntormax must be a non-negative integer.")
    if mpolmax is not None and (not isinstance(mpolmax, int) or mpolmax < 0):
        raise ValueError("mpolmax must be a non-negative integer.")

    # We check that all the input arguments are xarray DataArrays or Datasets and that they have the
    # same coordinates (Psi, Theta, Zeta).
    data = xr.Dataset()
    for i, arg in enumerate(args):
        if not isinstance(arg, (xr.DataArray, xr.Dataset)):
            raise ValueError("All input arguments must be xarray DataArrays or Datasets.")
        if isinstance(arg, xr.DataArray):
            data[f'field_{i}'] = arg
        else:
            for var in arg.data_vars:
                data[f'{var}_{i}'] = arg[var]
    
    # We stack all the data into a single array with (nfield*nPsi, nTheta, nZeta) shape,
    # where nfield is the number of input fields.
    tmp = data.to_array(dim='field')
    stacked = tmp.stack(stacked=['field', 'psi']).transpose('stacked', 'theta', 'nu').values

    theta = np.asarray(data.coords["theta"].values, dtype=np.float64)
    zeta = np.asarray(data.coords["nu"].values, dtype=np.float64)

    sine_sign = -1.0
    if data.theta.size == 1 and data.nu.size == 1:
        # If both Theta and Zeta have only one point, we are in the axisymmetric case and we only have the (n=0, m=0) mode.
        coeffs_cos = stacked.squeeze()[:, None, None]
        coeffs_sin = np.zeros_like(coeffs_cos)
        ntormax = 0
        mpolmax = 0
        mpol = [0]
        ntor = [0]
    elif data.theta.size == 1:
        nyquist_ntor = data.nu.size // 2
        ntormax = nyquist_ntor if ntormax is None else ntormax
        if ntormax > nyquist_ntor:
            logger.warning(
                f"The maximum toroidal mode is greater than the Nyquist frequency. Setting ntormax to {nyquist_ntor}."
            )
            ntormax = nyquist_ntor

        coeffs_cos, coeffs_sin = decompose2d(stacked.squeeze(), ntormax, angle=zeta)
        coeffs_cos = coeffs_cos[:, None, :]
        coeffs_sin = coeffs_sin[:, None, :]
        # decompose2d returns the coefficient of sin(n*zeta), while the paper
        # phase for a toroidal-only mode is exp(-i*n*zeta).
        sine_sign = 1.0
        mpolmax = 0
        mpol = [0]
        ntor = np.arange(ntormax+1)
    elif data.nu.size == 1:
        nyquist_mpol = data.theta.size // 2
        mpolmax = nyquist_mpol if mpolmax is None else mpolmax
        if mpolmax > nyquist_mpol:
            logger.warning(
                f"The maximum poloidal mode is greater than the Nyquist frequency. Setting mpolmax to {nyquist_mpol}."
            )
            mpolmax = nyquist_mpol

        coeffs_cos, coeffs_sin = decompose2d(stacked.squeeze(), mpolmax, angle=theta)
        coeffs_cos = coeffs_cos[:, :, None]
        coeffs_sin = coeffs_sin[:, :, None]
        ntormax = 0
        mpol = np.arange(mpolmax+1)
        ntor = [0]
    else:
        nyquist_ntor = stacked.shape[2] // 2
        nyquist_mpol = stacked.shape[1] // 2

        ntormax = nyquist_ntor if ntormax is None else ntormax
        mpolmax = nyquist_mpol if mpolmax is None else mpolmax

        if ntormax > nyquist_ntor:
            logger.warning(
                f"The maximum toroidal mode is greater than the Nyquist frequency. Setting ntormax to {nyquist_ntor}."
            )
            ntormax = nyquist_ntor
        if mpolmax > nyquist_mpol:
            logger.warning(
                f"The maximum poloidal mode is greater than the Nyquist frequency. Setting mpolmax to {nyquist_mpol}."
            )
            mpolmax = nyquist_mpol

        coeffs_cos, coeffs_sin = decompose3d(
            stacked,
            mpolmax,
            ntormax,
            theta=theta,
            zeta=zeta,
        )
        mpol = np.arange(mpolmax+1)
        ntor = np.arange(ntormax+1)
    
    # From the decomposition in cosine and sine coefficients, we can 
    # build the complex coefficients for the (n, m) harmonics. In the 
    # decomposition, n and m start from 0 to ntormax and mpolmax, 
    # respectively, and the 0th mode is the average (n=0, m=0) 
    # which is purely real.
    # In the paper convention, F(psi, theta, zeta) is expanded as
    # sum F_{m,n}(psi) exp(i*(m*theta - n*zeta)). For real fields the negative
    # pair is implied by F_{-m,-n} = F_{m,n}^*, so storing the nonnegative
    # (m, n) coefficient requires half of the real cosine/sine amplitude.
    nfields = tmp.field.size
    coeffs_complex = np.zeros((nfields*data.coords['psi'].size, mpolmax+1, ntormax+1), dtype=np.complex128)
    coeffs_complex[:, 0, 0] = coeffs_cos[:, 0, 0]  # n=0, m=0 mode is purely real
    for j in range(ntormax+1):
        for i in range(mpolmax+1):
            if i == 0 and j == 0:
                continue  # already set the (0, 0) mode
            coeffs_complex[:, i, j] = 0.5 * (coeffs_cos[:, i, j] + sine_sign * 1j * coeffs_sin[:, i, j])

    # We now build the Profile instances for each field. We need to reshape the 
    # coefficients back to (nfield, nPsi, mpol, ntor) shape and then we can call 
    # the Profile constructor for each field.
    coeffs_complex = coeffs_complex.reshape(nfields, data.coords['psi'].size, mpolmax+1, ntormax+1)

    profiles = []
    for i in range(nfields):
        profiles.append(
            Profile(
                x=data.coords['psi'].values,
                F_nm=coeffs_complex[i],
                ntor=ntor,
                mpol=mpol,
            )
        )
    return profiles

def ProfileFactoryRzPhi(
    magnetic_coordinates: MagneticCoordinates,
    *args,
    ntormax: Optional[int] = None,
    mpolmax: Optional[int] = None
) -> List["Profile"]:
    """
    Factory function to generate many Profiles at once, assuming the
    input field is defined on the cylindrical grid (R, z, phi).

    Input arguments (*args) are expected to be different datarrays or datasets,
    defined on the same (R, z, phi) grid, and the factory will return a list of Profile
    instances, one for each input field. 
    The factory will batch the process for all the input profiles: transforming them all
    to the (Psi, Theta, Zeta) coordinates and then calling the ProfileFactorPsiThetaZeta
    that will decompose them into (n, m) harmonics and build the Profile instances.
    """
    for iargs in args:
        if not isinstance(iargs, (xr.DataArray, xr.Dataset)):
            raise ValueError("All input arguments must be xarray DataArrays or Datasets.")
        
    # We combine all the arguments into a single dataset.
    data = xr.Dataset()
    for i, arg in enumerate(args):
        if isinstance(arg, xr.DataArray):
            data[f'field_{i}'] = arg
        else:
            for var in arg.data_vars:
                data[f'{var}_{i}'] = arg[var]

    # We now generate the Psi, Theta, Zeta coordinates using the magnetic_coordinates.
    data_mag = magnetic_coordinates.cyl2mag_scalar(field=data, return_psi_norm=True)

    return ProfileFactoryPsiThetaZeta(
        data_mag,
        ntormax=ntormax,
        mpolmax=mpolmax,
    )

class Profile:
    """
    This will contain the information of a collection of radial profiles, indexing them
    by their poloidal and toroidal mode numbers, n and m.
    """
    def __init__(self, x: np.ndarray, F_nm: np.ndarray, 
                 ntor: List[int]=None, mpol: List[int]=None,
                 attrs: dict=None):
        x = np.asarray(x, dtype=np.float64)
        F_nm = np.asarray(F_nm, dtype=np.complex128)
        
        if x.ndim != 1:
            raise ValueError("x should be a 1D array")
        if F_nm.ndim == 2:
            logger.debug("F_nm is 2D, assuming shape (x, mpol, ntor=0)")
            F_nm = F_nm[:, :, np.newaxis]  # Add ntor dimension
        elif F_nm.ndim == 1:
            logger.debug("F_nm is 1D, assuming shape (x, mpol=0, ntor=0)")
            F_nm = F_nm[:, np.newaxis, np.newaxis]  # Add mpol and ntor dimensions
        elif F_nm.ndim != 3:
            raise ValueError("F_nm should be a 1D, 2D, or 3D array")

        # Checking the mode numbers.
        if ntor is None:
            ntor = np.arange(F_nm.shape[2])
        if mpol is None:
            mpol = np.arange(F_nm.shape[1])

        ntor = np.asarray(ntor, dtype=np.int32)
        mpol = np.asarray(mpol, dtype=np.int32)

        if len(ntor) != F_nm.shape[2]:
            raise ValueError(f"Length of ntor ({ntor.size}) should match the third dimension of F_nm ({F_nm.shape[2]})")
        if len(mpol) != F_nm.shape[1]:
            raise ValueError(f"Length of mpol ({mpol.size}) should match the second dimension of F_nm ({F_nm.shape[1]})")
        
        # We store the profile as an xarray DataArray for easier indexing and metadata handling.
        self.data = xr.DataArray(F_nm, 
                                    dims=['x', 'mpol', 'ntor'], 
                                    coords={'x': x, 'mpol': mpol, 'ntor': ntor},
                                    attrs=attrs)
        
    @classmethod
    def profile_from_radialFE(cls, a_k: np.ndarray, radial_fe: RadialFE, 
                             ntor: List[int]=None, mpol: List[int]=None,
                             attrs: dict=None):
        """
        Construct a Profile from a given set of coefficients a_k and a radial finite element basis.
        """
        if a_k.ndim != 1:
            raise ValueError("a_k should be a 1D array of coefficients")
        
        x = radial_fe.mesh

        # Let's sanitize the mode numbers.
        if ntor is None:
            ntor = np.arange(a_k.shape[2]) if a_k.ndim == 3 else np.arange(1)
        if mpol is None:
            mpol = np.arange(a_k.shape[1]) if a_k.ndim >= 2 else np.arange(1)

        ntor = np.asarray(ntor, dtype=np.int32)
        mpol = np.asarray(mpol, dtype=np.int32)

        if a_k.ndim == 1:
            a_k = a_k[:, np.newaxis, np.newaxis]  # Add mpol and ntor dimensions
        elif a_k.ndim == 2:
            a_k = a_k[:, :, np.newaxis]  # Add ntor dimension
        elif a_k.ndim != 3:
            raise ValueError("a_k should be a 1D, 2D, or 3D array")

        if a_k.shape[1] != mpol.size:
            raise ValueError(f"Second dimension of a_k ({a_k.shape[1]}) should match the size of mpol ({mpol.size})")
        if a_k.shape[2] != ntor.size:
            raise ValueError(f"Third dimension of a_k ({a_k.shape[2]}) should match the size of ntor ({ntor.size})")

        F_nm = np.zeros((x.size, mpol.size, ntor.size), dtype=np.complex128)
        for ii, n in enumerate(ntor):
            for jj, m in enumerate(mpol):
                F_nm[:, jj, ii] = radial_fe.evaluate(a_k[:, jj, ii], x)
        
        return cls(x, F_nm, ntor=ntor, mpol=mpol, attrs=attrs)
    
    def projectonto(self, fe: RadialFE):
        """
        Projects the internal state into a given radial finite element basis, 
        return the a_k coefficients for each (n, m) mode.
        """
        a_k = np.zeros((fe.dofs, self.data.mpol.size, self.data.ntor.size), dtype=np.complex128)
        for ii, n in enumerate(self.data.ntor.values):
            for jj, m in enumerate(self.data.mpol.values):
                a_k[:, jj, ii] = fe.project(self.data[:, jj, ii].values, x=self.data.x.values)
        return a_k
    
    @property
    def m_pol(self):
        """
        Returns the poloidal mode numbers of the profile.
        """
        return self.data.mpol.values.astype(np.int32)

    @property
    def n_tor(self):
        """
        Returns the toroidal mode numbers of the profile.
        """
        return self.data.ntor.values.astype(np.int32)

    @property
    def x(self):
        """
        Returns the radial grid points of the profile.
        """
        return self.data.x.values.astype(np.float64)
    
    def copy(self, deepcopy: bool=True) -> "Profile":
        """
        Returns a copy of the Profile instance. 
        
        Parameters
        ----------
        deepcopy : bool, optional
            If True, the data will be copied deeply, meaning that 
            changes to the data in the copied instance will not 
            affect the original instance.
        """
        if deepcopy:
            new_data = self.data.copy(deep=True)
        else:
            new_data = self.data.copy(deep=False)
        
        return Profile(new_data.x.values.copy(), new_data.values, 
                       ntor=new_data.ntor.values.copy(), 
                       mpol=new_data.mpol.values.copy(),
                       attrs=new_data.attrs)
                       

    # -------------------
    # Overloading operators for simplicity.
    # -------------------
    
    def __add__(self, other: Any) -> "Profile":
        """
        Sum two profiles together, assuming they are defined on the same x grid and have the same mode numbers.

        When summing two profiles with the same (n, m) mode numbers, we simply sum each coefficient for each
        mode. 

        @TODO: We need to incorporate the non-matching case where the profiles may have different mode numbers,
        (n1, m1) and (n2, m2). The sum should be done on the intersecting range, keeping the rest of the non-overlaping
        as is, and returning the modes of Union(n1, n2) and Union(m1, m2).
        """
        if isinstance(other, Profile):
            if not np.array_equal(self.data.x.values, other.data.x.values):
                raise ValueError("Profiles must be defined on the same x grid to be summed.")
            
            if not np.array_equal(self.data.mpol.values, other.data.mpol.values):
                raise NotImplementedError("Profiles must have the same mpol mode numbers to be summed.")
            
            if not np.array_equal(self.data.ntor.values, other.data.ntor.values):
                raise NotImplementedError("Profiles must have the same ntor mode numbers to be summed.")
            
            new_F_nm = self.data.values + other.data.values
        
        elif isinstance(other, (int, float, complex)):
            # If other is a scalar, we add it to the (n=0, m=0) mode of the profile, which is the average.
            new_F_nm = self.data.values.copy()
            new_F_nm[:, 0, 0] += other
        else:
            raise ValueError("Unsupported type for addition with Profile. Supported types are Profile, int, and float.")

        return Profile(self.data.x.values, new_F_nm, 
                       ntor=self.data.ntor.values, 
                       mpol=self.data.mpol.values,
                       attrs=self.data.attrs)
    
    def __radd__(self, other: Any) -> "Profile":
        """
        Right-hand sum to support sum() function.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def __sub__(self, other: Any) -> "Profile":
        """
        Subtract two profiles together, assuming they are defined on the same x grid and have the same mode numbers.
        """
        if isinstance(other, Profile):
            if not np.array_equal(self.data.x.values, other.data.x.values):
                raise ValueError("Profiles must be defined on the same x grid to be subtracted.")
            
            if not np.array_equal(self.data.mpol.values, other.data.mpol.values):
                raise NotImplementedError("Profiles must have the same mpol mode numbers to be subtracted.")
            
            if not np.array_equal(self.data.ntor.values, other.data.ntor.values):
                raise NotImplementedError("Profiles must have the same ntor mode numbers to be subtracted.")
            
            new_F_nm = self.data.values - other.data.values
        
        elif isinstance(other, (int, float, complex)):
            # If other is a scalar, we subtract it from the (n=0, m=0) mode of the profile, which is the average.
            new_F_nm = self.data.values.copy()
            new_F_nm[:, 0, 0] -= other
        else:
            raise ValueError("Unsupported type for subtraction with Profile. Supported types are Profile, int, and float.")

        return Profile(self.data.x.values, new_F_nm, 
                       ntor=self.data.ntor.values, 
                       mpol=self.data.mpol.values,
                       attrs=self.data.attrs)

    def __rsub__(self, other: Any) -> "Profile":
        """
        Right-hand subtraction to support subtraction with scalars.
        """
        if isinstance(other, (int, float, complex)):
            # If other is a scalar, we subtract the profile from it, which means we subtract the (n=0, m=0) mode from the scalar and negate the rest.
            new_F_nm = -self.data.values.copy()
            new_F_nm[:, 0, 0] += other
            return Profile(self.data.x.values, new_F_nm, 
                           ntor=self.data.ntor.values, 
                           mpol=self.data.mpol.values,
                           attrs=self.data.attrs)
        else:
            raise ValueError("Unsupported type for right-hand subtraction with Profile. Supported types are int and float.")
    
    def __mul__(self, other: Any) -> "Profile":
        """
        Multiplication of two profiles together, assuming they are defined on the same x-grid.

        The multiplication of (n1, m1) x (n2, m2) should be done via a convolution product, that
        can lead to elements outside of the initial range of modes. So far, we will assume that only
        one of the two has a single mode (n=0, m=0) so the multiplication is just scaling all the elements
        of the other profile by the (n=0, m=0) mode.

        @TODO: implement the convolution product for the general case, but leaving the 1D profile multiplication
        as a special fast case.
        """
        if isinstance(other, Profile):
            if not np.array_equal(self.data.x.values, other.data.x.values):
                raise ValueError("Profiles must be defined on the same x grid to be multiplied.")
            
            if self.data.mpol.size > 1 and other.data.mpol.size > 1:
                raise NotImplementedError("At least one of the profiles must have a single mpol mode " +
                                           "(m=0) for multiplication to be implemented.")
            
            if self.data.ntor.size > 1 and other.data.ntor.size > 1:
                raise NotImplementedError("At least one of the profiles must have a single ntor mode "+
                                           "(n=0) for multiplication to be implemented.")
            
            # We need to check that, in fact, the input profile with a single mode has that mode as (n=0, m=0), 
            # so it can be used as a scaling factor for the other profile.
            if self.data.mpol.size == 1 and self.data.ntor.size == 1:
                if self.data.mpol.values[0] != 0 or self.data.ntor.values[0] != 0:
                    raise ValueError("The profile with a single mode must have that mode as (n=0, m=0) for multiplication.")
            elif other.data.mpol.size == 1 and other.data.ntor.size == 1:
                if other.data.mpol.values[0] != 0 or other.data.ntor.values[0] != 0:
                    raise ValueError("The profile with a single mode must have that mode as (n=0, m=0) for multiplication.")
            else:
                # This should not even happen, but who knows.
                raise ValueError("At least one of the profiles must have a single mode (n=0, m=0) for multiplication.")
                
            new_F_nm = self.data.values.copy() * other.data.values # This should not fail as [nr, nm, nn] x [nr, 1, 1].

        elif isinstance(other, (int, float, complex)):
            # If other is a scalar, we multiply it to all modes of the profile.
            new_F_nm = self.data.values.copy() * other
        else:
            raise ValueError("Unsupported type for multiplication with Profile. Supported types are Profile, int, and float.")

        return Profile(self.data.x.values, new_F_nm, 
                       ntor=self.data.ntor.values, 
                       mpol=self.data.mpol.values,
                       attrs=self.data.attrs)
    
    def __rmul__(self, other: Any) -> "Profile":
        """
        We just call the original multiplication, as it effectively 
        commutative under our assumptions.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other: Any) -> "Profile":
        """
        Division of two profiles together, assuming they are defined on the same x-grid.

        The division of (n1, m1) / (n2, m2) should be done via a convolution product, that
        can lead to elements outside of the initial range of modes. So far, we will assume that only
        one of the two has a single mode (n=0, m=0) so the division is just scaling all the elements
        of the other profile by the inverse of the (n=0, m=0) mode.

        Notes for self: __truediv__ is called when this profile on the left of the division operator.

        self / other -> self.__truediv__(other)
        other / self -> self.__rtruediv__(other)

        This implementation makes sense only when the right operand is the 1D profile, in this case "other".
        """
        if isinstance(other, Profile):
            if not np.array_equal(self.data.x.values, other.data.x.values):
                raise ValueError("Profiles must be defined on the same x grid to be divided.")
            
            if other.data.mpol.size > 1:
                raise ValueError("Denominator needs to be a 1D profile (m > 1)")
            
            if other.data.ntor.size > 1:
                raise ValueError("Denominator needs to be a 1D profile (n > 1)")
                
            new_F_nm = self.data.values.copy() / other.data.values # This should not fail as [nr, nm, nn] x [nr, 1, 1].

        elif isinstance(other, (int, float, complex)):
            # If other is a scalar, we divide all modes of the profile by it.
            new_F_nm = self.data.values.copy() / other
        else:
            raise ValueError("Unsupported type for division with Profile. Supported types are Profile, int, and float.")

        return Profile(self.data.x.values, new_F_nm, 
                       ntor=self.data.ntor.values, 
                       mpol=self.data.mpol.values,
                       attrs=self.data.attrs)
    
    def __rtruediv__(self, other: Any) -> "Profile":
        """
        Division of two profiles together, assuming they are defined on the same x-grid.

        The division of (n1, m1) / (n2, m2) should be done via a convolution product, that
        can lead to elements outside of the initial range of modes. So far, we will assume that only
        one of the two has a single mode (n=0, m=0) so the division is just scaling all the elements
        of the other profile by the inverse of the (n=0, m=0) mode.

        Notes for self: __truediv__ is called when this profile on the left of the division operator.

        self / other -> self.__truediv__(other)
        other / self -> self.__rtruediv__(other)

        This implementation makes sense only when the right operand is the 1D profile, in this case "self".
        """
        # The check here is universal, cause 1/profile has to be done either way, which is what
        # we need to avoid altogether.
        if self.data.mpol.size > 1:
            raise ValueError("Denominator needs to be a 1D profile (m > 1)")
        
        if self.data.ntor.size > 1:
            raise ValueError("Denominator needs to be a 1D profile (n > 1)")

        if isinstance(other, Profile):
            if not np.array_equal(self.data.x.values, other.data.x.values):
                raise ValueError("Profiles must be defined on the same x grid to be divided.")
                
            new_F_nm = other.data.values/ self.data.values.copy()  # This should not fail as [nr, nm, nn] x [nr, 1, 1].

        elif isinstance(other, (int, float, complex)):
            # If other is a scalar, we divide all modes of the profile by it.
            new_F_nm = other / self.data.values.copy()
        else:
            raise ValueError("Unsupported type for division with Profile. Supported types are Profile, int, and float.")

        return Profile(self.data.x.values, new_F_nm, 
                       ntor=self.data.ntor.values, 
                       mpol=self.data.mpol.values,
                       attrs=self.data.attrs)