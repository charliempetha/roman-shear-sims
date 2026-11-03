import numpy as np

import galsim
from galsim import roman

from .constant import IMCOM_PSF_FWHM

PSF_TYPES = ["gauss", "airy", "obs_airy", "moffat", "roman", "imcom"]

_rad_to_arcsec = 180 * 3600 / np.pi


class PSFMaker:
    """
    A class to create different types of PSFs.

    Parameters
    ----------
    psf_type : str
        The type of PSF to create.
        Options are 'gauss', 'airy', 'obs_airy', 'moffat', 'roman', 'imcom'.
    fwhm : float, optional
        The full width at half maximum of the Gaussian PSF in arcseconds.
    chromatic : bool, optional
        Whether the PSF is chromatic. Default is True.
    wave : float, optional
        The wavelength in nanometers for the PSF.
        Required if `psf_type` is 'roman'.
    sca : int, optional
        The SCA number for the Roman PSF. Default is 1.
    pupil_bin : int, optional
        The binning factor for the pupil. Default is 8.
    n_waves : int, optional
        The number of wavelengths to use for the Roman PSF. Default is 10.
    """

    def __init__(
        self,
        psf_type="gauss",
        fwhm=None,
        chromatic=True,
        wave=None,
        sca=1,
        pupil_bin=8,
        n_waves=10,
    ):
        if not isinstance(psf_type, str):
            raise TypeError("psf_type must be a string")
        psf_type = psf_type.lower()
        if psf_type not in PSF_TYPES:
            raise ValueError(
                f"psf_type must be one of {PSF_TYPES}, got {psf_type}"
            )
        self.psf_type = psf_type

        self.fwhm = fwhm
        self.sca = sca
        self.pupil_bin = pupil_bin
        self.chromatic = chromatic
        self.wave = wave
        self.n_waves = n_waves

    def init_psf(self, band="Y106"):
        """
        Initialize the PSF parameters based on the bandpass.

        Parameters
        ----------
        band : str
            The bandpass name for which to initialize the PSF parameters.
        """
        if not isinstance(band, str):
            raise TypeError("band must be a string")

        self.band = band
        if self.psf_type != "imcom":
            if self.wave is None:
                bp = roman.getBandpasses()[self.band]
                self.wave = bp.effective_wavelength
            if self.fwhm is None:
                self.fwhm = self.wave / roman.pixel_scale * 1e-9
                self.fwhm *= _rad_to_arcsec
        else:
            if self.fwhm is None:
                if self.band not in IMCOM_PSF_FWHM:
                    raise ValueError(
                        f"Band {self.band} not a valid IMCOM band."
                    )
                self.fwhm = IMCOM_PSF_FWHM[self.band]

    def get_psf(self, sca=None, image_pos=None, wcs=None):
        """
        Get the PSF object

        Parameters
        ----------
        sca : int, optional
            The SCA number for the Roman PSF. Default is the initialized value.
        image_pos : galsim.CelestialCoord, optional
            The celestial coordinates of the image center. Required if
            `psf_type` is 'roman'.
        wcs : galsim.WCS, optional
            The WCS object for the image. Required if `psf_type` is 'roman'.
        Returns
        -------
        galsim.GSObject
            The PSF object.
        """
        if self.psf_type == "gauss" or self.psf_type == "imcom":
            psf = galsim.Gaussian(fwhm=self.fwhm)
        elif self.psf_type == "airy":
            psf = galsim.Airy(lam=self.wave, diam=roman.diameter)
        elif self.psf_type == "obs_airy":
            psf = galsim.Airy(
                lam=self.wave,
                diam=roman.diameter,
                obscuration=roman.obscuration,
            )
        elif self.psf_type == 'moffat':
            psf = galsim.Moffat(beta=3.5, fwhm=self.fwhm)
        elif self.psf_type == "roman":
            wave = None if self.chromatic else self.wave
            psf = roman.getPSF(
                SCA=sca,
                bandpass=self.band,
                SCA_pos=image_pos,
                pupil_bin=self.pupil_bin,
                wcs=wcs,
                n_waves=self.n_waves,
                wavelength=wave,
            )
        return psf
