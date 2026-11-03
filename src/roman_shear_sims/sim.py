"""sim.py"""

from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np

import galsim
from galsim import roman

from .noise import make_roman_noise, make_simple_noise
from .wcs import (
    get_SCA_WCS,
    get_IMCOM_WCS,
    make_oversample_local_wcs,
)
from .constant import WORLD_ORIGIN


def make_sim(
    rng,
    galaxy_catalog,
    psf_maker,
    simu_type="sca",
    n_epochs=6,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    bands=None,
    g1=0.0,
    g2=0.0,
    chromatic=False,
    simple_noise=False,
    noise_sigma=1.0,
    n_noise_realizations=1,
    image_factor=1.0,
    draw_method="phot",
    n_photons=None,
    avg_gal_sed_path=None,
    make_deconv_psf=True,
    make_true_psf=True,
    verbose=True,
):
    """Make the simulation

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator to use for the simulation.
    galaxy_catalog : GalaxyCatalog
        The galaxy catalog to use for the simulation.
    psf_maker : PSFMaker
        The PSF maker to use for the simulation.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation,
        'imcom' for IMCOM simulation, 'sh19' for SH19 simulation. Default: 'sca'.
    n_epochs : int, optional
        The number of epochs to simulate. Default: 6.
    exp_time : float, optional
        The exposure time in seconds. Default: 107.
    cell_size_pix : int, optional
        The size of the cell in pixels. Default: 500.
    oversamp_factor : int, optional
        The oversampling factor for the WCS. Default: 3.
    bands : list of str, optional
        The list of bands to simulate.
        If None, defaults to ['Y106', 'J129', 'H158'].
    g1 : float or array-like, optional
        The shear component g1. Default: 0.0.
    g2 : float or array-like, optional
        The shear component g2. Default: 0.0.
    chromatic : bool, optional
        Whether the PSF is chromatic. Default: False.
    simple_noise : bool, optional
        Whether to use simple Gaussian noise instead of the full Roman noise
        model. Default: False.
    noise_sigma : float, optional
        The standard deviation of the Gaussian noise if `simple_noise` is True.
        Required if `simple_noise` is True. Default: 1.0.
    n_noise_realizations : int, optional
        The number of independent noise realizations to create. Default: 1.
    image_factor : float, optional
        A factor to scale the noise level. Default: 1.0.
    draw_method : str, optional
        The method to use for drawing the objects. Default: 'phot'.
    n_photons : int, optional
        The number of photons to use for the 'phot' draw method.
        If None, it will be calculated based on the galaxy flux
        and image_factor. Default: None.
    avg_gal_sed_path : str, optional
        The path to the average galaxy SED file.
        Required if `chromatic` is True.
    make_deconv_psf : bool, optional
        Whether to create the deconvolved PSF image. If sim_type is 'imcom',
        this parameter is used to determine whether to create the PSF image.
        Default: True.
    make_true_psf : bool, optional
        Whether to create the true PSF image. Default: True.
    verbose : bool, optional
        Whether to print progress messages. Default: True.

    Returns
    -------
    dict
        A dictionary containing the simulation results for each band.
        Each key is a band name and the value is a list of dictionaries for
        each epoch.
        Each epoch dictionary contains the following keys:

        * 'sca': The SCA number.
        * 'wcs': The WCS object for the epoch.
        * 'flux_scaling': The flux scaling factor for the band.
        * 'psf_avg': The average PSF image for deconvolution.
        * 'psf_true_galsim': The true PSF object for the epoch.
        * 'sci': A dictionary with keys 'shear_<g1>_<g2>' for each shear
          component, containing the science image array.
        * 'noise': List of noise realizations as image arrays.
        * 'noise_var': The variance of the noise images.
        * 'weight': The weight image array.

    """
    # Set center
    # cell_center_ra, cell_center_dec = \
    #   random_ra_dec_in_healpix(rng, 32, 10307)

    # cell_center_world = galsim.CelestialCoord(
    #     cell_center_ra * galsim.degrees, cell_center_dec * galsim.degrees
    # )
    cell_center_world = WORLD_ORIGIN

    # Bands
    if bands is None:
        bands = ["Y106", "J129", "H158"]

    # epochs
    if simu_type == "imcom":
        n_epochs = 1

    # Prepare g1, g2
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)

    # Band loop
    final_dict = {}
    for band in tqdm(bands, desc="Band loop", disable=not verbose):
        epoch_list = []

        psf_maker.init_psf(band=band)
        pa = rng.uniform(low=150, high=190)

        # Epoch loop
        for _ in tqdm(
            range(n_epochs),
            desc="Epoch loop",
            leave=False,
            disable=not verbose,
        ):
            if simu_type in ["sca", "sh19"]:
                epoch_dict = make_exp(
                    rng,
                    galaxy_catalog,
                    psf_maker,
                    band,
                    cell_center_world,
                    g1,
                    g2,
                    pa_point=pa,
                    exp_time=exp_time,
                    cell_size_pix=cell_size_pix,
                    oversamp_factor=oversamp_factor,
                    chromatic=chromatic,
                    simple_noise=simple_noise,
                    noise_sigma=noise_sigma,
                    n_noise_realizations=n_noise_realizations,
                    image_factor=image_factor,
                    draw_method=draw_method,
                    n_photons=n_photons,
                    avg_gal_sed_path=avg_gal_sed_path,
                    make_deconv_psf=make_deconv_psf,
                    make_true_psf=make_true_psf,
                    verbose=verbose,
                )
                epoch_dict["cell_center_world"] = cell_center_world
                epoch_list.append(epoch_dict)
            elif simu_type == "imcom":
                epoch_dict = make_IMCOM(
                    rng,
                    galaxy_catalog,
                    psf_maker,
                    band,
                    cell_center_world,
                    g1,
                    g2,
                    exp_time=exp_time,
                    cell_size_pix=cell_size_pix,
                    simple_noise=simple_noise,
                    noise_sigma=noise_sigma,
                    n_noise_realizations=n_noise_realizations,
                    image_factor=6,
                    draw_method=draw_method,
                    n_photons=n_photons,
                    avg_gal_sed_path=avg_gal_sed_path,
                    make_psf=make_deconv_psf,  # Use make_deconv_psf for IMCOM
                    verbose=verbose,
                )
                epoch_dict["cell_center_world"] = cell_center_world
                epoch_list.append(epoch_dict)
            else:
                raise ValueError(
                    f"Unknown simu_type: {simu_type}. "
                    "Expected 'sca', 'imcom', or 'sh19'."
                )
        final_dict[band] = epoch_list

    return final_dict


def make_exp(
    rng,
    galaxy_catalog,
    psf_maker,
    band,
    cell_center_world,
    g1,
    g2,
    pa_point=0.0,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    chromatic=False,
    simple_noise=False,
    noise_sigma=None,
    n_noise_realizations=1,
    image_factor=1.0,
    draw_method="phot",
    n_photons=None,
    avg_gal_sed_path=None,
    make_deconv_psf=False,
    make_true_psf=False,
    verbose=True,
):
    """Make a single exposure for the simulation.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator to use for the simulation.
    galaxy_catalog : GalaxyCatalog
        The galaxy catalog to use for the simulation.
    psf_maker : PSFMaker
        The PSF maker to use for the simulation.
    band : str, optional
        The band to simulate.
        If None, defaults to 'Y106'.
    cell_center_world : galsim.CelestialCoord
        The celestial coordinates of the cell center.
    g1 : float or array-like, optional
        The shear component g1. Default: 0.0.
    g2 : float or array-like, optional
        The shear component g2. Default: 0.0.
    pa_point : float, optional
        The position angle in degrees for the WCS. Default: 0.0.
    exp_time : float, optional
        The exposure time in seconds. Default: 107.
    cell_size_pix : int, optional
        The size of the cell in pixels. Default: 500.
    oversamp_factor : int, optional
        The oversampling factor for the WCS. Default: 3.
    chromatic : bool, optional
        Whether the PSF is chromatic. Default: False.
    simple_noise : bool, optional
        Whether to use simple Gaussian noise instead of the full Roman noise
        model. Default: False.
    noise_sigma : float, optional
        The standard deviation of the Gaussian noise if `simple_noise` is True.
        Required if `simple_noise` is True. Default: 1.0.
    n_noise_realizations : int, optional
        The number of independent noise realizations to create. Default: 1.
    image_factor : float, optional
        A factor to scale the noise level. Default: 1.0.
    draw_method : str, optional
        The method to use for drawing the objects. Default: 'phot'.
    n_photons : int, optional
        The number of photons to use for the 'phot' draw method.
        If None, it will be calculated based on the galaxy flux
        and image_factor. Default: None.
    avg_gal_sed_path : str, optional
        The path to the average galaxy SED file.
        Required if `chromatic` is True.
    make_deconv_psf : bool, optional
        Whether to create the deconvolved PSF image. Default: True.
    make_true_psf : bool, optional
        Whether to create the true PSF image. Default: True.
    verbose : bool, optional
        Whether to print progress messages. Default: True.

    """
    bp_ = roman.getBandpasses()[band]
    seed_epoch = rng.randint(0, 2**32)

    sca = rng.randint(1, roman.n_sca + 1)
    epoch_dict = {
        "sca": sca,
    }
    exp_center = cell_center_world
    wcs = get_SCA_WCS(
        # cell_center_world,
        exp_center,
        sca,
        # PA=pa_point,
        PA=0.0 * galsim.degrees,
        img_size=(cell_size_pix, cell_size_pix),
    )
    # wcs = make_simple_exp_wcs(
    #     cell_center_world,
    #     PA=0.0,
    #     img_size=cell_size_pix,
    # )
    # wcs = make_simple_coadd_wcs(
    #     cell_center_world,
    #     img_size=cell_size_pix,
    # )
    epoch_dict["wcs"] = wcs
    epoch_dict["flux_scaling"] = galaxy_catalog.get_flux_scaling(band)

    psf = psf_maker.get_psf(
        sca=sca,
        image_pos=galsim.PositionD(roman.n_pix / 2, roman.n_pix / 2),
        wcs=wcs,
    )

    ## Make noise for science image
    noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
    rng_galsim = galsim.BaseDeviate(seed_epoch)
    if not simple_noise:
        make_roman_noise(
            noise_img,
            bp_,
            exp_time,
            # cell_center_world,
            exp_center,
            rng_galsim,
            image_factor=image_factor,
        )
    else:
        if noise_sigma is None:
            raise ValueError("noise_sigma must be provided")
        make_simple_noise(
            noise_img,
            noise_sigma / np.sqrt(image_factor),
            rng_galsim,
        )

    # Make avg PSF used for deconvolution
    wcs_oversampled = make_oversample_local_wcs(
        wcs,
        cell_center_world,
        oversamp_factor,
    )
    epoch_dict["wcs_oversampled"] = wcs_oversampled
    if make_deconv_psf:
        psf_img_deconv = get_deconv_psf(
            psf,
            wcs,
            wcs_oversampled,
            # cell_center_world,
            exp_center,
            bp=bp_,
            chromatic=chromatic,
            # avg_gal_sed_path=avg_gal_sed_path,
            avg_gal_sed_path=None,
        )
        epoch_dict["psf_avg"] = psf_img_deconv
    else:
        epoch_dict["psf_avg"] = None

    # Make true PSF
    if chromatic & make_true_psf:
        epoch_dict["psf_true_galsim"] = get_true_psf(
            galaxy_catalog.get_gsobject_delta().withFlux(1, bp_),
            psf,
            wcs_oversampled,
            bp_,
        )
    else:
        epoch_dict["psf_true_galsim"] = None
    epoch_dict["wcs_oversampled"] = wcs_oversampled

    # n_obj = galaxy_catalog.getNObjects()
    objlist = galaxy_catalog.getObjList(bandpass=bp_)
    n_obj = len(objlist["gsobject"])

    epoch_dict["sci"] = {}
    for g1_, g2_ in zip(g1, g2):
        # Make image
        final_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)

        # Object loop
        for obj_ind in tqdm(
            range(n_obj),
            total=n_obj,
            leave=False,
            desc="Obj loop",
            disable=not verbose,
        ):
            # Make obj
            gal = objlist["gsobject"][obj_ind]
            gal_flux = gal.flux
            dx = objlist["dx"][obj_ind]
            dy = objlist["dy"][obj_ind]

            gal = gal.shear(g1=g1_, g2=g2_)
            obj = galsim.Convolve(gal, psf)

            stamp_image = get_stamp(
                obj,
                dx,
                dy,
                wcs,
                cell_center_world,
                bp_,
                gal_flux,
                rng_galsim,
                draw_method=draw_method,
                image_factor=image_factor,
                n_photons=n_photons,
            )

            b = stamp_image.bounds & final_img.bounds
            if b.isDefined():
                final_img[b] += stamp_image[b]

        final_img /= roman.gain
        final_img += noise_img
        epoch_dict["sci"][f"shear_{g1_}_{g2_}"] = final_img.array
    # epoch_dict["noise"] = noise_img.array
    epoch_dict["noise_var"] = noise_img.array.var()
    epoch_dict["weight"] = (
        np.ones_like(noise_img.array) / noise_img.array.var()
    )

    # Make independent noise image
    noise_realizations = []
    for _ in range(n_noise_realizations):
        noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
        if not simple_noise:
            make_roman_noise(
                noise_img,
                bp_,
                exp_time,
                # cell_center_world,
                exp_center,
                rng_galsim,
                image_factor=image_factor,
            )
        else:
            if noise_sigma is None:
                raise ValueError("noise_sigma must be provided")
            make_simple_noise(
                noise_img,
                noise_sigma / np.sqrt(image_factor),
                rng_galsim,
            )
        noise_realizations.append(noise_img.array)
    epoch_dict["noise"] = noise_realizations

    return epoch_dict


def make_IMCOM(
    rng,
    galaxy_catalog,
    psf_maker,
    band,
    cell_center_world,
    g1,
    g2,
    exp_time=107,
    cell_size_pix=500,
    simple_noise=False,
    noise_sigma=None,
    n_noise_realizations=1,
    image_factor=6.0,
    draw_method="phot",
    n_photons=None,
    avg_gal_sed_path=None,
    make_psf=False,
    verbose=True,
):
    """Make a single exposure for the simulation.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator to use for the simulation.
    galaxy_catalog : GalaxyCatalog
        The galaxy catalog to use for the simulation.
    psf_maker : PSFMaker
        The PSF maker to use for the simulation.
    band : str, optional
        The band to simulate.
        If None, defaults to 'Y106'.
    cell_center_world : galsim.CelestialCoord
        The celestial coordinates of the cell center.
    g1 : float or array-like, optional
        The shear component g1. Default: 0.0.
    g2 : float or array-like, optional
        The shear component g2. Default: 0.0.
    exp_time : float, optional
        The exposure time in seconds. Default: 107.
    cell_size_pix : int, optional
        The size of the cell in pixels. Default: 500.
    simple_noise : bool, optional
        Whether to use simple Gaussian noise instead of the full Roman noise
        model. Default: False.
    noise_sigma : float, optional
        The standard deviation of the Gaussian noise if `simple_noise` is True.
        Required if `simple_noise` is True. Default: 1.0.
    n_noise_realizations : int, optional
        The number of independent noise realizations to create. Default: 1.
    image_factor : float, optional
        A factor to scale the noise level. Default: 1.0.
    draw_method : str, optional
        The method to use for drawing the objects. Default: 'phot'.
    n_photons : int, optional
        The number of photons to use for the 'phot' draw method.
        If None, it will be calculated based on the galaxy flux
        and image_factor. Default: None.
    avg_gal_sed_path : str, optional
        The path to the average galaxy SED file.
        Required if `chromatic` is True.
    make_psf : bool, optional
        Whether to create the PSF image. Default: False.
    verbose : bool, optional
        Whether to print progress messages. Default: True.

    """
    bp_ = roman.getBandpasses()[band]
    seed_epoch = rng.randint(0, 2**32)

    imcom_dict = {}
    exp_center = cell_center_world
    wcs = get_IMCOM_WCS(
        cell_center_world,
        img_size=cell_size_pix,
    )
    imcom_dict["wcs"] = wcs
    imcom_dict["flux_scaling"] = galaxy_catalog.get_flux_scaling(band)

    psf = psf_maker.get_psf()

    ## Make noise
    noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
    rng_galsim = galsim.BaseDeviate(seed_epoch)
    if not simple_noise:
        make_roman_noise(
            noise_img,
            bp_,
            exp_time,
            exp_center,
            rng_galsim,
            image_factor=image_factor,
        )
    else:
        if noise_sigma is None:
            raise ValueError("noise_sigma must be provided")
        make_simple_noise(
            noise_img,
            noise_sigma,
            rng_galsim,
        )

    # if make_deconv_psf:
    #     psf_img_deconv = get_deconv_psf(
    #         psf,
    #         wcs,
    #         wcs_oversampled,
    #         # cell_center_world,
    #         exp_center,
    #         bp=bp_,
    #         chromatic=chromatic,
    #         # avg_gal_sed_path=avg_gal_sed_path,
    #         avg_gal_sed_path=None,
    #     )
    #     imcom_dict["psf_avg"] = psf_img_deconv
    # else:
    #     imcom_dict["psf_avg"] = None

    # Make true PSF
    if make_psf:
        imcom_dict["psf"] = get_true_psf(
            galsim.DeltaFunction().withFlux(1),
            psf,
            wcs,
            bp=None,
            draw_method="no_pixel",
        )
    else:
        imcom_dict["psf"] = None

    objlist = galaxy_catalog.getObjList(bandpass=bp_)
    n_obj = len(objlist["gsobject"])

    imcom_dict["sci"] = {}
    for g1_, g2_ in zip(g1, g2):
        # Make image
        final_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)

        # Object loop
        for obj_ind in tqdm(
            range(n_obj),
            total=n_obj,
            leave=False,
            desc="Obj loop",
            disable=not verbose,
        ):
            # Make obj
            gal = objlist["gsobject"][obj_ind]
            gal_flux = gal.flux
            dx = objlist["dx"][obj_ind]
            dy = objlist["dy"][obj_ind]

            gal = gal.shear(g1=g1_, g2=g2_)
            obj = galsim.Convolve(gal, psf)

            stamp_image = get_stamp(
                obj,
                dx,
                dy,
                wcs,
                cell_center_world,
                bp_,
                gal_flux,
                rng_galsim,
                draw_method=draw_method,
                image_factor=image_factor,
                n_photons=n_photons,
            )

            b = stamp_image.bounds & final_img.bounds
            if b.isDefined():
                final_img[b] += stamp_image[b]

        final_img /= roman.gain
        final_img += noise_img
        imcom_dict["sci"][f"shear_{g1_}_{g2_}"] = final_img.array
    # imcom_dict["noise"] = noise_img.array
    imcom_dict["noise_var"] = noise_img.array.var()
    imcom_dict["weight"] = (
        np.ones_like(noise_img.array) / noise_img.array.var()
    )

    # Make independent noise image
    noise_realizations = []
    for _ in range(n_noise_realizations):
        new_noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
        make_simple_noise(
            new_noise_img,
            noise_sigma,
            rng_galsim,
        )
        noise_realizations.append(new_noise_img.array)
    imcom_dict["noise"] = noise_realizations

    return imcom_dict


def get_stamp(
    obj,
    dx,
    dy,
    wcs,
    cell_center_world,
    bp,
    gal_flux,
    rng_galsim,
    draw_method="phot",
    image_factor=1.0,
    n_photons=None,
):
    """Get a stamp for one object. This where we draw the object.

    Parameters
    ----------
    obj : galsim.GSObject
        The object to draw.
    dx : float
        The x offset in arcseconds from the cell center.
    dy : float
        The y offset in arcseconds from the cell center.
    wcs : galsim.BaseWCS
        The WCS object for the image.
    cell_center_world : galsim.CelestialCoord
        The celestial coordinates of the cell center.
    bp : galsim.Bandpass
        The bandpass for which to draw the object.
    gal_flux : float
        The flux of the galaxy in the bandpass.
    rng_galsim : galsim.BaseDeviate
        The random number generator to use for drawing the object.
    draw_method : str, optional
        The method to use for drawing the object. Default: 'phot'.
    image_factor : float, optional
        A factor to scale the noise level. Default: 1.0.
    n_photons : int, optional
        The number of photons to use for the 'phot' draw method.
        If None, it will be calculated based on the galaxy flux
        and image_factor. Default: None.

    Returns
    -------
    galsim.Image
        The drawn image of the object.

    """
    # gal = objlist["gsobject"][obj_ind]
    # dx = objlist["dx"][obj_ind]
    # dy = objlist["dy"][obj_ind]

    # # Make obj
    # gal = gal.shear(g1=g1, g2=g2)
    # obj = galsim.Convolve([gal, psf])

    world_pos = cell_center_world.deproject(
        dx * galsim.arcsec,
        dy * galsim.arcsec,
    )
    image_pos = wcs.toImage(world_pos)

    # # Set center and offset
    # nominal_x = image_pos.x + 0.5
    # nominal_y = image_pos.y + 0.5

    # stamp_center = galsim.PositionI(
    #     int(math.floor(nominal_x + 0.5)),
    #     int(math.floor(nominal_y + 0.5)),
    # )
    # stamp_offset = galsim.PositionD(
    #     nominal_x - stamp_center.x, nominal_y - stamp_center.y
    # )

    if draw_method == "phot":
        stamp_size = 150
        rng_draw = rng_galsim
        maxN = int(1e6)
        if n_photons is None:
            n_photons = 0.0
            if image_factor > 1.0:
                n_photons = galsim.PoissonDeviate(rng_galsim, mean=gal_flux)()
                n_photons *= image_factor
    else:
        # stamp_size = obj.getGoodImageSize(roman.pixel_scale)
        stamp_size = 100
        rng_draw = None
        maxN = None
        n_photons = 0.0

    stamp_image = obj.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        bandpass=bp,
        wcs=wcs.local(world_pos=world_pos),
        method=draw_method,
        center=image_pos,
        rng=rng_draw,
        n_photons=n_photons,
        maxN=maxN,
    )

    return stamp_image


def get_true_psf(star, psf, wcs, bp, draw_method="no_pixel"):
    """Get the true PSF image.

    Parameters
    ----------
    star : galsim.DeltaFunction
        The star object to use for the PSF with the SED.
    psf : galsim.GSObject
        The PSF object.
    wcs : galsim.BaseWCS
        The WCS object for the image.
    bp : galsim.Bandpass
        The bandpass for which to draw the PSF.
    draw_method : str, optional
        The method to use for drawing the PSF. Default: 'no_pixel'.

    Returns
    -------
    np.ndarray
        The true PSF image array.

    """
    tmp_true_ = galsim.Convolve(
        star,
        psf,
    )

    true_psf_img = tmp_true_.drawImage(
        nx=301,
        ny=301,
        wcs=wcs,
        bandpass=bp,
        method=draw_method,
        # method="phot",
        # n_photons=1e6,
    )

    # interp_img = galsim.InterpolatedImage(
    #     true_psf_img, x_interpolant="lanczos15"
    # )
    # pix = wcs.toWorld(galsim.Pixel(1))
    # inv_pix = galsim.Deconvolve(pix)
    # interp_img = galsim.Convolve(interp_img, inv_pix)

    # return interp_img
    return true_psf_img.array


def get_deconv_psf(
    psf,
    wcs,
    wcs_oversampled,
    cell_center_world,
    bp=None,
    chromatic=False,
    avg_gal_sed_path=None,
):
    """Get the PSF image used for the deconvolution.

    Parameters
    ----------
    psf : galsim.GSObject
        The PSF object.
    wcs : galsim.BaseWCS
        The WCS object for the image.
    wcs_oversampled : galsim.BaseWCS
        The oversampled WCS object for the image.
    cell_center_world : galsim.CelestialCoord
        The celestial coordinates of the image center.
    bp : galsim.Bandpass, optional
        The bandpass for which to draw the PSF.
        Required if `chromatic` is True.
    chromatic : bool, optional
        Whether the PSF is chromatic. Default: False.
    avg_gal_sed_path : str, optional
        The path to the average galaxy SED file.
        Required if `chromatic` is True.
        Default: None.

    Returns
    -------
    np.ndarray
        The PSF image array.

    """
    star_psf = galsim.DeltaFunction()

    # Average galaxy SED
    if chromatic:
        if avg_gal_sed_path is None:
            sed = galsim.SED("vega.txt", "nm", "flambda")
            # sed = galsim.SED(
            #     galsim.LookupTable(
            #         [100, 1000, 2000], [0.0, 1.0, 10.0], interpolant="linear"
            #     ),
            #     wave_type="nm",
            #     flux_type="flambda",
            # )
            # avg_star_sed_path = (
            #     "/Users/aguinot/Documents/roman/test_metacoadd/"
            #     "star_avg_sed.npz"
            # )
            # sed_wave, avg_star_sed_arr = (
            #     np.load(avg_star_sed_path)["x"],
            #     np.load(avg_star_sed_path)["y"],
            # )
            # sed_lt = galsim.LookupTable(
            #     sed_wave, avg_star_sed_arr, interpolant="linear"
            # )
            # sed = galsim.SED(sed_lt, wave_type="nm", flux_type="flambda")
        else:
            sed_wave, avg_gal_sed_arr = np.load(avg_gal_sed_path)
            sed_lt = galsim.LookupTable(
                sed_wave, avg_gal_sed_arr, interpolant="linear"
            )
            sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")
        star_psf *= sed.withFlux(1, bp)

    wcs_local_center = wcs.local(world_pos=cell_center_world)
    pixel_ori = wcs_local_center.toWorld(galsim.Pixel(1))
    psf_obj = galsim.Convolve(psf, star_psf)
    psf_obj = galsim.Convolve(psf_obj, pixel_ori)
    psf_img = psf_obj.drawImage(
        nx=301,
        ny=301,
        wcs=wcs_oversampled,
        bandpass=bp,
        method="no_pixel",
        # method="phot",
        # n_photons=1e6,
        # rng=galsim.BaseDeviate(42),
    )

    return psf_img.array
