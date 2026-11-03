import numpy as np
from scipy.stats import skewnorm

import galsim
from galsim import roman

from .constant import (
    ROMAN_N_EFF,
    DEFAULT_HLR,
    DEFAULT_MAG,
    COMMON_ZERO_POINT,
    WORLD_ORIGIN,
    IMCOM_PIXEL_SCALE,
)
from .skycatlog_parser import (
    SkyCatalogParser,
    get_knot_n,
    get_knot_size,
    COMPONENTS,
)


LAYOUT_KINDS = ["grid", "random"]
GAL_TYPES = ["gauss", "exp", "dev"]


class SimpleGalaxyCatalog:
    """A simple galaxy catalog with a single galaxy type.

    Parameters
    ----------
    img_size : int
        Image size in pixels.
    seed : int
        Random seed for the catalog.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation
        and 'imcom' for IMCOM simulation. Default: 'sca'.
    gal_type : str
        Galaxy type, one of ['gauss', 'exp', 'dev']. Default: 'exp'.
    hlr : float
        Half-light radius of the galaxy. Default: DEFAULT_HLR.
    mag : float
        Magnitude of the galaxy. Default: DEFAULT_MAG.
    layout_kind : str
        Layout kind, one of ['grid', 'random']. Default: 'grid'.
    buffer : int
        Buffer size in pixels. Default: 0.
    spacing : float
        Spacing between galaxies in pixels. Default: 5.0.
    n_gal : int or None
        Number of galaxies to draw. If None, will use as many as possible.
        Default: None.
    exp_time : float
        Exposure time in seconds.
    chromatic : bool
        Whether to use chromatic SEDs. Default: False.
    gal_sed_path : str or None
        Path to the galaxy SED file. If None, will use a default SED. Default:
        None.

    """

    def __init__(
        self,
        img_size,
        seed,
        simu_type="sca",
        gal_type="exp",
        hlr=DEFAULT_HLR,
        mag=DEFAULT_MAG,
        layout_kind="grid",
        buffer=0,
        spacing=5.0,
        n_gal=None,
        exp_time=roman.exptime,
        chromatic=False,
        gal_sed_path=None,
    ):
        self._simu_type = simu_type
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        if gal_type not in GAL_TYPES:
            raise ValueError(f"gal_type must be one of {GAL_TYPES}, got {gal_type}")
        self.gal_type = gal_type

        self._hlr = hlr
        self._mag = mag
        self._exp_time = exp_time

        if layout_kind not in LAYOUT_KINDS:
            raise ValueError(
                f"layout_kind must be one of {LAYOUT_KINDS}, got {layout_kind}"
            )
        self.layout_kind = layout_kind
        self.buffer = buffer
        self.spacing = spacing
        self.n_gal = n_gal

        self._chromatic = chromatic
        self._init_catalog(chromatic, gal_sed_path)

    def getNObjects(self):
        """Get the number of objects in the catalog.

        Returns
        -------
        int
            The number of objects in the catalog.

        """
        return self._n_gal

    def getObjList(self, bandpass):
        """Get the object list to draw for a given bandpass.

        Parameters
        ----------
        bandpass : galsim.Bandpass
            The bandpass to use for the objects.

        Returns
        -------
        dict
            A dictionary containing the object list.

        """
        zp = self._get_zeropoint(bandpass)
        objlist = {
            "gsobject": [],
            "dx": [],
            "dy": [],
        }
        for i in range(self.getNObjects()):
            flux = self.get_flux(i, zp)
            if self._chromatic:
                gsobject = self._get_gsobject(i).withFlux(flux, bandpass=bandpass)
                gsobject.flux = flux
            else:
                gsobject = self._get_gsobject(i).withFlux(flux)
            objlist["gsobject"].append(gsobject)
            objlist["dx"].append(self.dx[i])
            objlist["dy"].append(self.dy[i])
        return objlist

    def _init_catalog(self, chromatic=False, gal_sed_path=None):
        self.dx, self.dy = get_simple_pos(
            self._simu_type,
            self.img_size,
            self.rng,
            layout_kind=self.layout_kind,
            buffer=self.buffer,
            spacing=self.spacing,
            n_gal=self.n_gal,
        )
        self._n_gal = len(self.dx)

        gal_sed = self._get_sed(gal_sed_path) if chromatic else 1

        self._set_gsobject(gal_sed)
        self._set_gsobject_delta(gal_sed)

    def _get_gsobject(self, index):
        return self.gsobject

    def get_gsobject_delta(self):
        """Get a galsim DeltaFunction gsobject with the appropriate SED.

        Returns
        -------
        galsim.DeltaFunction
            The galsim DeltaFunction gsobject.

        """
        return self.gsobject_delta

    def get_flux(self, index, zp):
        """Get the flux of the object at a given index in the catalog.

        Parameters
        ----------
        index : int
            The index of the object.
        zp : float
            The zero point to use for the flux calculation.

        Returns
        -------
        float
            The flux of the object.

        """
        return 10 ** (-0.4 * (self._mag - zp))

    def _set_gsobject(self, gal_sed=1):
        if self.gal_type == "gauss":
            self.gsobject = galsim.Gaussian(half_light_radius=self._hlr)
        elif self.gal_type == "exp":
            self.gsobject = galsim.Exponential(half_light_radius=self._hlr)
        elif self.gal_type == "dev":
            self.gsobject = galsim.DeVaucouleurs(half_light_radius=self._hlr)

        self.gsobject *= gal_sed

    def _set_gsobject_delta(self, gal_sed=None):
        self.gsobject_delta = galsim.DeltaFunction()
        if gal_sed is not None:
            self.gsobject_delta *= gal_sed

    def _get_zeropoint(self, bp):
        return (
            bp.zeropoint
            + 2.5 * np.log10(self._exp_time * roman.collecting_area)
            - 2.5 * np.log10(roman.gain)
        )

    def get_flux_scaling(self, band, target_zp=COMMON_ZERO_POINT):
        """Get the flux scaling factor to convert from the current zeropoint
        to the target zeropoint.

        Parameters
        ----------
        band : str
            The band to use for the flux scaling.
        target_zp : float
            The target zero point.

        Returns
        -------
        float
            The flux scaling factor.

        """
        bp = roman.getBandpasses()[band]
        current_zp = self._get_zeropoint(bp)
        zp_diff = target_zp - current_zp
        return 10 ** (0.4 * zp_diff)

    def _get_sed(self, gal_sed_path):
        sed_wave, avg_gal_sed_arr = np.load(gal_sed_path)
        sed_lt = galsim.LookupTable(sed_wave, avg_gal_sed_arr, interpolant="linear")
        sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")
        return sed


class GalaxyCatalog(SimpleGalaxyCatalog):
    """Galaxy simulation from Sheldon et al. 2019
    ref: https://arxiv.org/abs/1911.02505
    Section 3.2

    Parameters
    ----------
    img_size : int
        Image size in pixels.
    seed : int
        Random seed for the catalog.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation
        and 'imcom' for IMCOM simulation. Default: 'sca'.
    flux_range : list or None
        Range of flux values for the galaxies.
        Default: None, which means no limit.
    layout_kind : str
        Layout kind, one of ['grid', 'random']. Default: 'grid'.
    buffer : int
        Buffer size in pixels. Default: 0.
    spacing : float
        Spacing between galaxies in pixels. Default: 5.0.
    n_gal : int or None
        Number of galaxies to draw. If None, will use as many as possible.
        Default: None.
    exp_time : float
        Exposure time in seconds.
    chromatic : bool
        Whether to use chromatic SEDs. Default: False.
    gal_sed_path : str or None
        Path to the galaxy SED file. If None, will use a default SED.
        Default: None.
    ref_band : str
        Reference band for flux range selection. Default: "Y106".

    """

    def __init__(
        self,
        img_size,
        seed,
        simu_type="sca",
        flux_range=None,
        flux_dist="uniform",
        hlr_range=None,
        layout_kind="grid",
        buffer=0,
        spacing=5.0,
        n_gal=None,
        exp_time=roman.exptime,
        chromatic=False,
        gal_sed_path=None,
        ref_band="Y106",
    ):
        # self._mag_range = mag_range
        if flux_range is None:
            flux_range = [10, 10_000]
        self._flux_range = flux_range
        self.flux_dist = flux_dist

        if hlr_range is None:
            hlr_range = [0.1, 0.4]
        self._hlr_range = hlr_range
        self.set_bandpass_ref(ref_band)
        super().__init__(
            img_size,
            seed,
            simu_type=simu_type,
            layout_kind=layout_kind,
            buffer=buffer,
            spacing=spacing,
            n_gal=n_gal,
            exp_time=exp_time,
            chromatic=chromatic,
            gal_sed_path=gal_sed_path,
        )

    def getObjList(self, bandpass):
        """Get the object list to draw for a given bandpass.

        Parameters
        ----------
        bandpass : galsim.Bandpass
            The bandpass to use for the objects.

        Returns
        -------
        dict
            A dictionary containing the object list.

        """
        objlist = {
            "gsobject": [],
            "dx": [],
            "dy": [],
        }
        for i in range(self.getNObjects()):
            gsobject = self._get_gsobject(i)
            objlist["gsobject"].append(gsobject)
            objlist["dx"].append(self.dx[i])
            objlist["dy"].append(self.dy[i])
        return objlist

    def _get_gsobject(self, index):
        return self.gsobject_list[index]

    def set_bandpass_ref(self, ref_band):
        """Set the reference bandpass for the flux range selection.

        Parameters
        ----------
        ref_band : str
            The reference bandpass name.

        """
        self._bp_ref = roman.getBandpasses()[ref_band]

    def _set_gsobject(self, gal_sed=1):
        self.gsobject_list = []
        for _ in range(self._n_gal):
            if self.flux_dist == "uniform":
                flux_tot = self.rng.uniform(
                    low=self._flux_range[0], high=self._flux_range[1]
                )
            elif self.flux_dist == "log_normal":
                log_flux = skewnorm.rvs(
                    a=5,
                    loc=np.log(5),
                    scale=1,
                    random_state=self.rng,
                )
                flux_tot = np.exp(log_flux)
                flux_tot = np.clip(flux_tot, self._flux_range[0], self._flux_range[1])

            BT_flux = self.rng.uniform(low=0, high=1)
            bulge_flux = flux_tot * BT_flux
            disk_flux = flux_tot - bulge_flux

            disk_ell = get_g_BA(self.rng, sigma=0.2, size=1)[0]
            disk_theta = self.rng.uniform(0, 2 * np.pi)
            disk_hlr = self.rng.uniform(self._hlr_range[0], self._hlr_range[1])

            bulge_ell = self.rng.uniform(0, 0.4) * disk_ell
            bulge_theta = disk_theta
            bulge_hlr = self.rng.uniform(0.4, 0.6) * disk_hlr

            shift_amp = self.rng.uniform(0.0, 0.05) * disk_hlr
            shift_angle = self.rng.uniform(0, 2 * np.pi)
            bulge_x_shift = shift_amp * np.cos(shift_angle)
            bulge_y_shift = shift_amp * np.sin(shift_angle)

            disk = galsim.Exponential(half_light_radius=disk_hlr, flux=disk_flux).shear(
                g=disk_ell, beta=disk_theta * galsim.radians
            )
            bulge = (
                galsim.DeVaucouleurs(half_light_radius=bulge_hlr, flux=bulge_flux)
                .shear(g=bulge_ell, beta=bulge_theta * galsim.radians)
                .shift(bulge_x_shift, bulge_y_shift)
            )
            if self._chromatic:
                new_gal_sed = gal_sed.withFlux(flux_tot, bandpass=self._bp_ref)
            else:
                new_gal_sed = 1.0

            self.gsobject_list.append(galsim.Add(disk + bulge) * new_gal_sed)
            if self._chromatic:
                self.gsobject_list[-1].flux = flux_tot


class DiffSkyCatalog(GalaxyCatalog):
    """Build catalog based on the DiffSky simulation.

    Parameters
    ----------
    img_size : int
        Image size in pixels.
    seed : int
        Random seed for the catalog.
    skycatalog_config_path : str
        Path to the SkyCatalogs configuration file.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation
        and 'imcom' for IMCOM simulation. Default: 'sca'.
    object_types : list
        List of object types to include in the catalog.
        Default: ['diffsky_galaxy'].
    do_knots : bool
        Whether to include knots in the galaxy model. Default: True.
    flux_range : list or None
        Range of flux values for the galaxies.
        Default: None, which means no limit.
    cell_world_center : galsim.celestial.CelestialCoord
        Center of the cell in world coordinates (RA, Dec).
        Default: WORLD_ORIGIN.
    buffer : int
        Buffer size around the image in pixels. Default: 0.
    exp_time : float
        Exposure time in seconds. Default: roman.exptime.
    chromatic : bool
        Whether to include chromatic effects in the galaxy model.
        Default: False.
    gal_sed_path : str or None
        If provided, will use this SED for all galaxies. Path to the galaxy
        SED file. Default: None.
    ref_band : str
        Reference band for flux range selection. Default: "Y106".

    """

    def __init__(
        self,
        img_size,
        seed,
        skycatalog_config_path,
        simu_type="sca",
        object_types=None,
        do_knots=True,
        flux_range=None,
        cell_world_center=WORLD_ORIGIN,
        buffer=0,
        exp_time=roman.exptime,
        chromatic=False,
        gal_sed_path=None,
        ref_band="Y106",
    ):
        self.img_size = img_size
        self._simu_type = simu_type
        if object_types is None:
            self._object_types = ["diffsky_galaxy"]
        else:
            self._object_types = object_types
        self._do_knots = do_knots
        self._coadd_center = cell_world_center
        self._chromatic = chromatic
        self._exp_time = exp_time
        self._gal_sed_path = gal_sed_path
        self.rng = np.random.RandomState(seed)
        self.galsim_rng = galsim.BaseDeviate(seed)

        self.set_bandpass_ref(ref_band)
        self._set_flux(flux_range)
        self._init_catalog(
            cell_world_center,
            img_size,
            skycatalog_config_path,
            simu_type=simu_type,
            buffer=buffer,
            chromatic=chromatic,
            gal_sed_path=gal_sed_path,
        )

    def _init_catalog(
        self,
        coadd_center,
        img_size,
        skycatalog_config_path,
        simu_type="sca",
        buffer=0,
        chromatic=True,
        gal_sed_path=None,
    ):
        self.skycat_parser = SkyCatalogParser(
            skycatalog_config_path,
            coadd_center,
            img_size,
            simu_type=simu_type,
            buffer=buffer,
            object_types=self._object_types,
        )

        n_obj = 0
        for object_type in self._object_types:
            self.skycat_parser.set_catalog(object_type)
            self.skycat_parser.set_sed_catalog(object_type)
            n_obj += len(self.skycat_parser.catalog[object_type])
        self._n_gal = n_obj

        self._fixed_sed = False
        if gal_sed_path is not None and chromatic:
            gal_sed = self._get_sed(gal_sed_path)
            self._fixed_sed = True
        else:
            gal_sed = None

        self._init_pos()
        self._set_gsobject(gal_sed=gal_sed, rng=self.galsim_rng)
        self._set_gsobject_delta(gal_sed)

    def _init_pos(self):
        self.dx = np.empty(self._n_gal, dtype=np.float64)
        self.dy = np.empty(self._n_gal, dtype=np.float64)

    def _get_gsobject(self, index, bandpass=None):
        sed_dict = self.sed_comp_list[index]
        gsobj_dict = self.gsobject_list[index]
        flux_dict, flux_tot = self._get_flux_components(sed_dict, bandpass, cache=True)
        # if flux_tot < self._flux_range[0] or flux_tot > self._flux_range[1]:
        #     return None
        if self._chromatic:
            if not self._fixed_sed:
                gsobj = galsim.Add(
                    list(gsobj_dict.values()),
                )
                gsobj.flux = flux_tot
            else:
                gsobj = galsim.Add(
                    [
                        gsobj_dict[comp].withFlux(flux_dict[comp], bandpass=bandpass)
                        for comp in gsobj_dict
                    ]
                )
                gsobj.flux = flux_tot
        else:
            gsobj = galsim.Add(
                [gsobj_dict[comp].withFlux(flux_dict[comp]) for comp in gsobj_dict]
            )

        return gsobj

    def getObjList(self, bandpass):
        """Get the object list to draw for a given bandpass.

        Parameters
        ----------
        bandpass : galsim.Bandpass
            The bandpass to use for the objects.

        Returns
        -------
        dict
            A dictionary containing the object list.

        """
        objlist = {
            "gsobject": [],
            "dx": [],
            "dy": [],
        }
        for i in range(len(self.gsobject_list)):
            if self._chromatic:
                gsobject = self._get_gsobject(i, bandpass)
            else:
                gsobject = self._get_gsobject(i, bandpass)
            if gsobject is None:
                continue
            objlist["gsobject"].append(gsobject)
            objlist["dx"].append(self.dx[i])
            objlist["dy"].append(self.dy[i])
        return objlist

    def _set_gsobject(self, gal_sed=None, gsparams=None, rng=None):
        self.gsobject_list = []
        self.sed_comp_list = []
        for object_type in self._object_types:
            for i, row_ in enumerate(
                self.skycat_parser.catalog[object_type].itertuples()
            ):
                row = row_._asdict()
                sed_row = (
                    self.skycat_parser.sed_catalog[object_type]
                    .loc[row["Index"][2]]
                    .to_dict()
                )
                _, flux_tot = self._get_flux_components(sed_row, self._bp_ref)
                if flux_tot < self._flux_range[0] or flux_tot > self._flux_range[1]:
                    continue

                if object_type == "diffsky_galaxy":
                    gsobj = self._get_diffsky_gsobject(
                        row,
                        sed_row,
                        gal_sed=gal_sed,
                        gsparams=gsparams,
                        rng=rng,
                    )
                elif object_type == "star":
                    gsobj = self._get_star_gsobject(row, sed_row, gsparams=gsparams)
                self.gsobject_list.append(gsobj)
                self.sed_comp_list.append(sed_row)
                self._get_pos(i, row["ra"], row["dec"])
                if len(self.gsobject_list) >= self._n_gal:
                    break
            if len(self.gsobject_list) >= self._n_gal:
                break

    def _get_diffsky_gsobject(
        self,
        row,
        sed_row,
        gal_sed=None,
        gsparams=None,
        rng=None,
    ):
        if rng is None:
            rng = galsim.BaseDeviate(int(row["galaxy_id"]))

        comp_dict = {}
        for component in COMPONENTS:
            # knots use the same major/minor axes as the disk component.
            my_cmp = "disk" if component != "bulge" else "spheroid"
            hlr = row[f"{my_cmp}HalfLightRadiusArcsec"]

            # Get ellipticities saved in catalog. Not sure they're what
            # we need
            e1 = row[f"{my_cmp}Ellipticity1"]
            e2 = row[f"{my_cmp}Ellipticity2"]
            shear = galsim.Shear(g1=e1, g2=e2)

            if component == "knots" and self._do_knots:
                npoints = get_knot_n(
                    row["um_source_galaxy_obs_sm"],
                    gal_id=row["galaxy_id"],
                )
                assert npoints > 0
                knot_profile = galsim.Sersic(
                    n=1,
                    half_light_radius=hlr / 2.0,
                    gsparams=gsparams,
                )
                knot_profile = knot_profile._shear(shear)
                obj = galsim.RandomKnots(
                    npoints=npoints,
                    profile=knot_profile,
                    rng=rng,
                    gsparams=gsparams,
                )
                z = row["redshift"]
                size = get_knot_size(z)  # get knot size
                if size is not None:
                    obj = galsim.Convolve(obj, galsim.Gaussian(sigma=size))
            else:
                n = 1 if component == "disk" else 4
                obj_ = galsim.Sersic(n=n, half_light_radius=hlr, gsparams=gsparams)
                obj = obj_._shear(shear)
            if self._chromatic:
                if gal_sed is None:
                    sed = sed_row[component] * self._exp_time * roman.collecting_area
                else:
                    sed = gal_sed
            else:
                sed = 1.0
            comp_dict[component] = obj * sed
        return comp_dict

    def _get_star_gsobject(self, row, sed_row, gsparams=None):
        raise NotImplementedError

    def _get_pos(self, i, ra, dec):
        coord = galsim.CelestialCoord(
            ra=ra * galsim.degrees,
            dec=dec * galsim.degrees,
        )
        u, v = self._coadd_center.project(coord)
        self.dx[i] = u.deg * 3600
        self.dy[i] = v.deg * 3600

    def _get_flux_components(self, sed_dict, bandpass, cache=True):
        flux_dict = {}
        flux_tot = 0.0
        for key in sed_dict:
            flux = (
                sed_dict[key].calculateFlux(bandpass)
                * self._exp_time
                * roman.collecting_area
            )
            flux_dict[key] = flux
            flux_tot += flux
        return flux_dict, flux_tot

    def _set_flux(self, flux_range):
        if flux_range is None:
            self._flux_range = [-np.inf, np.inf]
        elif isinstance(flux_range, (list, tuple)):
            if len(flux_range) != 2:
                raise ValueError(
                    "If provided, flux_range should be a list or tuple of"
                    f"length 2, got {len(flux_range)}"
                )
            flux_range_tmp = [None, None]
            if flux_range[0] is None:
                flux_range_tmp[0] = -np.inf
            elif isinstance(flux_range[0], (int, float)):
                flux_range_tmp[0] = float(flux_range[0])
            else:
                raise ValueError(
                    "flux_range[0] should be a number or None, got "
                    f"{type(flux_range[0])}"
                )
            if flux_range[1] is None:
                flux_range_tmp[1] = np.inf
            elif isinstance(flux_range[1], (int, float)):
                flux_range_tmp[1] = float(flux_range[1])
            else:
                raise ValueError(
                    "flux_range[1] should be a number or None, got "
                    f"{type(flux_range[1])}"
                )
            if flux_range_tmp[0] >= flux_range_tmp[1]:
                raise ValueError(
                    f"flux_range[0] should be less than flux_range[1], got {flux_range}"
                )
            self._flux_range = flux_range_tmp
        else:
            raise ValueError(
                "flux_range should be None or a list or tuple of length 2, "
                f"got {flux_range}"
            )


class SimpleDiffSkyCatalog(DiffSkyCatalog):
    """A simple DiffSky catalog with galaxies arranged in a grid or random
    layout.

    Parameters
    ----------
    img_size : int
        Image size in pixels.
    seed : int
        Random seed for the catalog.
    skycatalog_config_path : str
        Path to the SkyCatalogs configuration file.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation
        and 'imcom' for IMCOM simulation. Default: 'sca'.
    object_types : list
        List of object types to include in the catalog.
        Default: ['diffsky_galaxy'].
    do_knots : bool
        Whether to include knots in the galaxy model. Default: True.
    flux_range : list or None
        Range of flux values for the galaxies.
        Default: None, which means no limit.
    cell_world_center : galsim.celestial.CelestialCoord
        Center of the cell in world coordinates (RA, Dec).
        Default: WORLD_ORIGIN.
    layout_kind : str
        Layout kind, one of ['grid', 'random'].
    buffer : int
        Buffer size around the image in pixels. Default: 0.
    spacing : float
        Spacing between galaxies in arcsec. Default: 5.0.
    n_gal : int or None
        Number of galaxies to draw. If None, will use as many as possible.
        Default: None.
    exp_time : float
        Exposure time in seconds. Default: roman.exptime.
    chromatic : bool
        Whether to include chromatic effects in the galaxy model.
        Default: False.
    gal_sed_path : str or None
        If provided, will use this SED for all galaxies. Path to the galaxy
        SED file. Default: None.
    ref_band : str
        Reference band for flux range selection. Default: "Y106".

    """

    def __init__(
        self,
        img_size,
        seed,
        skycatalog_config_path,
        simu_type="sca",
        object_types=None,
        do_knots=True,
        flux_range=None,
        cell_world_center=WORLD_ORIGIN,
        layout_kind="grid",
        buffer=0,
        spacing=5.0,
        n_gal=None,
        exp_time=roman.exptime,
        chromatic=False,
        gal_sed_path=None,
        ref_band="Y106",
    ):
        self.layout_kind = layout_kind
        self.buffer = buffer
        self.spacing = spacing
        self.n_gal = n_gal

        super().__init__(
            img_size,
            seed,
            skycatalog_config_path=skycatalog_config_path,
            simu_type=simu_type,
            object_types=object_types,
            do_knots=do_knots,
            flux_range=flux_range,
            cell_world_center=cell_world_center,
            exp_time=exp_time,
            chromatic=chromatic,
            gal_sed_path=gal_sed_path,
            ref_band=ref_band,
        )

    def getObjList(self, bandpass):
        """Get the object list to draw for a given bandpass.

        Parameters
        ----------
        bandpass : galsim.Bandpass
            The bandpass to use for the objects.

        Returns
        -------
        dict
            A dictionary containing the object list.

        """
        objlist = {
            "gsobject": [],
            "dx": [],
            "dy": [],
        }
        k = 0
        for i in range(len(self.gsobject_list)):
            if self._chromatic:
                gsobject = self._get_gsobject(i, bandpass)
            else:
                gsobject = self._get_gsobject(i, bandpass)
            if gsobject is None:
                continue
            objlist["gsobject"].append(gsobject)
            objlist["dx"].append(self.dx[k])
            objlist["dy"].append(self.dy[k])
            k += 1
            if k >= self.getNObjects():
                break
        return objlist

    def _init_pos(self):
        self.dx, self.dy = get_simple_pos(
            self._simu_type,
            self.img_size,
            self.rng,
            layout_kind=self.layout_kind,
            buffer=self.buffer,
            spacing=self.spacing,
            n_gal=self.n_gal,
        )
        self._n_gal = len(self.dx)

    def _get_pos(self, i, ra, dec):
        pass


def get_simple_pos(
    simu_type,
    img_size,
    rng,
    layout_kind="grid",
    buffer=0,
    spacing=5.0,
    n_gal=None,
):
    """Those functions are taken from the descwl-shear-sims package,
    refs: https://github.com/LSSTDESC/descwl-shear-sims/blob/master/descwl_shear_sims/layout/shifts.py

    Parameters
    ----------
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation,
        'imcom' for IMCOM simulation, 'sh19' for SH19 simulation. Default: 'sca'.
    img_size : int
        Image size in pixels.
    rng : np.random.RandomState
        Random number generator.
    layout_kind : str
        Layout kind, one of ['grid', 'random']. Default: 'grid'.
    buffer : int
        Buffer size in pixels. Default: 0.
    spacing : float
        Spacing between galaxies in pixels. Default: 5.0.
    n_gal : int or None
        Number of galaxies to draw. If None, will use as many as possible.
        Default: None.

    Returns
    -------
    xx : np.ndarray
        x-coordinates of the galaxies in arcsec.
    yy : np.ndarray
        y-coordinates of the galaxies in arcsec.

    """
    if simu_type == "sca" or simu_type == "sh19":
        pixel_scale = roman.pixel_scale
    elif simu_type == "imcom":
        pixel_scale = IMCOM_PIXEL_SCALE
    else:
        raise ValueError(
            f"simu_type must be one of ['sca', 'imcom', 'sh19'], got {simu_type}"
        )

    img_size = img_size
    buff_img_size = img_size - 2 * buffer
    img_size_world = img_size * pixel_scale
    buff_img_size_world = buff_img_size * pixel_scale

    buffer_world = buffer * pixel_scale

    if layout_kind not in LAYOUT_KINDS:
        raise ValueError(
            f"layout_kind must be one of {LAYOUT_KINDS}, got {layout_kind}"
        )
    layout_kind = layout_kind
    buffer = buffer
    spacing = spacing

    if layout_kind == "grid":
        n_obj_side = int(np.floor((img_size_world) / spacing))
        if n_obj_side == 0:
            n_obj_side = 1

        x = spacing * (np.arange(n_obj_side) - (n_obj_side - 1) / 2)
        msk = (x >= -(img_size_world / 2 - buffer_world)) & (
            x <= img_size_world / 2 - buffer_world
        )
        x = x[msk]

        xx, yy = np.meshgrid(x, x)
        xx = xx.flatten()
        yy = yy.flatten()

        xx += rng.uniform(-pixel_scale / 2, pixel_scale / 2.0, len(xx))
        yy += rng.uniform(-pixel_scale / 2, pixel_scale / 2.0, len(yy))

    elif layout_kind == "random":
        if n_gal is None:
            area = buff_img_size_world**2
            area /= 3600
            n_gal = int(np.floor(area * ROMAN_N_EFF))

        xx = rng.uniform(
            low=buffer_world,
            high=buff_img_size_world + buffer_world,
            size=n_gal,
        )
        yy = rng.uniform(
            low=buffer_world,
            high=buff_img_size_world + buffer_world,
            size=n_gal,
        )

        xx -= img_size_world / 2.0
        yy -= img_size_world / 2.0

    if n_gal is None and layout_kind == "grid":
        n_gal = len(xx)

    return xx[:n_gal], yy[:n_gal]


def get_g_BA(rng, sigma=0.3, size=1):
    r"""Generate a set of galaxy intrinsic ellipticities using the
    Bernstein & Armstrong (2014) equation 24.

    .. math::
        p(e) = (1 - e^2)^2  e^{-e^2 / (2 \sigma^2)} 2 \pi e

    Parameters
    ----------
    rng : np.random.RandomState
        Random number generator.
    sigma : float
        Sigma parameter of the function. Default: 0.3.
    size : int
        Number of samples to generate. Default: 1.

    Returns
    -------
    np.ndarray
        Array of galaxy intrinsic ellipticities.

    """

    def pdf(e, sigma):
        return (1 - e**2) ** 2 * np.exp(-(e**2) / (2 * sigma**2)) * 2 * np.pi * e

    samples = []
    n_attempts = 0

    # Find the maximum value of the pdf on [0, 1]
    e_vals = np.linspace(0, 1, 1000)
    pdf_vals = pdf(e_vals, sigma)
    pdf_max = np.max(pdf_vals)

    while len(samples) < size:
        e = rng.uniform(0, 1)
        u = rng.uniform(0, pdf_max)
        if u < pdf(e, sigma):
            samples.append(e)
        n_attempts += 1
    return np.array(samples)
