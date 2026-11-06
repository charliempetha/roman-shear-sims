import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", **{"family": "serif", "serif": ["Times"]})
matplotlib.rc("text", usetex=True)
plt.rcParams["figure.dpi"] = 500
matplotlib.rc("font", **{"family": "serif", "serif": ["Times"]})  # this line
matplotlib.rc("text", usetex=True)  # this line
plt.rcParams["figure.dpi"] = 500
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
import copy
import os
import pickle
import time
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from roman_shear_sims.sim import make_sim
from roman_shear_sims.catalog import GalaxyCatalog, SimpleGalaxyCatalog
from roman_shear_sims.psf_makers import PSFMaker
from roman_shear_sims.constant import (
    IMCOM_BLOCK_SIZE,
    IMCOM_PSF_FWHM,
    IMCOM_PIXEL_SCALE,
    SCALE,
)

import ngmix
import galsim

import metadetect
from metadetect_utils import (
    TEST_METADETECT_CONFIG,
    get_cutout,
    get_shears_metadetect,
    sh17_snr_metadetect,
)
from metacal_functions import detection_metacal, get_shears_metacal


class StrongShearTest:
    def __init__(
        self,
        seed=120597,
        noise_sig=None,
        n_gal=None,
        simu_type="imcom",
        psf_type="imcom",
        simu_size=IMCOM_BLOCK_SIZE,
        simple=False,
        hlr_range=None,
        flux_dist="log_normal",
        snr_cut=5,
    ):

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.simple = simple
        self.flux_dist = flux_dist

        self.simu_type = simu_type
        # if self.simu_type == "imcom":
        #     psf_type = self.simu_type

        if simu_type == "sh19":
            fwhm = 0.9
        elif simu_type == "imcom" and psf_type not in (
            "imcom",
            "roman",
            "airy",
            "obs_airy",
        ):
            # gauss/moffat need an explicit FWHM; use the IMCOM reference PSF size
            fwhm = IMCOM_PSF_FWHM["Y106"]  # 0.22 arcsec
        else:
            # roman, airy, and obs_airy PSF types don't use fwhm
            fwhm = None

        self.simu_size = simu_size
        self.bands = ["Y106"]  # , "J129", "H158"]
        self.noise_sig = noise_sig

        layout_kind = "grid"
        self.chromatic = False
        spacing = (
            500.0  # arcsec, larger than any image so only 1 galaxy placed at center
        )
        buff = 5.0

        if self.simple == True:
            self.gal_mag = 25
            self.gal_hlr = 0.2
            self.gal_type = "exp"
        else:
            if self.flux_dist == "uniform":
                self.flux = self.noise_sig * 1e3  # ? Why
                self.flux_range = [self.flux * 0.1, self.flux * 10]
            elif self.flux_dist == "log_normal":
                self.flux = self.noise_sig * 1e1
                self.flux_range = [self.flux * 0.001, self.flux * 10000]
        self.snr_cut = snr_cut

        self.n_epochs = 1
        self.exp_time = 107
        self.hlr_range = hlr_range

        self.galaxy_catalog = self._set_galaxy_catalog(
            layout_kind, spacing, buff, n_gal
        )

        self.psf_maker = self._set_psf_maker(psf_type, fwhm=fwhm)

    def _set_galaxy_catalog(self, layout_kind, spacing, buff, n_gal):
        if self.simple == True:
            galaxy_catalog = SimpleGalaxyCatalog(
                self.simu_size,
                self.seed,
                simu_type=self.simu_type,
                gal_type=self.gal_type,
                mag=self.gal_mag,
                hlr=self.gal_hlr,
                layout_kind=layout_kind,
                exp_time=self.exp_time,
                spacing=spacing,
                buffer=buff,
                n_gal=n_gal,
                chromatic=self.chromatic,
            )
        else:
            galaxy_catalog = GalaxyCatalog(
                self.simu_size,
                self.seed,
                simu_type=self.simu_type,
                flux_range=self.flux_range,
                flux_dist=self.flux_dist,
                hlr_range=self.hlr_range,
                layout_kind=layout_kind,
                exp_time=self.exp_time,
                spacing=spacing,
                buffer=buff,
                n_gal=n_gal,
                chromatic=self.chromatic,
            )
        return galaxy_catalog

    def _set_psf_maker(self, psf_type, fwhm=None):
        psf_maker = PSFMaker(
            psf_type=psf_type,
            fwhm=fwhm,  # Set internally based on band
            chromatic=self.chromatic,
        )
        return psf_maker

    def run_sim(self, g1_in, g2_in):
        simu_dict = make_sim(
            self.rng,
            self.galaxy_catalog,
            self.psf_maker,
            simu_type=self.simu_type,
            n_epochs=self.n_epochs,
            exp_time=self.exp_time,
            cell_size_pix=self.simu_size,
            bands=self.bands,
            g1=g1_in,
            g2=g2_in,
            chromatic=self.chromatic,
            simple_noise=True,
            noise_sigma=self.noise_sig,
            draw_method="fft",
            verbose=False,
        )
        return simu_dict

    def plot_sim(self, simu_dict, g1_in, g2_in):

        img = simu_dict["Y106"][0]["sci"][f"shear_{g1_in}_{g2_in}"]
        m = np.mean(img)
        s = np.std(img)
        plt.figure(figsize=(8, 8))
        plt.imshow(
            img,
            origin="lower",
            cmap="grey",
            vmin=max(m - 3 * s, np.min(m)),
            vmax=m + 10 * s,
        )
        plt.colorbar()
        plt.show()

    def plot_cutouts(self, obs, det_cat, img, alg="metadetect", shear="noshear"):
        plt.imshow(
            img, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(img, 100)
        )
        for i in range(len(det_cat[shear])):
            if alg == "metadetect":
                plt.plot(
                    det_cat[shear]["sx_col"],
                    det_cat[shear]["sx_row"],
                    marker="o",
                    c="r",
                    fillstyle="none",
                    markersize=8,
                    ls="",
                )
            if alg == "metacal":
                stamp, row, col, ostart_row, ostart_col, oend_row, oend_col = (
                    get_cutout(
                        obs[shear].image,
                        det_cat[shear]["x"][i],
                        det_cat[shear]["y"][i],
                        11,
                        return_bounds=True,
                    )
                )
                # plt.plot(det_cat["noshear"]['x'][i], det_cat["noshear"]['y'][i], 'ro', markersize=2)
                plt.plot(
                    [ostart_col, oend_col, oend_col, ostart_col, ostart_col],
                    [ostart_row, ostart_row, oend_row, oend_row, ostart_row],
                    "r--",
                )
        plt.show()

    def shape_measurement(
        self,
        img,
        psf,
        pixel_scale,
        psf_pixel_scale=None,
        method="metadetect",
        plot=False,
        full=False,
        return_s2n=False,
        g1_in=0.02,
    ):
        """Measure shear using metacal or metadetect."""

        if psf_pixel_scale is None:
            psf_pixel_scale = pixel_scale

        img_shape = img.shape
        psf_shape = psf.shape

        img_center = ((img_shape[1] - 1) / 2.0, (img_shape[0] - 1) / 2.0)
        psf_center = ((psf_shape[1] - 1) / 2.0, (psf_shape[0] - 1) / 2.0)

        # Ngmix handles the WCS with a Jacobian transformation
        img_jacob = ngmix.DiagonalJacobian(
            row=img_center[1], col=img_center[0], scale=pixel_scale
        )
        psf_jacob = ngmix.DiagonalJacobian(
            row=psf_center[1], col=psf_center[0], scale=psf_pixel_scale
        )

        # We first create a PSF observation
        psf_obs = ngmix.Observation(image=psf, jacobian=psf_jacob)

        # Then we create the object observation and we pass the PSF observation
        obs = ngmix.Observation(
            image=img,
            weight=np.ones(img_shape)
            / self.noise_sig**2,  # Weights are inverse variance
            jacobian=img_jacob,
            psf=psf_obs,
            bmask=np.zeros(img_shape, dtype=np.int32),
            ormask=np.zeros(img_shape, dtype=np.int32),
        )

        step_size = 0.01

        if method == "metacal":
            obs_mcal = ngmix.metacal.get_all_metacal(
                obs,
                psf="fitgauss",
                step=step_size,
                fixnoise=False,
                types=["noshear", "1p", "1m", "2p", "2m"],
                rng=self.rng,
            )
            det_cat = detection_metacal(obs_mcal)
            if plot:
                self.plot_cutouts(obs_mcal, det_cat, img, alg="metacal")
            shear_dict = get_shears_metacal(
                obs_mcal,
                det_cat,
                pixel_scale,
                step=step_size,
                full=full,
                return_s2n=return_s2n,
            )
            if return_s2n:
                shear_dict, s2n_sh17 = shear_dict
                return shear_dict, {"sh17": s2n_sh17}

        elif method == "metadetect":
            obslist = ngmix.ObsList()
            mb_obslist = ngmix.MultiBandObsList()

            obslist.append(obs)
            mb_obslist.append(obslist)

            # Configure metadetect based on physical image properties
            cfg = copy.deepcopy(TEST_METADETECT_CONFIG)
            model = "wmom"  # wmom, pgauss, am
            cfg["model"] = model
            cfg["metacal"]["types"] = ["noshear", "1p", "1m", "2p", "2m"]
            cfg["metacal"]["psf"] = "fitgauss"
            cfg["metacal"]["step"] = step_size
            psf_moments = galsim.hsm.FindAdaptiveMom(
                galsim.Image(psf, scale=psf_pixel_scale)
            )
            fwhm = 2.3548 * psf_moments.moments_sigma * psf_pixel_scale  # arcsec
            cfg["weight"]["fwhm"] = fwhm

            # cfg["meds"]["min_box_size"] = 32  # pixels
            # cfg["meds"]["max_box_size"] = 48  # pixels
            # cfg["meds"]["box_type"] = "iso_radius"
            # cfg["meds"]["rad_min"] = 4
            # cfg["meds"]["rad_fac"] = 2
            # cfg["meds"]["box_padding"] = 2  # pixels

            # # Detection threshold: may need adjustment based on noise level
            # # For high noise (e.g., noise_sig~1), lower threshold
            # # For low noise (e.g., noise_sig~0.01), higher threshold
            # cfg["sx"]["detect_thresh"] = 1.0  # sky sigma

            # print(f"\nMetadetect config:")
            # print(f"  weight FWHM: {cfg['weight']['fwhm']:.2f} arcsec")
            # print(f"  box sizes: {cfg['meds']['min_box_size']}-{cfg['meds']['max_box_size']} pix")
            # print(f"  detect_thresh: {cfg['sx']['detect_thresh']:.1f} sky sigma")

            # Run metadetect on full image
            shear_dict = metadetect.do_metadetect(
                copy.deepcopy(cfg), mb_obslist, self.rng
            )

            # Plot and measure
            if plot:
                self.plot_cutouts(None, shear_dict, img, alg="metadetect")
            result = get_shears_metadetect(
                shear_dict,
                model=model,
                full=full,
                sn_cut=self.snr_cut,
                tmin=0,
                return_s2n=return_s2n,
            )
            if return_s2n:
                result, wmom_s2n = result
                sh17_s2n = sh17_snr_metadetect(
                    shear_dict, img, psf, pixel_scale, self.noise_sig, model=model
                )
                return result, {"wmom": wmom_s2n, "sh17": sh17_s2n}
            shear_dict = result

        return shear_dict

    def get_pixel_scale(self, wcs, world_pos):
        local_wcs = wcs.local(world_pos=world_pos)
        pixel_scale = np.sqrt(local_wcs.pixelArea())
        return pixel_scale

    def get_psf_image(self, pixel_scale, band="Y106", psf_size=51):
        """Draw a clean PSF image at the image pixel scale from the psf_maker."""
        self.psf_maker.init_psf(band=band)
        psf_gsobject = self.psf_maker.get_psf()
        psf_img = psf_gsobject.drawImage(
            nx=psf_size, ny=psf_size, scale=pixel_scale, method="auto"
        )
        return psf_img.array


def bootstrap_stat(d1, d2, func, rng, nboot=500):
    dim = d1.shape[0]
    stats = []
    for _ in range(nboot):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def meas_m_c_cancel(pres, mres, g1_in=0.02):

    x = np.nanmean(pres["g1"] - mres["g1"]) / 2
    y = np.nanmean(pres["R11"] + mres["R11"]) / 2
    m = x / y / g1_in - 1

    x = np.nanmean(pres["g2"] + mres["g2"]) / 2
    y = np.nanmean(pres["R22"] + mres["R22"]) / 2
    c = x / y

    return m, c


def boostrap_m_c(pres, mres, rng, g1_in=0.02):
    m, c = meas_m_c_cancel(pres, mres, g1_in=g1_in)
    bdata = bootstrap_stat(
        pres, mres, lambda d1, d2: meas_m_c_cancel(d1, d2, g1_in=g1_in), rng, nboot=1000
    )
    merr, cerr = np.nanstd(bdata, axis=0)
    return m, merr, c, cerr


def run_single_realization(
    seed,
    n_gal,
    simu_size,
    simu_type,
    psf_type,
    simple_gal_type,
    g1_in,
    g2_in,
    noise_sig,
    method,
    hlr_range=None,
    snr_cut=5,
):
    """Run a single simulation realization matching run_single() pattern."""
    test = StrongShearTest(
        seed=seed,
        n_gal=n_gal,
        noise_sig=noise_sig,
        simu_size=simu_size,
        simu_type=simu_type,
        psf_type=psf_type,
        simple=simple_gal_type,
        hlr_range=hlr_range,
        snr_cut=snr_cut,
    )
    res = []
    s2n_dict = None  # dict of {snr_type: array} collected from the +g1 pass
    for i, g1 in enumerate([g1_in, -g1_in]):
        # Run simulation
        simu_dict = test.run_sim(g1, g2_in)

        # Get pixel scale
        wcs = simu_dict["Y106"][0]["wcs"]
        world_pos = simu_dict["Y106"][0]["cell_center_world"]
        pixel_scale = test.get_pixel_scale(wcs, world_pos)

        # Draw clean PSF at image pixel scale
        psf = test.get_psf_image(pixel_scale, band="Y106")
        psf_pixel_scale = pixel_scale

        # Extract image and PSF
        img = simu_dict["Y106"][0]["sci"][f"shear_{g1}_{g2_in}"]

        # Measure shapes
        shear_dict, s2n = test.shape_measurement(
            img,
            psf,
            pixel_scale,
            psf_pixel_scale=psf_pixel_scale,
            method=method,
            return_s2n=True,
            g1_in=g1_in,
        )

        res.append(shear_dict)
        if i == 0:
            s2n_dict = s2n  # dict like {'wmom': arr, 'sh17': arr} or {'sh17': arr}

    return res, s2n_dict


def _save_checkpoint(path, pres, mres, all_s2n, done):
    """Atomically write partial results so a crash can't corrupt the file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {"pres": pres, "mres": mres, "all_s2n": all_s2n, "done": done},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)


def _load_checkpoint(path):
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Could not read checkpoint {path} ({e}); starting fresh")
        return None


def run_sims(
    base_seed=0,
    n_trials=1,
    n_gal=1,
    g1_in=0.02,
    g2_in=0.0,
    noise_sig=1e-5,
    simu_type="imcom",
    psf_type="roman",
    simu_size=50,
    simple_gal_type=False,
    method="metadetect",
    plot_one=False,
    hlr_range=None,
    snr_cut=5,
    checkpoint_dir=None,
    checkpoint_every=1000,
    resume_from_checkpoint=False,
):
    """Run multiple simulation realizations."""
    print(checkpoint_every)
    print(f"\n{'=' * 60}")
    print(f"Using {method} for shape measurement")
    print(f"Running {n_trials} realizations")
    print(f"  n_gal = {n_gal}")
    print(f"  noise_sig = {noise_sig}")
    print(f"  g1_in = {g1_in}, g2_in = {g2_in}")
    print(f"  method = {method}")

    # Generate seeds for each trial
    rng = np.random.RandomState(seed=base_seed)
    seeds = rng.randint(low=1, high=2**29, size=n_trials)

    pres = []
    mres = []
    # all_s2n: dict mapping snr_type -> list of per-trial arrays
    all_s2n = {}

    if plot_one:
        test = StrongShearTest(
            seed=seeds[0],
            n_gal=n_gal,
            noise_sig=noise_sig,
            simu_type=simu_type,
            simu_size=simu_size,
            psf_type=psf_type,
            simple=simple_gal_type,
            hlr_range=hlr_range,
            snr_cut=snr_cut,
        )
        simu_dict = test.run_sim(g1_in, g2_in)
        test.plot_sim(simu_dict, g1_in, g2_in)

    # ---- checkpoint / resume ----
    ckpt_path = None
    done = set()
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"ckpt_{method}_noise{noise_sig}_{n_trials}_{simu_type}_{psf_type}"
            f"_g1{g1_in}_seed{base_seed}.pkl",
        )
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None and resume_from_checkpoint:
            pres, mres = ckpt["pres"], ckpt["mres"]
            all_s2n, done = ckpt["all_s2n"], ckpt["done"]
            print(f"Resuming from checkpoint: {len(done)}/{n_trials} trials done")

    tm0 = time.time()

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                run_single_realization,
                seeds[i],
                n_gal,
                simu_size,
                simu_type,
                psf_type,
                simple_gal_type,
                g1_in,
                g2_in,
                noise_sig,
                method,
                hlr_range,
                snr_cut,
            ): i
            for i in range(n_trials)
            if i not in done
        }
        n_since_save = 0
        with tqdm(
            total=n_trials, initial=len(done), desc="Running simulations"
        ) as pbar:
            for fut in as_completed(futures):
                try:
                    res, s2n_dict = fut.result()
                    pres.append(res[0])
                    mres.append(res[1])
                    if s2n_dict is not None:
                        for key, arr in s2n_dict.items():
                            if arr is not None:
                                all_s2n.setdefault(key, []).append(arr)
                    done.add(futures[fut])
                    n_since_save += 1
                except Exception:
                    pass
                finally:
                    pbar.update(1)

                if ckpt_path is not None and n_since_save >= checkpoint_every:
                    _save_checkpoint(ckpt_path, pres, mres, all_s2n, done)
                    n_since_save = 0
                    pbar.write(f"Checkpoint saved: {len(done)}/{n_trials} trials")
    if ckpt_path is not None:
        _save_checkpoint(ckpt_path, pres, mres, all_s2n, done)

    total_time = time.time() - tm0
    pres = np.concatenate(pres)
    mres = np.concatenate(mres)
    m, merr, c, cerr = boostrap_m_c(pres, mres, rng, g1_in=g1_in)

    print(f"m = {m:.6f} +/- {merr:.6f} (1-sigma)")
    print(f"c = {c:.4e} +/- {cerr:.4e} (1-sigma)")
    print(f"\n3-sigma significance:")
    print(f"  m [1e-3]: {m / 1e-3:.2f} +/- {3 * merr / 1e-3:.2f}")
    print(f"  c [1e-5]: {c / 1e-5:.2f} +/- {3 * cerr / 1e-5:.2f}")
    print(f"\nTotal time: {total_time:.1f} seconds")
    print(f"Time per trial: {total_time / n_trials:.1f} seconds")
    print(f"{'=' * 60}\n")

    return m, merr, c, cerr, all_s2n


def plot_snr_distribution(entries, sn_cut=5, method="metadetect"):
    """Plot S/N distributions comparing wmom and SH17 estimators across shear values.

    Parameters
    ----------
    entries : list of (str, dict)
        Each entry is a (label, s2n_dict) tuple where s2n_dict maps snr_type
        ('wmom', 'sh17') to a list of per-trial 1-D arrays.
        For metacal, only 'sh17' is present.
    sn_cut : float
        The S/N threshold to mark with a vertical line.
    method : str
        Pipeline name, used as the figure title.
    """
    # Collect all SNR types present across entries
    all_keys = []
    for _, s2n_dict in entries:
        for k in s2n_dict:
            if k not in all_keys:
                all_keys.append(k)

    snr_type_styles = {
        "wmom": {"ls": "-", "label_suffix": " (wmom)"},
        "sh17": {"ls": "--", "label_suffix": " (SH17)"},
    }
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12 / 2.54, 8.6 / 2.54))
    ax.set_title(f"{method}", fontsize=12)

    snr_cut_flag = False
    curve_idx = 0
    for label, s2n_dict in entries:
        color = prop_cycle[curve_idx % len(prop_cycle)]
        curve_idx += 1
        for snr_type in all_keys:
            if snr_type not in s2n_dict:
                continue
            arrs = s2n_dict[snr_type]
            s2n_combined = np.concatenate(arrs)
            s2n_combined = s2n_combined[np.isfinite(s2n_combined)]
            if (s2n_combined < sn_cut).any():
                snr_cut_flag = True
            s2n_pos = s2n_combined[s2n_combined > 0]
            if len(s2n_pos) == 0:
                continue
            bins = np.logspace(np.log10(s2n_pos.min()), np.log10(s2n_pos.max()), 15)
            style = snr_type_styles.get(
                snr_type, {"ls": ":", "label_suffix": f" ({snr_type})"}
            )
            ax.hist(
                s2n_pos,
                bins=bins,
                histtype="step",
                color=color,
                ls=style["ls"],
                lw=1.5,
                label=label + style["label_suffix"],
            )
    if snr_cut_flag:
        ax.axvline(sn_cut, color="k", ls=":", lw=0.8, label=f"SNR cut = {sn_cut}")
    ax.set_xscale("log")
    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_m_shear(fnames, results_dir):

    markers = {"metacal": "s", "metadetect": "o"}
    colors = {"metacal": "xkcd:crimson", "metadetect": "xkcd:leaf green"}

    sh17 = np.loadtxt(
        f"{results_dir}/sh17_mean.csv", dtype=np.float64, delimiter=",", skiprows=1
    )
    sh17_high = np.loadtxt(
        f"{results_dir}/sh17_high.csv", dtype=np.float64, delimiter=",", skiprows=1
    )
    sh17_low = np.loadtxt(
        f"{results_dir}/sh17_low.csv", dtype=np.float64, delimiter=",", skiprows=1
    )
    sh17_err = (sh17_high[:, 1] - sh17_low[:, 1]) / 2

    def curve(x, alpha):
        return alpha * x**2

    fig, ax = plt.subplots(figsize=(10 / 2.54, 8.6 / 2.54))

    for fname in fnames:
        method = fname.split("_")[0]
        df = pd.read_csv(f"{results_dir}{fname}")
        gamma = df["g1_in"]  # / (1 - df['g1_in'])
        x_fit = np.linspace(0, 0.11, 100)
        p_fit, cov_fit = curve_fit(
            curve, gamma, df["m"], sigma=df["m_err"], absolute_sigma=True
        )
        alpha_fit = p_fit[0]
        alpha_fit_err = np.sqrt(np.diag(cov_fit))[0]
        print(f"\nFit results for {method}:")
        # print(f"  m = {m_fit:.3f} +/- {m_fit_err:.3f} (1-sigma)")
        print(f"  alpha = {alpha_fit:.3f} +/- {alpha_fit_err:.3f} (1-sigma)")
        print(
            f"  Reduced chi-squared: {np.sum(((df['m'] * 1e3 - curve(gamma, *p_fit)) / (df['m_err'] * 1e3)) ** 2) / (len(df) - len(p_fit)):.2f}"
        )
        ax.plot(
            x_fit,
            1e3 * curve(x_fit, *p_fit),
            c=colors[method],
            ls="-",
            lw=0.5,
            label=f"$({alpha_fit:.2f} \\pm {alpha_fit_err:.2f}) \\times g_1^2$",
        )
        ax.errorbar(
            gamma,
            df["m"] * 1e3,
            yerr=df["m_err"] * 1e3,
            ls="",
            marker=markers[method],
            fillstyle="full",
            c=colors[method],
            elinewidth=0.75,
            capsize=2,
            capthick=0.75,
            markersize=5,
            label=f"{method}",
        )

    ax.errorbar(
        sh17[:, 0],
        sh17[:, 1],
        yerr=sh17_err,
        ls="",
        marker="d",
        fillstyle="full",
        c="xkcd:dark blue",
        elinewidth=0.75,
        capsize=2,
        capthick=0.75,
        markersize=2,
        label="Sheldon \\& Huff 2017",
        zorder=10,
    )
    p_fit, cov_fit = curve_fit(
        curve, sh17[:, 0], sh17[:, 1] / 1e3, sigma=sh17_err / 1e3, absolute_sigma=True
    )
    ax.plot(
        x_fit,
        1e3 * curve(x_fit, *p_fit),
        c="xkcd:dark blue",
        ls="-",
        label=f"$({p_fit[0]:.2f} \\pm {np.sqrt(np.diag(cov_fit))[0]:.2f}) \\times g_1^2$",
        lw=0.5,
    )
    ax.axhline(0, color="k", ls="--")
    ax.set_xlim(0, 0.11)
    # ax.set_ylim(-2, 12)
    ax.set_xlabel("Input shear $g_1$", fontsize=12)
    ax.set_ylabel("$m [10^{-3}]$", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_dir = (
        "/Users/cmpetha/Projects/RomanClusters/test_metadetect/StrongShearResults/"
    )

    simu_type = "sh19"  # 'sh19', 'sca', 'imcom'
    psf_type = "moffat"  # 'gauss', 'airy', 'obs_airy', 'roman', 'imcom', 'moffat'

    methods = ["metadetect"]
    fnames = []

    # simu_size: fix pixel count to 48×(0.263/0.11)=114 for all simu_types.
    simu_size = 48  # int(48 * (0.263 / 0.11))   # 114 px for all simu_types

    hlr_range = None  # use GalaxyCatalog default [0.1, 0.4] arcsec for all simu_types
    print(f"  hlr_range = {hlr_range} (default [0.1, 0.4] arcsec)")

    print(f"Running strong shear test with:")
    print(f"  simu_type = {simu_type}")
    print(f"  psf_type  = {psf_type}")

    plot_only = False

    np.random.seed(120597)  # For reproducibility of seeds across runs

    if not plot_only:
        n_trials = 2048
        noise_sig = 1e-1
        snr_cut = 2  # S/N threshold for shape measurement

        g1_in_arr = [0.02, 0.04]  # , 0.06, 0.08, 0.1]
        # Generate shared seeds once so both methods run on identical realisations
        base_seeds = np.random.randint(low=1, high=2**29, size=len(g1_in_arr))

        # method = 'metadetect'  # 'metacal' or 'metadetect'
        for method in methods:
            snr_entries = []
            m_arr, merr_arr, c_arr, cerr_arr = [], [], [], []
            for g1_in, base_seed in zip(g1_in_arr, base_seeds):
                m, merr, c, cerr, all_s2n = run_sims(
                    base_seed=base_seed,
                    n_trials=n_trials,  # Number of realizations
                    simu_type=simu_type,
                    psf_type=psf_type,
                    simu_size=simu_size,
                    n_gal=1,
                    g1_in=g1_in,
                    g2_in=0.0,
                    noise_sig=noise_sig,
                    simple_gal_type=False,
                    method=method,
                    plot_one=False,
                    hlr_range=hlr_range,
                    snr_cut=snr_cut,
                    checkpoint_dir=results_dir + "checkpoints/",
                    checkpoint_every=250,
                )
                m_arr.append(m)
                merr_arr.append(merr)
                c_arr.append(c)
                cerr_arr.append(cerr)
                if all_s2n:
                    snr_entries.append((f"$g_1={g1_in}$", all_s2n))  # all_s2n is a dict
            df = pd.DataFrame(
                {
                    "g1_in": g1_in_arr,
                    "g2_in": [0.0] * len(g1_in_arr),
                    "m": m_arr,
                    "m_err": merr_arr,
                    "c": c_arr,
                    "c_err": cerr_arr,
                    "n_trials": [n_trials] * len(g1_in_arr),
                }
            )
            fname = f"{method}_noise{noise_sig}_{n_trials}_{simu_type}_{psf_type}.csv"
            fnames.append(fname)
            df.to_csv(results_dir + fname, index=False)
            plot_m_shear([fname], results_dir=results_dir)
            if snr_entries:
                plot_snr_distribution(snr_entries, sn_cut=snr_cut, method=method)
        plot_m_shear(fnames=fnames, results_dir=results_dir)

    if plot_only:
        methods = ["metadetect", "metacal"]
        fnames = [f"{method}_noise1e-05_512_imcom_imcom.csv" for method in methods]
        plot_m_shear(fnames=fnames, results_dir=results_dir)
