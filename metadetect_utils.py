import numpy as np
import galsim

TEST_METADETECT_CONFIG = {
    "model": "wmom",
    "weight": {
        "fwhm": 1.2,  # arcsec
    },
    "metacal": {
        "psf": "fitgauss",
        "types": ["noshear", "1p", "1m", "2p", "2m"],
    },
    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 0.8,
        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        "deblend_cont": 0.00001,
        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 4,
        "filter_type": "conv",
        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
        ],
    },
    "meds": {
        "min_box_size": 31,
        "max_box_size": 31,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    # check for an edge hit
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}


def get_cutout(img, x, y, stamp_size, return_bounds=False):
    orow = int(y)
    ocol = int(x)
    half_box_size = stamp_size // 2
    maxrow, maxcol = img.shape

    ostart_row = orow - half_box_size + 1
    ostart_col = ocol - half_box_size + 1
    oend_row = orow + half_box_size + 2  # plus one for slices
    oend_col = ocol + half_box_size + 2

    ostart_row = max(0, ostart_row)
    ostart_col = max(0, ostart_col)
    oend_row = min(maxrow, oend_row)
    oend_col = min(maxcol, oend_col)

    cutout_row = y - ostart_row
    cutout_col = x - ostart_col

    if return_bounds:
        return (
            img[ostart_row:oend_row, ostart_col:oend_col],
            cutout_row,
            cutout_col,
            ostart_row,
            ostart_col,
            oend_row,
            oend_col,
        )
    else:
        return (
            img[ostart_row:oend_row, ostart_col:oend_col],
            cutout_row,
            cutout_col,
        )

def shear_cuts(arr, model, sn_cut=10, tmin=None, debug=False):
    """
    Apply selection cuts
    """
    if tmin is None:
        if model == "wmom":
            tmin = 1.2
        else:
            tmin = 0.5
    flags_col = f'{model}_flags'
    s2n_col   = f'{model}_s2n'
    tratio_col = f'{model}_T_ratio'

    msk_flags  = arr[flags_col] == 0
    msk_s2n    = arr[s2n_col] > sn_cut
    msk_tratio = arr[tratio_col] > tmin
    msk = msk_flags & msk_s2n & msk_tratio

    if debug or msk.sum() == 0:
        n_total = len(arr)
        print(f"  [shear_cuts debug] model={model}, sn_cut={sn_cut}, tmin={tmin}")
        print(f"    total objects     : {n_total}")
        print(f"    pass flags==0     : {msk_flags.sum()}")
        print(f"    pass s2n>{sn_cut}   : {msk_s2n.sum()}")
        print(f"    pass T_ratio>{tmin}: {msk_tratio.sum()}")
        print(f"    pass all cuts     : {msk.sum()}")
        if n_total > 0:
            print(f"    {s2n_col} range  : [{arr[s2n_col].min():.2f}, {arr[s2n_col].max():.2f}]")
            print(f"    {tratio_col} range: [{arr[tratio_col].min():.2f}, {arr[tratio_col].max():.2f}]")
            print(f"    {flags_col} unique: {np.unique(arr[flags_col])}")

    return msk

def sh17_snr_metadetect(res, img, psf_img, pixel_scale, noise_sigma, model='wmom', stamp_size=64):
    """Compute SH17 matched-filter S/N for each object in the metadetect noshear catalog.

    Uses the HSM adaptive-moments fit on each stamp to build the PSF-convolved
    Gaussian model m(x,y), then evaluates (S/N)^2 = sum(m^2) / sigma^2.

    Parameters
    ----------
    res : dict
        Output of metadetect.do_metadetect(), keyed by shear type.
    img : 2-D ndarray
        Full noshear science image.
    psf_img : 2-D ndarray
        PSF image at the same pixel scale as img.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    noise_sigma : float
        RMS noise per pixel.
    model : str
        metadetect model name (used only to build the flag mask).
    stamp_size : int
        Cutout half-size in pixels.

    Returns
    -------
    ndarray
        SH17 S/N values, one per detected object (nan for failed fits).
    """
    noshear = res['noshear']
    s2n_list = []
    psf_sigma = galsim.hsm.FindAdaptiveMom(
        galsim.Image(psf_img, scale=pixel_scale)
    ).moments_sigma * pixel_scale  # arcsec

    for row in noshear:
        if row[f'{model}_flags'] != 0:
            s2n_list.append(float('nan'))
            continue
        col_px = float(row['sx_col'])
        row_px = float(row['sx_row'])
        stamp, _, _ = get_cutout(img, col_px, row_px, stamp_size)
        try:
            mom = galsim.hsm.FindAdaptiveMom(galsim.Image(stamp, scale=pixel_scale))
            sigma_obs = mom.moments_sigma * pixel_scale
            sigma_model = max(sigma_obs, psf_sigma)  # cannot be narrower than PSF
            model_obj = galsim.Gaussian(sigma=sigma_model, flux=mom.moments_amp)
            nx, ny = stamp.shape[1], stamp.shape[0]
            m = model_obj.drawImage(nx=nx, ny=ny, scale=pixel_scale, method='no_pixel').array
            s2n_list.append(float(np.sqrt(np.sum(m**2) / noise_sigma**2)))
        except Exception:
            s2n_list.append(float('nan'))

    return np.array(s2n_list)


def get_shears_metadetect(res, model, step=0.01, full=False, sn_cut=10, tmin=None, debug=False, return_s2n=False):
    """
    Measure the shear response (matching test_shear_meas.py).
    """

    def get_shear_stats(shear_type):
        """Helper to get n, g1, g2 for a shear type."""
        arr = res[shear_type]
        if debug:
            print(f"  [get_shear_stats] shear_type={shear_type}, n_detected={len(arr)}")
        msk = shear_cuts(arr, model, sn_cut=sn_cut, tmin=tmin)
        n = len(arr[f'{model}_g'][msk])
        g1 = np.nanmean(arr[f'{model}_g'][msk, 0])
        g2 = np.nanmean(arr[f'{model}_g'][msk, 1])
        return n, g1, g2

    # Get measurements for all shear types
    n, g1, g2 = get_shear_stats('noshear')
    n_1p, g1_1p, g2_1p = get_shear_stats('1p')
    n_1m, g1_1m, g2_1m = get_shear_stats('1m')
    n_2p, g1_2p, g2_2p = get_shear_stats('2p')
    n_2m, g1_2m, g2_2m = get_shear_stats('2m')

    # Calculate shear responses
    R11 = (g1_1p - g1_1m) / (2 * step)
    R22 = (g2_2p - g2_2m) / (2 * step)

    s2n_arr = res['noshear'][f'{model}_s2n'] if return_s2n else None

    if full == False:
        dt = [
            ('g1', 'f8'),
            ('g2', 'f8'),
            ('R11', 'f8'),
            ('R22', 'f8')]
        result = np.array([(g1, g2, R11, R22)], dtype=dt)
    else:
        result = {
            'n': n, 'g1': g1, 'g2': g2,
            'n_1p': n_1p, 'g1_1p': g1_1p, 'g2_1p': g2_1p,
            'n_1m': n_1m, 'g1_1m': g1_1m, 'g2_1m': g2_1m,
            'n_2p': n_2p, 'g1_2p': g1_2p, 'g2_2p': g2_2p,
            'n_2m': n_2m, 'g1_2m': g1_2m, 'g2_2m': g2_2m,
            'R11': R11, 'R22': R22
        }

    if return_s2n:
        return result, s2n_arr
    return result