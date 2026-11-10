import numpy as np

import sep
import galsim

from metadetect_utils import get_cutout


def sh17_snr(stamp, psf_img, pixel_scale, noise_sigma):
    """Compute the SH17 matched-filter S/N for one object stamp.

    (S/N)^2 = (1/sigma^2) * sum_xy  m(x,y)^2

    where m is the PSF-convolved best-fit Gaussian model at the observed flux,
    constructed from the HSM adaptive-moments fit to the stamp.  This is the
    optimal (matched-filter) S/N for a known profile in white Gaussian noise.

    Parameters
    ----------
    stamp : 2-D ndarray
        Cutout image of the object (pixels).
    psf_img : 2-D ndarray
        PSF image at the same pixel scale.
    pixel_scale : float
        Pixel scale in arcsec/pixel (used for both stamp and PSF).
    noise_sigma : float
        RMS noise per pixel.

    Returns
    -------
    float or nan
        The SH17 S/N.  Returns nan if the HSM fit fails.
    """
    try:
        gal_img = galsim.Image(stamp, scale=pixel_scale)
        mom = galsim.hsm.FindAdaptiveMom(gal_img)

        # Build the PSF-convolved round-Gaussian model at the fitted flux.
        # sigma_obs^2 = sigma_gal^2 + sigma_psf^2  (Gaussian convolution theorem)
        psf_mom = galsim.hsm.FindAdaptiveMom(galsim.Image(psf_img, scale=pixel_scale))
        sigma_obs = mom.moments_sigma * pixel_scale          # arcsec
        sigma_psf = psf_mom.moments_sigma * pixel_scale      # arcsec
        sigma_gal = max(sigma_obs**2 - sigma_psf**2, 0.0)**0.5  # arcsec (floor at 0)
        # Use sigma_obs (the total convolved profile) as the matched filter – this
        # is the profile we actually see, so it is the optimal weight.
        model_obj = galsim.Gaussian(sigma=max(sigma_obs, sigma_psf), flux=mom.moments_amp)
        nx, ny = stamp.shape[1], stamp.shape[0]
        model_img = model_obj.drawImage(nx=nx, ny=ny, scale=pixel_scale, method='no_pixel')
        m = model_img.array
        snr2 = np.sum(m**2) / noise_sigma**2
        return float(np.sqrt(snr2))
    except Exception:
        return float('nan')

def detection_metacal(obs_mcal):
        """Detect objects in metacal shear types using SEP."""
        det_cat = {}
        for shear_type in obs_mcal:
            img = obs_mcal[shear_type].image.copy()
            weight = obs_mcal[shear_type].weight.copy()
            rms = np.sqrt(1. / np.median(weight))
            cat = sep.extract(
                img,
                err=rms,
                thresh=1.5)
            det_cat[shear_type] = cat
        return det_cat

def get_shears_metacal(obs_mcal, det_cat, pixel_scale, step=0.01, full=False, debug=False, return_s2n=False):
    """Measure shapes for metacal shear types using HSM."""
    noise_sigma = float(np.sqrt(1.0 / np.median(obs_mcal['noshear'].weight)))

    # Measure shapes for each shear type and compute statistics
    def measure_shear_type(shear_type, collect_s2n=False):
        g1_list, g2_list, s2n_list = [], [], []
        n_detected = len(det_cat[shear_type])
        n_failed = 0
        if debug:
            print(f"  [measure_shear_type] shear_type={shear_type}, n_detected={n_detected}")
        psf_img = obs_mcal[shear_type].psf.image
        psf_jacob = obs_mcal[shear_type].psf.jacobian
        psf_pixel_scale = psf_jacob.scale
        if debug:
            print(f"    PSF image shape: {psf_img.shape}, pixel_scale passed to HSM: {pixel_scale:.4f} arcsec/pix")
        if debug:
            print(f"    PSF ngmix jacobian scale: {psf_pixel_scale:.4f} arcsec/pix")
        psf_moments = galsim.hsm.FindAdaptiveMom(galsim.Image(psf_img, scale=psf_pixel_scale))
        if debug:
            print(f"    PSF adaptive moments sigma (arcsec): {psf_moments.moments_sigma * psf_pixel_scale:.4f}")
        for cat_obj in det_cat[shear_type]:
            try:
                stamp, row, col = get_cutout(obs_mcal[shear_type].image, cat_obj['x'], cat_obj['y'], 64)
                if debug:
                    gal_moments = galsim.hsm.FindAdaptiveMom(galsim.Image(stamp, scale=pixel_scale))
                    print(f"    Galaxy adaptive moments sigma (arcsec): {gal_moments.moments_sigma * pixel_scale:.4f}")
                meas_shape = galsim.hsm.EstimateShear(
                    galsim.Image(stamp, scale=pixel_scale),
                    galsim.Image(psf_img, scale=psf_pixel_scale)
                )
                s = galsim.Shear(e1=meas_shape.corrected_e1, e2=meas_shape.corrected_e2)
                g1_list.append(s.g1)
                g2_list.append(s.g2)
                if collect_s2n:
                    s2n_list.append(sh17_snr(stamp, psf_img, pixel_scale, noise_sigma))
            except Exception as e:
                n_failed += 1
                if debug:
                    print(f"    HSM exception: {e}")
                g1_list.append(np.nan)
                g2_list.append(np.nan)
                if collect_s2n:
                    s2n_list.append(float('nan'))
        n_good = np.sum(~np.isnan(g1_list))
        if n_failed > 0 or n_good == 0:
            if debug:
                print(f"    HSM failed: {n_failed}/{n_detected}, good measurements: {n_good}/{n_detected}")
        s2n_arr = np.array(s2n_list) if collect_s2n else None
        return len(g1_list), np.nanmean(g1_list), np.nanmean(g2_list), s2n_arr

    # Get statistics for all shear types; collect SH17 S/N only for noshear
    n, g1, g2, s2n_arr = measure_shear_type('noshear', collect_s2n=return_s2n)
    n_1p, g1_1p, g2_1p, _ = measure_shear_type('1p')
    n_1m, g1_1m, g2_1m, _ = measure_shear_type('1m')
    n_2p, g1_2p, g2_2p, _ = measure_shear_type('2p')
    n_2m, g1_2m, g2_2m, _ = measure_shear_type('2m')

    # Calculate shear responses
    R11 = (g1_1p - g1_1m) / (2 * step)
    R22 = (g2_2p - g2_2m) / (2 * step)

    if full == True:
        # Return structured dict matching metadetect output format
        shear_dict = {
            'n': n, 'g1': g1, 'g2': g2,
            'n_1p': n_1p, 'g1_1p': g1_1p, 'g2_1p': g2_1p,
            'n_1m': n_1m, 'g1_1m': g1_1m, 'g2_1m': g2_1m,
            'n_2p': n_2p, 'g1_2p': g1_2p, 'g2_2p': g2_2p,
            'n_2m': n_2m, 'g1_2m': g1_2m, 'g2_2m': g2_2m,
            'R11': R11, 'R22': R22
        }
        result = shear_dict
    else:
        dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
        result = np.array([(g1, g2, R11, R22)], dtype=dt)

    if return_s2n:
        return result, s2n_arr
    return result
