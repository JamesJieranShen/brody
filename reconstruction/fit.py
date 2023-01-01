import numpy as np
from scipy.optimize import minimize, dual_annealing, differential_evolution
import jax
import jax.numpy as jnp


def fit_position_time(
    positions,
    times,
    group_velocity,
    tresid_nll,
    seed=None,
    both_stages=False,
    return_nll=False,
    include_solid_angle=False,
    include_attenuation=False,
    Nscint=1,  # theorized number of scintillation photons generated
    rc=20 * 25.4,
):
    """
    Runs a position time fit on a collection of hits given a speed'o'light and
    callable object to evaluate negative log likelihood values for hit time
    residuals.

    Generates a seed using the average hit position and emission time most
    consistent with this position if seed is None.

    Runs simplex to find minima region, then default quasi-Newton BFGS minimizer

    Parameters
    ----------
    positions : array_like
        Positions of hit PMTs
    times : array_like
        Hit times of hit PMTs
    group_velocity : float
        Hypothesized group velocity of a photon in the target medium.
    tresid_nll : callable
        Callable object to evaluate negative log likelihood values for a given hit time resiudal.
    seed : [x, y, z, t], optional
        Seed for the position and time fit. If None, a seed is generated using the average hit position and hit time residual.
    both_stages : bool, optional
        If true, returns fit results from both stages of the fit.
    return_nll : bool, optional
        If true, returns the nll function.
    include_solid_angle : bool, optional
        Include solid angle probabilities.
    include_attenuation : bool, optional
        Include attenuation probabilities.
    Nscint : int, optional
        Theorized number of scintillation photons generated
    rc : float, optional
        Concentrator radius in mm, by default 20 inches.

    Returns
    -----------
        A MinimizerResult for parameters (x,y,z,t).
    """

    def nll_postime(par):
        x, y, z, vt = par
        vpos = np.asarray([x, y, z])
        P = positions - vpos
        D = np.sqrt(np.sum(np.square(P), axis=1))
        T = times - vt
        tresid = T - D / group_velocity

        nll = tresid_nll(tresid)
        nllsum = np.sum(nll)
        if include_solid_angle:
            # PD = (P.T / D).T
            # todo: include directionality correction here (^d_i * ^r_i) (see notes) on p. 6. This will be
            # todo: the three-vector for the direction of the PMT dotted with PD
            D = np.where(D > 0, D, 1e-4)
            solid_angle_prob = Nscint * (1 / (4 * np.pi)) * np.pi * rc**2 / D**2
            nllsum += np.sum(-np.log(solid_angle_prob) + solid_angle_prob * -nll)

        if include_attenuation:
            raise NotImplementedError("Attenuation has yet to be implemented")

        return nllsum

    if return_nll:
        return nll_postime

    # guess pos/time seed by minimizing time residuals
    if seed is None:
        pos_guess = np.mean(positions, axis=0)
        P = positions - pos_guess
        D = np.sqrt(np.sum(np.square(P), axis=1))
        mean = tresid_nll.mean()
        t_guess = np.mean(times - D / group_velocity) - mean
        guess = np.concatenate([pos_guess, [t_guess]])
    else:
        guess = seed[:4]

    # find best minima coarsely with simplex
    m1 = minimize(nll_postime, x0=guess, method="Nelder-Mead")
    # compute proper minima
    m2 = minimize(nll_postime, x0=m1.x, method="BFGS")
    # print('stage1:',guess,m.x)

    if both_stages:
        return m1, m2
    else:
        return m2


def fit_direction(
    positions,
    vpos,
    cosalpha_nll,
    seed=None,
    method="orig",
    both_stages=False,
    return_nll=False,
    include_solid_angle=False,
    include_attenuation=False,
    Ncher=1,
    rc=20 * 25.4,
):
    """
    Runs a direction fit using a fixed position `pos` and a callable
    object to evaluate negative log likelihoods for cos(theta) values.

    Generates a seed using the average hit direction if seed is None

    Parameters
    ----------
    positions : array_like
        Positions of hit PMTs to be used in direction reconstruction. Usually, a time
        cut (t_resid < 5 ns) is applied to preferentially use mostly Cherenkov hits.
    vpos : array_like, [x,y,z]
        Fixed (often reconstructed) event vertex position.
    cosalpha_nll : callable
        Callable object to evaluate negative log likelihood values for a given cos[alpha]
        distribution.
    seed : [theta, phi], optional
        Seed for the position and time fit. If None, a seed is generated using the average
        hit position and hit time residual.
    method : str, optional
        Method to use for fitting. Options are orig, diff_evo, and dual_anneal.
    both_stages : bool, optional
        If true, returns fit results from both stages of the fit. (Only applicable for the orig method.)
    return_nll : bool, optional
        If true, returns the nll function.
    include_solid_angle : bool, optional
        Include solid angle probabilities.
    include_attenuation : bool, optional
        Include attenuation probabilities.
    Ncher : int, optional
        Theorized number of Cherenkov photons generated
    rc : float, optional
        Concentrator radius in mm, by default 20 inches.

    Returns
    -----------
        A MinimizerResult for parameters (theta, phi).
    """

    P = positions - vpos
    D = np.sqrt(np.sum(np.square(P), axis=1))
    PD = (P.T / D).T
    solid_angle_prob = Ncher * (1 / (4 * np.pi)) * np.pi * rc**2 / D**2

    def nll_dir(par):
        theta, phi = par
        dvec = np.asarray(
            [np.cos(phi) * np.sin(theta),
             np.sin(phi) * np.sin(theta),
             np.cos(theta)])
        cosalpha = np.sum(PD * dvec, axis=1)
        nll = cosalpha_nll(cosalpha)  # "g(cos[alpha])"
        nllsum = np.sum(nll)
        if include_solid_angle:
            # todo: include directionality correction here (^d_i * ^r_i) (see notes) on p. 6
            nllsum += np.sum(-np.log(solid_angle_prob) + solid_angle_prob * -nll)
        if include_attenuation:
            raise NotImplementedError("Attenuation has yet to be implemented")

        return nllsum

    if return_nll:
        return nll_dir

    # fit direction with pos/time fixed
    if seed is None:
        P = positions - vpos
        D = np.sqrt(np.sum(np.square(P), axis=1))
        avg_dir = np.mean(P.T / D, axis=1)
        avg_dir = avg_dir / np.sqrt(np.sum(np.square(avg_dir)))
        cosalpha = avg_dir[2]
        cosphi = avg_dir[0] / np.sqrt(1 - cosalpha**2)
        guess = (np.arccos(cosalpha), np.arccos(cosphi))
    else:
        guess = seed[-2:]

    if method == "orig":
        # find best minima coarsely with simplex
        m1 = minimize(nll_dir, x0=guess, method="Nelder-Mead")
        # compute proper minima
        m2 = minimize(nll_dir, x0=m1.x, method="BFGS")
        if both_stages:
            return m1, m2
        else:
            return m2
    elif method == "diff_evo":
        return differential_evolution(nll_dir, ((0, np.pi), (-np.pi, np.pi)))
    elif method == "dual_anneal":
        return dual_annealing(nll_dir, ((0, np.pi), (-np.pi, np.pi)))
    else:
        raise Exception("Method not implemented: " + method)


def fit_direction_accelerated(
    pmt_mask: jnp.ndarray,
    vpos: jnp.ndarray,
    cosalpha_nll,
    seed=None,
    method="orig",
    both_stages=False,
    return_nll=False,
    include_solid_angle=False,
    include_attenuation=False,
    Ncher=1,
    rc=20 * 25.4,
):
    """
    ! DOES NOT WORK
    Runs a direction fit using a fixed position `pos` and a callable
    object to evaluate negative log likelihoods for cos(theta) values.

    Generates a seed using the average hit direction if seed is None

    Parameters
    ----------
    positions : array_like
        Positions of hit PMTs to be used in direction reconstruction. Usually, a time
        cut (t_resid < 5 ns) is applied to preferentially use mostly Cherenkov hits.
    vpos : array_like, [x,y,z]
        Fixed (often reconstructed) event vertex position.
    cosalpha_nll : callable
        Callable object to evaluate negative log likelihood values for a given cos[alpha]
        distribution.
    seed : [theta, phi], optional
        Seed for the position and time fit. If None, a seed is generated using the average
        hit position and hit time residual.
    method : str, optional
        Method to use for fitting. Options are orig, diff_evo, and dual_anneal.
    both_stages : bool, optional
        If true, returns fit results from both stages of the fit. (Only applicable for the orig method.)
    return_nll : bool, optional
        If true, returns the nll function.
    include_solid_angle : bool, optional
        Include solid angle probabilities.
    include_attenuation : bool, optional
        Include attenuation probabilities.
    Ncher : int, optional
        Theorized number of Cherenkov photons generated
    rc : float, optional
        Concentrator radius in mm, by default 20 inches.

    Returns
    -----------
        A MinimizerResult for parameters (theta, phi).
    """

    P = positions - vpos
    D = np.sqrt(np.sum(np.square(P), axis=1))
    PD = (P.T / D).T
    solid_angle_prob = Ncher * (1 / (4 * np.pi)) * np.pi * rc**2 / D**2

    # ! TODO: JIT this function!

    def nll_dir(par):
        theta, phi = par
        dvec = np.asarray(
            [np.cos(phi) * np.sin(theta),
             np.sin(phi) * np.sin(theta),
             np.cos(theta)])
        cosalpha = np.sum(PD * dvec, axis=1)
        nll = cosalpha_nll(cosalpha)  # "g(cos[alpha])"
        nllsum = np.sum(nll)
        if include_solid_angle:
            # todo: include directionality correction here (^d_i * ^r_i) (see notes) on p. 6
            nllsum += np.sum(-np.log(solid_angle_prob) + solid_angle_prob * -nll)
        if include_attenuation:
            raise NotImplementedError("Attenuation has yet to be implemented")

        return nllsum

    if return_nll:
        return nll_dir

    # fit direction with pos/time fixed
    if seed is None:
        P = positions - vpos
        D = np.sqrt(np.sum(np.square(P), axis=1))
        avg_dir = np.mean(P.T / D, axis=1)
        avg_dir = avg_dir / np.sqrt(np.sum(np.square(avg_dir)))
        cosalpha = avg_dir[2]
        cosphi = avg_dir[0] / np.sqrt(1 - cosalpha**2)
        guess = (np.arccos(cosalpha), np.arccos(cosphi))
    else:
        guess = seed[-2:]

    if method == "orig":
        # find best minima coarsely with simplex
        m1 = minimize(nll_dir, x0=guess, method="Nelder-Mead")
        # compute proper minima
        m2 = minimize(nll_dir, x0=m1.x, method="BFGS")
        if both_stages:
            return m1, m2
        else:
            return m2
    elif method == "diff_evo":
        return differential_evolution(nll_dir, ((0, np.pi), (-np.pi, np.pi)))
    elif method == "dual_anneal":
        return dual_annealing(nll_dir, ((0, np.pi), (-np.pi, np.pi)))
    else:
        raise Exception("Method not implemented: " + method)


def fit_direction_2D(
    positions,
    vpos,
    times,
    vt,
    group_velocity,
    dirtime_nll,
    seed=None,
    # method="orig",
    both_stages=False,
    return_nll=False,
    include_solid_angle=False,
    include_attenuation=False,
    Ncher=1,
    rc=20 * 25.4,
):
    P = positions - vpos
    D = np.sqrt(np.sum(np.square(P), axis=1))
    T = times - vt
    tresid = T - D / group_velocity
    solid_angle_prob = Ncher * (1 / 4) * np.pi * rc * rc / (D * D)
    PD = (P.T / D).T
    cosalpha = np.sum(PD * dvec, axis=1)

    def nll_postime(par):
        theta, phi = par
        dvec = np.asarray(
            [np.cos(phi) * np.sin(theta),
             np.sin(phi) * np.sin(theta),
             np.cos(theta)])

        nll = dirtime_nll(tresid)
        nllsum = np.sum(nll)

        if include_solid_angle:
            PD = (P.T / D).T
            # todo: include directionality correction here (^d_i * ^r_i) (see notes) on p. 6. This will be
            # todo: the three-vector for the direction of the PMT dotted with PD
            nllsum += np.sum(-np.log(solid_angle_prob) - nll * solid_angle_prob)

        if include_attenuation:
            raise NotImplementedError("Attenuation has yet to be implemented")

    if return_nll:
        return nll_postime

    # guess pos/time seed by minimizing time residuals
    if seed is None:
        guess = np.concatenate([vpos, [vt]])
    else:
        guess = seed[:4]

    # find best minima coarsely with simplex
    m1 = minimize(nll_postime, x0=guess, method="Nelder-Mead")
    # compute proper minima
    m2 = minimize(nll_postime, x0=m1.x, method="BFGS")
    # print('stage1:',guess,m.x)

    if both_stages:
        return m1, m2
    else:
        return m2


def fit_vertextime(*args, **kwargs):
    pass
