from typing import Callable, Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from .utils import polyform
import logging
from brody.log import logger


class LikelihoodException(Exception):
    pass


class NLL1D(object):
    """
    A callable object that evaluates the likelihood values given some binned data

    Does a few nonstandard things:
        - Forces a 'min_val' for each bin to avoid -log(0) to make low stats behave and avoid NaNs
        - Rolls off the probability at PDF boundaries to 'pull' outliers into the defined range
            (probability e^(-Ax) with x being distance from pdf boundary and A being 'pull')

    If you don't specify a min_val, this will guess one as being 100 times less than the bin
    with the most counts.

    If you don't specify a pull, this will guess one by calculating the average A using at most
    10 points on each side of the data range passing the min_val cut.
    """

    def __init__(self, counts, edges, min_val=None, pull=None):
        """
        Set pull to None to estimate exponential falloff from data (if desired)
        Set min_val high enough to avoid poisson fluctuations near edges of PDF (if desired)
        """

        pdf = counts.copy()
        # find bounds on PDF where counts are above the minimum value
        if min_val is None:
            min_val = np.max(pdf) / 100
        pdf = np.where(pdf > min_val, pdf, min_val)
        pull_bounds = np.argwhere(pdf > min_val)
        left, right = pull_bounds[0][0], pull_bounds[-1][0]
        # if (right - left) < 20:
        #     raise LikelihoodException("Too few bins remaining for a good PDF (bin finer?)")

        # apply cut from above to the PDF
        pdf = pdf[left:right]
        widths = (edges[1:] - edges[:-1])[left:right]
        # calculate norm and normalize the PDF to 1
        norm = np.sum(pdf * widths)
        pdf = pdf / norm
        self.centers = ((edges[:-1] + edges[1:]) / 2)[left:right]

        # convert to NLL
        self.nll = -np.log(pdf)
        logged_min_val = -np.log(min_val / norm)
        self.nll[self.nll > logged_min_val] = logged_min_val

        if pull is None:
            npts = min(len(self.nll) // 2, 10)
            # fmt: off
            left_pull = -np.mean((self.nll[:npts-1]-self.nll[1:npts]) /
                                 (self.centers[:npts-1]-self.centers[1:npts]))
            right_pull = np.mean(
                (self.nll[-npts: -1] - self.nll[-npts + 1:]) /
                (self.centers[-npts: -1] - self.centers[-npts + 1:]))
            self.pull = (left_pull, right_pull)
            # fmt: on
        else:
            self.pull = (pull, pull)

        self._call_fn = np.interp
        self._call_args = (self.centers, self.nll)

    def __call__(self, x, apply_pull=True):
        # this interp returns self.nll[0] and self.nll[-1] at values more or less than the centers
        # it is thus necessary to roll off probabilities at the edges to avoid degeneracies
        vec = self._call_fn(x, *self._call_args)
        if apply_pull:
            vec = self.apply_pull(x, vec)
        return vec

    def apply_pull(self, x, vec):
        """
        Roll off probabilities for x values past PDF boundaries
        """
        left_pull, right_pull = self.pull
        greater = x > self.centers[-1]
        deltax = x[greater] - self.centers[-1]
        vec[greater] = self.nll[-1] + deltax * right_pull
        lesser = x < self.centers[0]
        deltax = self.centers[0] - x[lesser]
        vec[lesser] = self.nll[0] + deltax * left_pull
        return vec

    def mean(self):
        weights = np.exp(-self.nll)
        return np.sum(self.centers * weights) / np.sum(weights)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.centers, self.nll, c='k',
                lw=0.5, label='data', zorder=-999)
        ax.plot(self.centers, self(self.centers), c='r', lw=0.5, label='fit')
        ax.set_xlabel("$x$")
        ax.set_ylabel("-log(likelihood)")
        ax.set_xlim(self.centers[0], self.centers[-1])
        ax.legend(frameon=False)
        ax.set_title(self.__class__.__name__)
        return ax


class CosAlphaNLL(NLL1D):
    """
    A callable object that evaluates the -log(likelihood) values by fitting
    Nth degree polynomials above and below the cherenkov peak of binned data
    """

    def __init__(self, counts, edges, order=5, *args, **kwargs):
        super().__init__(counts, edges, *args, **kwargs)

        cher_peak = self.centers[np.argmin(self.nll)]
        poly_right = np.zeros(order)
        poly_left = np.zeros(order)

        p, __ = curve_fit(polyform, self.centers, self.nll, p0=np.concatenate(
            [[cher_peak], poly_left, poly_right]))

        self._call_fn = polyform
        self._call_args = p


class NLL2D(object):
    cosalpha_edges: np.ndarray
    tresid_edges: np.ndarray
    nll_values: np.ndarray
    interp_fn: Callable
    pull: Tuple[float, float] # (left, right)

    def __init__(
            self, counts: np.ndarray, cosalpha_edges: np.ndarray, tresid_edges: np.ndarray,
            pull_npts=(10, 100), cosFitOrder=None, cosFitTol=9) -> None:
        # X is cosAlpha, Y is tresid
        assert counts.shape == (len(cosalpha_edges) - 1, len(tresid_edges) - 1)
        self.cosalpha_edges = cosalpha_edges
        self.tresid_edges = tresid_edges
        # Assume the counts are un-normalized, set all zeros to 1, and then normalilze
        if logger.level >= logging.WARNING:
            N_zeroBins = np.sum(counts == 0)
            if N_zeroBins > 0:
                logger.warning(f"Found {N_zeroBins} zero bins in 2D NLL, setting them to 1.")
        counts = np.where(counts == 0, 1, counts)
        counts /= np.sum(counts)
        self.nll_values = -np.log(counts / np.sum(counts))
        if cosFitOrder is not None:
            self.fit_cosalpha(cosFitOrder, cosFitTol)
        # Create interpolation function
        tresid_centers = (tresid_edges[:-1] + tresid_edges[1:]) / 2
        cosalpha_centers = (cosalpha_edges[:-1] + cosalpha_edges[1:]) / 2
        self.interp_fn = RectBivariateSpline(cosalpha_centers, tresid_centers,
                                             self.nll_values, kx=1, ky=1, s=0)

        # Calculate Pull
        # NOTE: Pull on the left can be very large, causing the NLL to blow up. Could consider adding a ceiling if this
        # becomes a problem.
        nll_mean = np.mean(self.nll_values, axis=0)
        p = np.polyfit(tresid_centers[:pull_npts[0]], nll_mean[:pull_npts[0]], 1)
        left_pull = -p[0]
        p = np.polyfit(tresid_centers[-pull_npts[1]:], nll_mean[-pull_npts[1]:], 1)
        right_pull = p[0]
        self.pull = (left_pull, right_pull)

    def __call__(self, tresid, cosAlpha):
        # NOTE: Vectorization needs to be guaranteed for this function
        assert np.all(cosAlpha >= -1) and np.all(cosAlpha <= 1), f"cosAlpha must be in [-1, 1], max is {np.max(cosAlpha)}, min is {np.min(cosAlpha)}"
        # Always assume input are two vectors of all desired calculations
        value = self.interp_fn(cosAlpha, tresid, grid=False)
        value = np.where(tresid < self.tresid_edges[0],
                         self.interp_fn(cosAlpha, self.tresid_edges[0], grid=False),
                         value)
        value = np.where(tresid > self.tresid_edges[-1],
                         self.interp_fn(cosAlpha, self.tresid_edges[-1], grid=False),
                         value)
        distance_to_edge_l = np.where(
            tresid < self.tresid_edges[0],
            self.tresid_edges[0] - tresid, 0)
        distance_to_edge_r = np.where(
            tresid > self.tresid_edges[-1],
            tresid - self.tresid_edges[-1], 0)
        value += distance_to_edge_l * self.pull[0] + distance_to_edge_r * self.pull[1]
        return value

    def fit_cosalpha(self, order, tol):
        # Fit a polynomial to the cosAlpha axis
        assert order >= 0
        if order == 0:
            return
        x_centers = self.cosalpha_edges[:-1] + np.diff(self.cosalpha_edges) / 2
        y_centers = self.tresid_edges[:-1] + np.diff(self.tresid_edges) / 2
        min_idx = np.argmin(self.nll_values)
        min_idx = np.unravel_index(min_idx, self.nll_values.shape)
        cher_peak = (x_centers[min_idx[0]])

        def fitCosDistribution(x, y, cherPeak, order, tol=9):
            def polyform(x, *p):
                peak = p[0]
                offset = p[1]
                order = (len(p) - 2) // 2
                # Enforce that the two polynomials evaluate to the same value at the cher peak to ensure continuity.
                paramsLeft = np.concatenate((p[2:order+2], [offset]))
                paramsRight = np.concatenate((p[order+2:], [offset]))
                return np.where(x < peak, np.polyval(paramsLeft, x - peak), np.polyval(paramsRight, x - peak))
            if np.min(y) < tol:
                p, _ = curve_fit(polyform, x, y, p0 = np.concatenate([[cherPeak], np.zeros(order*2 + 1)]))
                return lambda x: polyform(x, *p)
            else:
                p = np.polyfit(x, y, order)
                return lambda x: np.polyval(p, x)

        fittedNLL = np.zeros(self.nll_values.shape)
        for yidx in range(len(self.nll_values[0, :])):
            fitFunction = fitCosDistribution(x_centers, self.nll_values[:, yidx], cher_peak, order, tol=tol)
            fittedNLL[:, yidx] = fitFunction(x_centers)