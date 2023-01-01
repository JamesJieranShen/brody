from .prompt_direction import PromptDirectionStaged
from .likelihood import NLL2D

import numpy as np


class PromptDirection2D:
    coords: PromptDirectionStaged.Coordinators
    nll_long: NLL2D
    nll_short: NLL2D

    def __init__(self, coords: PromptDirectionStaged.Coordinators) -> None:
        self.coords = coords
        # X is cosAlpha, Y is tresid
        self.nll_long = NLL2D(
            coords['long'].dirtime_counts, coords['long'].dirtime_xedges,
            coords['long'].dirtime_yedges)
        self.nll_short = NLL2D(
            coords['short'].dirtime_counts,
            coords['short'].dirtime_xedges,
            coords['short'].dirtime_yedges)

    def computeTimeResidualCosAlpha(self, hypothesis, positions, times, longMask, shortMask):
        x, y, z, t0, theta, phi = hypothesis
        vpos = np.asarray([x, y, z])
        P = positions - vpos
        Pmag = np.linalg.norm(P, axis=1)
        T = times - t0
        groupVelocity = np.where(
            longMask, self.coords['long'].group_velocity, self.coords['short'].group_velocity)
        tresid = T - Pmag / groupVelocity
        vdir = np.asarray(
            [np.cos(phi) * np.sin(theta),
             np.sin(phi) * np.sin(theta),
             np.cos(theta)])
        P_normalized = P / Pmag[:, np.newaxis]
        cosalpha = np.dot(P_normalized, vdir)
        return np.array([tresid, cosalpha]).T

    def evaluate_NLL(self, hypothesis, positions, times, longMask, shortMask, channelWeights=None):
        # Guarantee proper assignment of PMT channels
        assert np.logical_xor(longMask, shortMask).all()
        tresid_cosalpha = self.computeTimeResidualCosAlpha(
            hypothesis, positions, times, longMask, shortMask)
        tresid_cosalpha_long = tresid_cosalpha[longMask]
        tresid_cosalpha_short = tresid_cosalpha[shortMask]
        nll_long = self.nll_long(tresid_cosalpha_long[:, 0], tresid_cosalpha_long[:, 1])
        nll_short = self.nll_short(tresid_cosalpha_short[:, 0], tresid_cosalpha_short[:, 1])
        if channelWeights is None:
            channelWeights = [1, 1]
        return np.sum(nll_long)*channelWeights[0] + np.sum(nll_short)*channelWeights[1]
