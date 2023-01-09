from .prompt_direction import PromptDirectionStaged
from .likelihood import NLL2D
from brody.log import logger

import numpy as np


class PromptDirection2D:
    coords: PromptDirectionStaged.Coordinators
    nll_long: NLL2D
    nll_short: NLL2D

    def __init__(self, coords: PromptDirectionStaged.Coordinators, smoothing=False) -> None:
        self.coords = coords
        # X is cosAlpha, Y is tresid
        self.nll_long = NLL2D(
            coords['long'].dirtime_counts, coords['long'].dirtime_xedges,
            coords['long'].dirtime_yedges, smoothing=smoothing)
        self.nll_short = NLL2D(
            coords['short'].dirtime_counts,
            coords['short'].dirtime_xedges,
            coords['short'].dirtime_yedges, smoothing=smoothing)

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
    
    def evaluate_NLL_xyzt(self, xyzt, dir, positions, times, longMask, shortMask, channelWeights):
        hypothesis = np.concatenate([xyzt, dir])
        return self.evaluate_NLL(hypothesis, positions, times, longMask, shortMask, channelWeights)

    def evaluate_NLL_dir(self, dir, xyzt, positions, times, longMask, shortMask, channelWeights):
        hypothesis = np.concatenate([xyzt, dir])
        return self.evaluate_NLL(hypothesis, positions, times, longMask, shortMask, channelWeights)

    def fit(self, positions, times, longMask, shortMask, channelWeights=None):
        from scipy.optimize import minimize
        # Calculate seed
        seed_pos = np.mean(positions[times < 100], axis=0)
        seed_dir = np.mean(positions[shortMask] - seed_pos, axis=0)
        seed_dir /= np.linalg.norm(seed_dir)
        seed_theta = np.arccos(seed_dir[2])
        seed_phi = np.arctan2(seed_dir[1], seed_dir[0])
        x0 = np.concatenate([seed_pos, [0, seed_theta, seed_phi]])
        result = minimize(self.evaluate_NLL, x0=x0, 
                args=(positions, times, longMask, shortMask, channelWeights),
                method='Nelder-Mead',
                options={'adaptive': True, 'maxfev': 5000})
        if not result.success:
            logger.warning('PromptDirection2D fit failed: %s', result.message)
        return result.x

    def stagedFit(self, positions, times, longMask, shortMask, promptCut=5):
        from scipy.optimize import minimize
        # Calculate seed
        seed_pos = np.mean(positions[times < 100], axis=0)
        seed_xyzt = np.concatenate([seed_pos, [0]])
        seed_dir_xyz = np.mean(positions[longMask] - seed_pos, axis=0)
        seed_dir_xyz /= np.linalg.norm(seed_dir_xyz)
        seed_theta = np.arccos(seed_dir_xyz[2])
        seed_phi = np.arctan2(seed_dir_xyz[1], seed_dir_xyz[0])
        seed_dir = np.array([seed_theta, seed_phi])
        # Fit position
        position_result = minimize(self.evaluate_NLL_xyzt, x0=seed_xyzt,
                args=(seed_dir, positions, times, longMask, shortMask, (1, 1)),
                method='Nelder-Mead',
                options={'adaptive': False, 'maxfev': 5000})
        if not position_result.success:
            logger.warning('PromptDirection2D position fit failed: %s', position_result.message)
        # Fit direction
        seed_dir_xyz = np.mean(positions[longMask] - position_result.x[:3], axis=0)
        seed_dir_xyz /= np.linalg.norm(seed_dir_xyz)
        seed_theta = np.arccos(seed_dir_xyz[2])
        seed_phi = np.arctan2(seed_dir_xyz[1], seed_dir_xyz[0])
        seed_dir = np.array([seed_theta, seed_phi])
        # Calculate time residuals to generate a prompt cut
        tresid_cosalpha = self.computeTimeResidualCosAlpha(
            np.concatenate([position_result.x, seed_dir]), positions, times, longMask, shortMask)
        if promptCut is None:
            tresid_filter = np.ones(len(tresid_cosalpha), dtype=bool)
        else:
            tresid_filter = tresid_cosalpha[:, 0] < promptCut
        direction_result = minimize(
            self.evaluate_NLL_dir, x0=seed_dir,
            args=(position_result.x, 
                  positions[tresid_filter],
                  times[tresid_filter],
                  longMask[tresid_filter],
                  shortMask[tresid_filter],
                  (1, 1)),
            method='Nelder-Mead', options={'adaptive': False, 'maxfev': 5000})
        if not direction_result.success:
            logger.warning('PromptDirection2D direction fit failed: %s', direction_result.message)
        return np.concatenate([position_result.x, direction_result.x])
        
