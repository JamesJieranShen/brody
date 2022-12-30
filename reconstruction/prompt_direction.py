import database
from copy import deepcopy
import pickle

import numpy as np
import chroma.event
from chroma.log import logger

from daq.simple_daq import simple_daq
from .likelihood import NLL1D, NLL2D, CosAlphaNLL
from .fit import fit_position_time, fit_direction_2D, fit_direction
from brody.misc_utils import long_cherenkov, long_scintillation, short_cherenkov, short_scintillation


class PromptDirectionStaged:
    class Coordinators(dict):
        def __init__(self, f: dict or str, reset=True):
            """Package multiple coordinators into one dict-like object.

                Either supply dict of coordinators or a filename of a pickled Coordinators
                object to load from.
                """
            super().__init__()

            if isinstance(f, str):
                coords = self.drain_pickle(f)
            elif isinstance(f, dict):
                coords = f
            else:
                raise TypeError("f must be a dict or a filename")

            for k, v in coords.items():
                if reset:
                    coords[k] = v.reset(v)
                self[k] = v

        def __add__(self, other):
            _self = deepcopy(self)
            for k in self:
                _self[k] += other[k]
            return _self

        def save(self, f: str):
            with open(f, "wb") as fout:
                pickle.dump(self, fout)

        @classmethod
        def load(cls, filename: str, reset: bool = True) -> 'Coordinators':
            coords = cls.drain_pickle(filename)
            if reset:
                for k in coords:
                    coords[k] = coords[k].reset(coords[k])

            return coords

        @staticmethod
        def drain_pickle(filename: str):
            with open(filename, "rb") as fin:
                coords = pickle.load(fin)
                try:
                    while True:
                        coords += pickle.load(fin)
                except EOFError:
                    pass
                return coords

        def reset(self):
            for k in self:
                self[k] = self[k].reset(self[k])

        def as_dict(self):
            return dict(self)

        def __repr__(self):
            return str(self.as_dict())

    class Coordinator:
        def __init__(
            self,
            group_velocity,
            prompt_cut=5.0,
            tresid_bins=np.linspace(-5, 100, 200),
            cosalpha_bins=np.linspace(-1, 1, 100),
            cf=1.0,
        ) -> None:
            """
            Parameters
            ----------
            group_velocity : [long_group_velocity, short_group_velocity] (mm/ns)
                Hypothesized group velocity of a photon in the target medium. Group velocity depends
                on (1) the target medium, and more importantly, (2) the photon's wavelength. It thus
                makes sense to specify different group velocities for longer and shorter wavelength photons.
            prompt_cut : float, default 5 ns
                A time residual cut in ns to remove much of the scintillation hits.
            tresid_bins : _type_, optional
                by default np.linspace(-5, 100, 200)
            cosalpha_bins : _type_, optional
                by default np.linspace(-1, 1, 100)
            cf : float, optional
                "coarse factor", or the percentage scaling to use in determining how much smaller to
                make 2D bins. By default 1.0
            """

            self.group_velocity = group_velocity  # did you forget mm/ns !?!?!?!?
            self.prompt_cut = prompt_cut
            self.tresid_bins = tresid_bins
            self.cosalpha_bins = cosalpha_bins
            self.tresid_counts, self.tresid_edges = np.histogram([], bins=self.tresid_bins)
            self.cosalpha_counts, self.cosalpha_edges = np.histogram([], bins=self.cosalpha_bins)

            # apply coarse factor for 2D histogram binning
            Ncherbins = int(cf * len(cosalpha_bins))
            self.dirtime_cosalpha_bins = np.linspace(
                min(cosalpha_bins), max(cosalpha_bins), Ncherbins)
            Ntresbins = int(cf * len(tresid_bins))
            self.dirtime_tresid_bins = np.linspace(min(tresid_bins), max(tresid_bins), Ntresbins)
            self.dirtime_counts, self.dirtime_xedges, self.dirtime_yedges = np.histogram2d(
                [], [], bins=[self.dirtime_cosalpha_bins, self.dirtime_tresid_bins])

        def __add__(self, other):
            """Add two coordinators together."""
            # check that properties are equal between the two coordinators
            for k in self.__dict__.keys():
                if 'counts' in k:
                    continue
                if np.any(self.__dict__[k] != other.__dict__[k]):
                    raise ValueError(f'Property mismatch ({k})')

            _self = deepcopy(self)  # let's not modify the original object
            _self.tresid_counts += other.tresid_counts
            _self.cosalpha_counts += other.cosalpha_counts
            _self.dirtime_counts += other.dirtime_counts
            return _self

        def digest(
            self,
            positions: np.ndarray,
            times: np.ndarray,
            true_vpos: np.ndarray,
            true_vdir: np.ndarray,
            true_vtime: np.ndarray,
            dirfit_mask: np.ndarray = None,
        ) -> None:
            """
            Digests position and time information from a singular event.

            Parameters
            ----------
            positions : np.ndarray of shape (N, 3)
                Positions of hit PMTs
            times : np.ndarray of shape (N, )
                Hit times of hit PMTs
            true_vpos : [x, y, z]
                The true event vertex position.
            true_vdir : [dx, dy, dz]
                The true event vertex direction.
            true_vtime : float
                The true event vertex time.
            dirfit_mask : np.ndarray, optional
                A boolean mask that selects all long wavelength PMT hits for direction reconstruction, by default None
            """

            P = positions - true_vpos
            D = np.sqrt(np.sum(np.square(P), axis=1))
            # D /= np.linalg.norm
            T = times - true_vtime
            tresid = T - D / self.group_velocity

            if dirfit_mask is not None:
                # cosalpha_i = ^p_i . ^v_true
                cosalpha = np.sum(P[dirfit_mask] * true_vdir, axis=1) / D[dirfit_mask]
                tresid = tresid[dirfit_mask]
            else:
                cosalpha = np.sum(P * true_vdir, axis=1) / D
                tresid = tresid

            self.tresid_counts += np.histogram(tresid, bins=self.tresid_bins)[0]
            counts_2d = np.histogram2d(cosalpha, tresid,
                                                  bins=[self.dirtime_cosalpha_bins, self.dirtime_tresid_bins])[0]
            self.dirtime_counts += counts_2d
            # apply cut to 1D cosalpha (NOTE: cut is not applied to cosalpha in the 2D histogram above)
            if self.prompt_cut is not None:
                cosalpha = cosalpha[tresid < self.prompt_cut]
            self.cosalpha_counts += np.histogram(cosalpha, bins=self.cosalpha_bins)[0]

        def digest_event(
            self,
            ev: chroma.event.Event,
            db: database.Database,
            pmt_type: str = None,
            return_info: bool = False,
        ) -> None:
            true_pos = ev.vertices[0].pos
            true_dir = ev.vertices[0].dir
            true_time = ev.vertices[0].t0

            hit_channels, hit_positions, hit_times = simple_daq(
                ev, db.det.channel_index_to_position, tts=db.theia_pmt_tts)
            if pmt_type:
                dirfit_mask = getattr(db, f"dirfit_mask_{pmt_type[0]}")[hit_channels]
            else:
                dirfit_mask = None
            self.digest(hit_positions, hit_times, true_pos, true_dir, true_time, dirfit_mask)

            if return_info:
                from ..unpack import Unpack

                unpacker = Unpack(db.det, db.group_velocity)
                unpacker.digest_event(ev)
                data = deepcopy(unpacker.dataset.as_dict())
                unpacker.close()
                return data

        def create_nlls(self):
            tresid_nll = NLL1D(self.tresid_counts, self.tresid_edges)
            cosalpha_nll = CosAlphaNLL(self.cosalpha_counts, self.cosalpha_edges)
            costresid_nll = NLL2D()
            return tresid_nll, cosalpha_nll, costresid_nll

        @classmethod
        def reset(cls, coord):
            assert coord.__class__ is cls, "To be reset, the coordinator must be of type PathFitter.Coordinator."

            new_coord = cls(coord.group_velocity, coord.prompt_cut,
                            coord.tresid_bins, coord.cosalpha_bins)

            attrs = [
                "tresid_counts",
                "tresid_edges",
                "cosalpha_counts",
                "cosalpha_edges",
                "dirtime_counts",
                "dirtime_xedges",
                "dirtime_yedges",
            ]
            for attr in attrs:
                setattr(new_coord, attr, getattr(coord, attr))

            return new_coord

    def __init__(self, coords: Coordinators, prompt_cut, group_velocity):
        self.prompt_cut = prompt_cut
        # FIXME: is this import necessary? Group velocity is already defined in the coordinator
        self.group_velocity_l, self.group_velocity_s = group_velocity
        # Note that one additional PromptDirectionStaged.Coordinator is required for each PMT type.
        # i.e., for Dichroicon studies, a long and short PMT Coordinators should be
        # created.
        self.nlls = {}
        for k, coord in coords.items():
            self.nlls[k] = {_k: _v for _k, _v in zip(
                ["tresid", "cosalpha", "dirtime"], coord.create_nlls())}

    def fit(
        self,
        positions,
        times,
        long_mask=None,
        short_mask=None,
        truth_vals=None,
        use_truth=False,
        **minimize_kwargs,
    ):
        res = {}
        if truth_vals is not None:
            x, y, z, t, dx, dy, dz = truth_vals
            true_vpos = np.asarray([x, y, z])
            true_vt = t
            true_vdir = np.asarray([dx, dy, dz])

        # we first call fit_postime with PMT positions and timings
        # using all PMTs (or just scint. up to u)
        Nscint_tot = len(positions) / short_scintillation
        if long_mask.any():
            group_velocities = np.where(long_mask, self.group_velocity_l, self.group_velocity_s)
        else:
            group_velocities = self.group_velocity_s
        tresid_nll = self.nlls['short']['tresid']
        m_xyzt = fit_position_time(
            positions,
            times,
            group_velocities,
            tresid_nll,
            Nscint=Nscint_tot,
            include_solid_angle=False,
        )
        *fit_vpos, fit_vt = m_xyzt.x
        res["pos+time"] = np.concatenate([fit_vpos, [fit_vt]])

        if use_truth:
            fit_vpos = true_vpos
            fit_vt = true_vt

        # we then use long PMT detections to reconstruct direction.
        # also, apply necessary cuts (e.g., prompt cut)
        hit_positions = positions if long_mask is None else positions[long_mask]
        hit_times = times if long_mask is None else times[long_mask]
        Ncher_long = len(hit_positions) / long_cherenkov
        P = hit_positions - fit_vpos
        D = np.sqrt(np.sum(np.square(P), axis=1))
        T = hit_times - fit_vt
        tresid = T - D / self.group_velocity_l
        prompt_mask = tresid < self.prompt_cut
        if len(hit_positions) > 0 and prompt_mask.any():

            """
            1D direction fit
            uses:
              - long cos[alpha]'s
            """
            res['nhit_long'] = len(hit_positions[prompt_mask])
            m_dir = fit_direction(
                positions=hit_positions[prompt_mask],
                vpos=fit_vpos,
                cosalpha_nll=self.nlls["long"]["cosalpha"],
                Ncher=Ncher_long,
                include_solid_angle=False,
                include_attenuation=False,
                **minimize_kwargs,
            )

            theta, phi = m_dir.x
            dvec1D = np.asarray(
                [np.cos(phi) * np.sin(theta),
                 np.sin(phi) * np.sin(theta),
                 np.cos(theta)])
            res["dir1D"] = dvec1D

            # """
            # 2D direction fit
            # uses:
            #   - 2D long cos[alpha]'s & t_resid's
            # """
            # m_dir2D = fit_direction_2D()
            # theta2D, phi2D = m_dir2D.x[:2]
            # dvec2D = np.asarray([np.cos(phi2D) * np.sin(theta2D), np.sin(phi2D) * np.sin(theta2D), np.cos(theta2D)])

            # m_xyztdir = fit_vertextime(
            #     hit_positions[prompt_mask],
            #     hit_times[prompt_mask],
            #     fit_vpos,
            #     fit_vt,  # ! USUALLY fit_t. debugging right now.
            #     self.costime_long_2D,
            #     self.group_velocity_l,
            #     truth_vals=truth_vals,
            #     ev_idx=self.ev_idx,
            #     **minimize_kwargs,
            # )

            # # note: will include a third term, fit_t, if time is floated via float_time = True in **opts
            # theta2D, phi2D = m_dir2D.x[:2]
            # dvec2D = np.asarray(
            #     [
            #         np.cos(phi2D) * np.sin(theta2D),
            #         np.sin(phi2D) * np.sin(theta2D),
            #         np.cos(theta2D),
            #     ]
            # )
        else:
            logger.warn("no prompt hits on long PMTs found. exiting direction fit.")
            res["dir1D"] = None
            # res["dir1D_old"] =
        return res
