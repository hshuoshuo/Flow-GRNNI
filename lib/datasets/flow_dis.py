import os
import scipy.sparse.csgraph

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import disjoint_months, sample_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel
import h5py

class FlowDis(PandasDataset):
    SEED = 3210

    def __init__(self, impute_nans=False, small=False, freq='D', masked_sensors=None):
        self.random = np.random.default_rng(self.SEED)
        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = 'next'
        self.eval_mask = None
        df, dist, mask = self.load(impute_nans=impute_nans, small=small, masked_sensors=masked_sensors)
        self.dist = dist
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        super().__init__(dataframe=df, u=None, mask=mask, name='flo', freq=freq, aggr='nearest')
        # print(self._mask[77:90])
        # print(self.eval_mask[77:90])

    def load_raw(self, small=False):
        if small:
            path = os.path.join(datasets_path['flo'], 'pooled.h5')
            eval_mask = None
        else:
            path = os.path.join(datasets_path['flo'], 'discharge.h5')
            eval_mask = None

        # print(path)
        # with pd.HDFStore(path) as store:
        #     print("Available keys in HDF5 file:", store.keys())

        df = pd.DataFrame(pd.read_hdf(path, 'flo'))
        # df = pd.HDFStore(path, "r")
        # with h5py.File(path, 'r') as h5file:
            # print(list(h5file.keys()))
            # dataset = h5file['flo']
            # df = pd.DataFrame(dataset)
        direction = pd.DataFrame(pd.read_hdf(path, 'stations'))
        return df, direction, eval_mask

    def load(self, impute_nans=True, small=False, masked_sensors=None):
        # load readings and stations metadata
        df, direction, eval_mask = self.load_raw(small)
        # compute the masks
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
        # print(mask[77:90])
        # print(eval_mask)
        if eval_mask is None:
            # eval_mask = infer_mask(df, infer_from=self.infer_eval_from)
            eval_mask = sample_mask(df.shape,
                                p=0.02,
                                p_noise=0.2,
                                min_seq=12,
                                max_seq=48,
                                rng=self.random)
        # print(eval_mask[77:90])

        eval_mask = eval_mask.astype('uint8')
        # eval_mask = eval_mask.values.astype('uint8')
        if masked_sensors is not None:
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)

        self.eval_mask = eval_mask  # 1 if value is ground-truth for imputation else 0
        masking_ratio = np.mean(eval_mask)
        print(masking_ratio)
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(compute_mean(df))
            
        # # compute distances from latitude and longitude degrees
        st_coord = direction.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        return df, dist, mask
    

    def splitter(self, dataset, val_len=1., in_sample=False, window=0):
        nontest_idxs, test_idxs = disjoint_months(dataset, months=self.test_months, synch_mode='horizon')
        if in_sample:
            train_idxs = np.arange(len(dataset))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = disjoint_months(dataset, months=val_months, synch_mode='horizon')
        else:
            # take equal number of samples before each month of testing
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(dataset)
            # remove overlapping indices from training set
            ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs, synch_mode='horizon', as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:36, :36])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        return self._mask

    @property
    def training_mask(self):
        # print(self._mask[:3])
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))

    def test_interval_mask(self, dtype=bool, squeeze=True):
        m = np.in1d(self.df.index.month, self.test_months).astype(dtype)
        if squeeze:
            return m
        return m[:, None]
