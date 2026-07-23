from pathlib import Path

import pandas as pd
import numpy as np

from ttde.dl_routine import TensorDatasetX


__all__ = ['load_gas_dataset']


class _GAS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, root: Path):

        file = root / 'gas' / 'ethylene_CO.pickle'
        trn, val, tst = _load_data_and_clean_and_split(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def _load_data(file):
    """读取 GAS pickle。原 MAF `ethylene_CO.pickle` 由旧 pandas 序列化，
    在 pandas≥2 上会因 `pandas.core.index` / `Int64Index` 缺失而失败。
    优先读同目录现代副本 `ethylene_CO.pandas3.pkl`；否则做兼容 unpickle。
    """
    file = Path(file)
    modern = file.with_name("ethylene_CO.pandas3.pkl")
    if modern.exists():
        data = pd.read_pickle(modern)
    else:
        import sys
        import pickle
        import pandas.core.indexes.base as idxbase

        sys.modules.setdefault("pandas.core.index", idxbase)
        if not hasattr(idxbase, "Int64Index"):
            idxbase.Int64Index = pd.Index
        if not hasattr(idxbase, "Float64Index"):
            idxbase.Float64Index = pd.Index
        if not hasattr(idxbase, "UInt64Index"):
            idxbase.UInt64Index = pd.Index
        with file.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        try:
            data.to_pickle(modern)
        except Exception:
            pass
    # data = pd.read_pickle(file).sample(frac=0.25)
    # data.to_pickle(file)
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)
    return data


def _get_correlation_numbers(data):
    C = data.corr()
    A = C > 0.98
    B = A.values.sum(axis=1)
    return B


def _load_data_and_clean(file):

    data = _load_data(file)
    B = _get_correlation_numbers(data)

    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = _get_correlation_numbers(data)
    # print(data.corr())
    data = (data - data.mean()) / data.std()

    return data


def _load_data_and_clean_and_split(file):

    data = _load_data_and_clean(file).values
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data_train = data[0:-N_test]
    N_validate = int(0.1 * data_train.shape[0])
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test


def load_gas_dataset(path: Path, to_jax: bool = True):
    power = _GAS(path)

    data_train = TensorDatasetX(power.trn.x)
    data_val = TensorDatasetX(power.val.x)

    if to_jax:
        data_train = data_train.to_jax()
        data_val = data_val.to_jax()

    return data_train, data_val
