import pathlib
import numpy as np


class LocalStorage:

    dir = "/data/dataB/temp/localstorage/"

    @staticmethod
    def _add_npy(name):
        return name if name.endswith(".npy") else name+".npy"

    @staticmethod
    def get(name):
        name = LocalStorage._add_npy(name)
        return np.load(LocalStorage.dir+name)

    @staticmethod
    def get_or_default(name, default=None):
        name = LocalStorage._add_npy(name)
        if LocalStorage.contains(name):
            return LocalStorage.get(name)
        else:
            return default

    @staticmethod
    def set(name, value):
        name = LocalStorage._add_npy(name)
        np.save(LocalStorage.dir+name, value)

    @staticmethod
    def contains(name):
        name = LocalStorage._add_npy(name)
        return pathlib.Path(LocalStorage.dir+name).exists()


pathlib.Path(LocalStorage.dir).mkdir(parents=True, exist_ok=True)
