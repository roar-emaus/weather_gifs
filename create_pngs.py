
import os
import xarray as xr
import numpy as np
from numba import jit
from pathlib import Path
from PIL import Image, ImageOps
from data.cmaps import viridis, spectral_r


def open_dataset():
    root = Path(os.environ["PROJECT_FOLDER"])
    ds = xr.open_dataset(root/"data/met_forecast_1_0km_nordic_latest.nc")
    return ds


def create_temperature_pngs():
    ds = open_dataset()
    air_t = ds.air_temperature_2m.to_numpy()
    min_dist = 273.15 - air_t.min() if air_t.min() < 273.15 else 0.01
    max_dist = air_t.max() - 273.15 if air_t.max() > 273.15 else 0.01
    span = np.linspace(air_t.min(), air_t.max(), 254) 
    
    matrix = convert_to_rgb(air_t, span, spectral_r)
    save_to_png(matrix, name="air_temp")


@jit(nopython=True)
def convert_to_rgb(matrix, span, lut):
    t_n, r_n, c_n = matrix.shape
    #air_range = np.linspace(233.14, 313.15, 254)
    result = np.zeros(matrix.shape + (3,))
    for t in range(t_n):
        for r in range(r_n):
            for c in range(c_n):
                ind = int(np.argmax(span > matrix[t, r, c]))
                result[t, r, c] = lut[ind]
        print(t)
    return result


def save_to_png(matrix, name):
    for t in range(matrix.shape[0]):
        img = ImageOps.flip(
                Image.fromarray(
                    matrix[t, :, :].astype(np.uint8), mode="RGB"
                    )
                )
        img.save(f"{name}_{t:02d}.png")


if __name__ == "__main__":
    create_temperature_pngs()

