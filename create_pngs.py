
import os
import xarray as xr
import numpy as np
from numba import jit
from pathlib import Path
from PIL import Image, ImageOps
from data.cmaps import viridis


def open_dataset():
    root = Path(os.environ["PROJECT_FOLDER"])
    ds = xr.open_dataset(root/"data/met_forecast_1_0km_nordic_latest.nc")
    return ds


def create_temperature_pngs():
    ds = open_dataset()
    air_t = ds.air_temperature_2m
    matrix = convert_to_rgb(air_t, viridis)
    save_to_png(matrix)


@jit(nopython=True)
def convert_to_rgb(matrix, lut):
    t_n, r_n, c_n = matrix.shape
    air_range = np.linspace(233.14, 313.15, 254)
    result = np.zeros(matrix.shape + (3,))
    for t in range(t_n-53):
        for r in range(r_n):
            for c in range(c_n):
                ind = int(np.argmax(air_range > matrix[t, r, c]))
                result[t, r, c] = lut[ind]
        print(t)
    return result


def save_to_png(matrix):
    for t in range(matrix.shape[0]):
        img = ImageOps.flip(
                Image.fromarray(
                    matrix[t, :, :].astype(np.uint8), mode="RGB"
                    )
                )
        img.save(f"test_{t:02d}.png")


if __name__ == "__main__":
    create_temperature_pngs()

