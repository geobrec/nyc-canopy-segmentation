"""Align the land cover raster to the NAIP grid, create image chips."""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

import config


def align_labels():
    """Resample land cover onto the NAIP pixel grid. Returns (binary, 8-class)."""
    with rasterio.open(config.NAIP) as naip:
        transform = naip.window_transform(config.AOI)

        # create a new profile for the aligned labels
        profile = naip.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            width=config.AOI.width,
            height=config.AOI.height,
            transform=transform,
            compress="lzw",
        )
        # read the landcover raster, resample to the NAIP grid, and write out a new file
        with rasterio.open(config.LANDCOVER) as lc:
            # VRT as the original file is 1.7GB
            # mode resampling is appropriate for categorical data
            with WarpedVRT(
                lc,
                crs=naip.crs,
                transform=transform,
                width=config.AOI.width,
                height=config.AOI.height,
                # mode resampling is appropriate for categorical data
                resampling=Resampling.mode,
            ) as vrt:
                landcover = vrt.read(1)
    # create a binary mask for canopy vs non-canopy
    canopy = (landcover == config.CANOPY_CLASS).astype("uint8")

    with rasterio.open(config.ALIGNED, "w", **profile) as dst:
        dst.write(canopy, 1)
    # compute the fraction of pixels that are canopy
    frac = canopy.mean()
    print(f"canopy fraction: {frac:.3f}")
    if frac < 0.02 or frac > 0.95:
        print("  ^ implausible, check the AOI actually falls inside the tile")

    return canopy, landcover


def cut_chips(landcover):
    """Cut the AOI into 256 px image/label pairs. landcover from align_labels()."""
    step = config.CHIP
    kept = skipped = 0

    with rasterio.open(config.NAIP) as naip, rasterio.open(config.ALIGNED) as lab:
        labels = lab.read(1)

        for r in range(0, config.AOI.height - step + 1, step):
            for c in range(0, config.AOI.width - step + 1, step):
                # change the window to the AOI offset, so we read the correct pixels from the NAIP file
                w = Window(config.AOI.col_off + c, config.AOI.row_off + r, step, step)
                img = naip.read(window=w)

                msk = labels[r:r + step, c:c + step]
                lc8 = landcover[r:r + step, c:c + step]

                if (img.sum(axis=0) == 0).mean() > 0.10:
                    skipped += 1
                    continue

                # row/col needed again in nb02 (split) and nb04 (mosaic)
                np.savez_compressed(
                    config.CHIPS_DIR / f"chip_{r:05d}_{c:05d}.npz",
                    img=img,
                    msk=msk,
                    lc8=lc8,
                    row=r,
                    col=c,
                )
                kept += 1

    print(f"{kept} chips written, {skipped} skipped")


def load_chips():
    out = []
    for f in sorted(config.CHIPS_DIR.glob("*.npz")):
        d = np.load(f)
        out.append(
            {
                "img": d["img"],
                "msk": d["msk"],
                "lc8": d["lc8"],
                "row": int(d["row"]),
                "col": int(d["col"]),
            }
        )
    return out