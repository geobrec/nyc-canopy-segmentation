"""Paths and constants."""

from pathlib import Path
from rasterio.windows import Window

# create absolute paths for directories and files
ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

NAIP = RAW / "naip_2021_tile.tif"
LANDCOVER = RAW / "landcover_nyc_2021_6in.tif"
NTA_RAW = RAW / "nta.geojson"

ALIGNED = INTERIM / "labels_aligned.tif"
CHIPS_DIR = INTERIM / "chips"
PRED = PROCESSED / "canopy_pred.tif"
NTA_STATS = PROCESSED / "nta_canopy.geojson"
MODEL_PATH = ROOT / "best_model.pth"

# canopy is defined by class 1 in the landcover raster
CANOPY_CLASS = 1
CHIP = 256

# 8192 px @ 0.6 m = 4.9 km. offsets skip the nodata border on the tile edge
AOI = Window(col_off=1000, row_off=1000, width=8192, height=8192)