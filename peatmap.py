
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.windows import from_bounds
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# -----------------------------------------
# 1. Load only the Netherlands
# -----------------------------------------

file_path = "GPM2022_GMC/GLOpeat_GPA22WGS_2cl_1x1km/peatGPA22WGS_2cl.tif"

# Approximate bounding box of the Netherlands
west = 3.0
east = 7.5
south = 50.7
north = 53.7

with rasterio.open(file_path) as src:

    # Convert geographic coordinates to raster window
    window = from_bounds(
        west, south, east, north,
        transform=src.transform
    )

    # Read only the selected area
    peat = src.read(
        1,
        window=window
    )

    bounds = rasterio.windows.bounds(
        window, src.transform
    )

    print("Netherlands raster dimensions:", peat.shape)

# -----------------------------------------
# 2. Mask NoData
# -----------------------------------------

peat = np.ma.masked_where(peat == 255, peat)

# -----------------------------------------
# 3. Define colours
# -----------------------------------------

cmap = ListedColormap([
    "#238b45",  # Peat dominated
    "#a1d99b"   # Peat in soil mosaic
])

cmap.set_bad(alpha=0)

# -----------------------------------------
# 4. Create map
# -----------------------------------------

fig = plt.figure(figsize=(10, 12))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [west, east, south, north],
    crs=ccrs.PlateCarree()
)

# Background
ax.add_feature(
    cfeature.OCEAN,
    facecolor="#eaf3f7"
)

ax.add_feature(
    cfeature.LAND,
    facecolor="#f2f0e6"
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.7,
    edgecolor="gray"
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.7
)

# -----------------------------------------
# 5. Display peatland data
# -----------------------------------------

ax.imshow(
    peat,
    extent=[
        bounds[0],
        bounds[2],
        bounds[1],
        bounds[3]
    ],
    origin="upper",
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=1,
    vmax=2,
    interpolation="nearest"
)

# -----------------------------------------
# 6. Legend and title
# -----------------------------------------

legend = [
    Patch(facecolor="#238b45", label="Peat dominated"),
    Patch(facecolor="#a1d99b", label="Peat in soil mosaic")
]

ax.legend(
    handles=legend,
    loc="lower left"
)

plt.title(
    "Peatland Distribution in the Netherlands\nGlobal Peatland Map 2.0",
    fontsize=14
)

plt.tight_layout()

# Save image
plt.savefig(
    "netherlands_peatland_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
