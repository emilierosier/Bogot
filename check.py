
import os
import rasterio
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

# Use the same file paths as in your main script
ELEVATION_TIF = "M_25GN2.tif"
PEAT_TIF = "Veendikte_Ned_Versie2022/Veendikte_Ned_Versie2022.tif"
SOIL_GPKG = "BRO_DownloadBodemkaart.gpkg"


def check_raster(name, path):
    print(f"\n--- {name} ---")

    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        return None

    try:
        with rasterio.open(path) as src:
            print(f"File: {path}")
            print(f"CRS: {src.crs}")
            print(f"Resolution: {src.res}")
            print(f"Bounds in dataset CRS: {src.bounds}")
            print(f"Bounds in GPS coordinates: ", end="")

            # Convert the raster's bounding box to latitude/longitude
            transformer = Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True
            )

            bounds = src.bounds
            corners = [
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.bottom),
                (bounds.right, bounds.top),
            ]

            gps_corners = [
                transformer.transform(x, y) for x, y in corners
            ]

            longitudes = [p[0] for p in gps_corners]
            latitudes = [p[1] for p in gps_corners]

            print(
                f"longitude {min(longitudes):.5f} to "
                f"{max(longitudes):.5f}, latitude "
                f"{min(latitudes):.5f} to {max(latitudes):.5f}"
            )

            return (
                min(longitudes), min(latitudes),
                max(longitudes), max(latitudes)
            )

    except Exception as e:
        print(f"Could not read raster: {e}")
        return None


def check_soil(path):
    print("\n--- Soil map ---")

    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        return None

    try:
        layers = gpd.list_layers(path)
        print("Available layers:")
        print(layers[["name", "geometry_type"]].to_string(index=False))

        # Check the spatial extent of each layer
        for layer_name in layers["name"]:
            try:
                gdf = gpd.read_file(
                    path, layer=layer_name, rows=1
                )
                if gdf.crs is None:
                    print(f"{layer_name}: CRS is missing")
                    continue

                # Get the full layer extent without loading all features
                import pyogrio
                info = pyogrio.read_info(path, layer=layer_name)
                extent = info.get("total_bounds")

                if extent is not None:
                    transformer = Transformer.from_crs(
                        gdf.crs, "EPSG:4326", always_xy=True
                    )
                    x_min, y_min, x_max, y_max = extent
                    corners = [
                        transformer.transform(x_min, y_min),
                        transformer.transform(x_min, y_max),
                        transformer.transform(x_max, y_min),
                        transformer.transform(x_max, y_max),
                    ]
                    lons = [p[0] for p in corners]
                    lats = [p[1] for p in corners]

                    print(
                        f"\nLayer: {layer_name}\n"
                        f"  CRS: {gdf.crs}\n"
                        f"  Longitude: {min(lons):.5f} to "
                        f"{max(lons):.5f}\n"
                        f"  Latitude: {min(lats):.5f} to "
                        f"{max(lats):.5f}"
                    )
            except Exception as e:
                print(f"{layer_name}: Could not inspect: {e}")

    except Exception as e:
        print(f"Could not read soil map: {e}")


if __name__ == "__main__":
    print("DATASET COVERAGE CHECK")
    print("This reports approximate rectangular bounds,")
    print("not the exact shape of the mapped areas.")

    elevation_bounds = check_raster(
        "Ground elevation", ELEVATION_TIF
    )
    peat_bounds = check_raster(
        "Peat thickness", PEAT_TIF
    )
    check_soil(SOIL_GPKG)

    print("\n--- Summary ---")
    print("Compare the latitude/longitude bounds above.")
    print("The three rectangles must overlap for a location")
    print("to be a candidate for testing all three maps.")
    print("A point inside a rectangle can still contain NoData.")
