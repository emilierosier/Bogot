import os
import re
import zipfile
import warnings

import numpy as np
import rasterio
import geopandas as gpd
import pyogrio

from pyproj import Transformer
from shapely.geometry import Point

# ---------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------

#ELEVATION_TIF = "M_25GN2.tif"
ELEVATION_TIF = "flevoland.tif"

#PEAT_TIF = "veendikte_kaart.tiff"
PEAT_TIF = "Veendikte_Ned_Versie2022/Veendikte_Ned_Versie2022.tif"

SOIL_GPKG = "BRO_DownloadBodemkaart.gpkg"

SOIL_LAYER = "soilarea"


GROUNDWATER_GPKG = "grondwaterspiegeldiepte/BRO_Grondwaterspiegeldieptemodel_V2024-01_1.gpkg"
GROUNDWATER_NETCDF_ZIP = "bro-wdm-netcdf.zip"
GROUNDWATER_EXTRACT_DIR = "groundwater_netcdf_extracted"


# ---------------------------------------------------------
# 2. HIDE KNOWN WARNINGS
# ---------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=".*GeoPackage user_version.*",
    category=RuntimeWarning
)

warnings.filterwarnings(
    "ignore",
    message=".*More than one layer found.*",
    category=UserWarning
)


# ---------------------------------------------------------
# 3. READ A VALUE FROM A RASTER
# ---------------------------------------------------------

def sample_raster(path, latitude, longitude, extra_nodata=None):
    """Read a raster value at a GPS coordinate from a normal file path."""
    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    return sample_raster_source(path, latitude, longitude, extra_nodata)


def sample_raster_source(source, latitude, longitude, extra_nodata=None):
    """Read a raster value from a file or a GDAL subdataset URI."""
    try:
        with rasterio.open(source) as src:
            if src.crs is None:
                return None, "Raster has no coordinate reference system"

            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            )
            x, y = transformer.transform(longitude, latitude)
            row, col = src.index(x, y)

            if not (0 <= row < src.height and 0 <= col < src.width):
                return None, "Coordinate is outside this raster"

            pixel = next(src.sample([(x, y)], masked=True))[0]
            if np.ma.is_masked(pixel):
                return None, f"NoData at this coordinate (raster NoData value: {src.nodata})"

            value = float(pixel)
            if not np.isfinite(value):
                return None, "Raster returned a non-finite value"

            if extra_nodata is not None:
                for marker in extra_nodata:
                    if np.isclose(value, marker, rtol=0, atol=1e-6):
                        return None, f"Dataset-specific missing-value code ({value})"

            if src.nodata is not None and np.isclose(
                value, float(src.nodata), rtol=1e-6, atol=0.0
            ):
                return None, f"NoData at this coordinate (raster value: {value})"

            return value, None
    except Exception as error:
        return None, f"Could not read raster source: {error}"


# ---------------------------------------------------------
# 4. READ SOIL INFORMATION
# ---------------------------------------------------------

def sample_soil(path, latitude, longitude):
    """
    Find the soil polygon covering the GPS coordinate
    and retrieve its soil classifications and characteristics.
    """

    if not os.path.isfile(path):
        return None, f"Soil file not found: {path}"

    # Read the soil polygons.
    polygons = gpd.read_file(
        path,
        layer=SOIL_LAYER
    )

    if polygons.crs is None:
        return None, "Soil layer has no coordinate reference system"

    if polygons.empty:
        return None, "The soil layer contains no polygons"

    # Convert the GPS point to the soil map's CRS.
    point = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326"
    ).to_crs(polygons.crs).iloc[0]

    # Check the overall extent of the soil dataset.
    min_x, min_y, max_x, max_y = polygons.total_bounds

    if not (
        min_x <= point.x <= max_x
        and min_y <= point.y <= max_y
    ):
        return None, (
            "Coordinate is outside the overall extent "
            "of the soil dataset"
        )

    # Find the polygon that actually covers the point.
    matches = polygons[
        polygons.geometry.notna()
        & polygons.geometry.covers(point)
    ]

    if matches.empty:
        return None, (
            "No soil polygon covers this coordinate. "
            "The location may be unmapped or fall in a gap "
            "between mapped polygons."
        )

    # Use the first matching polygon.
    maparea_id = matches.iloc[0]["maparea_id"]

    # Read the soil classification tables.
    links = pyogrio.read_dataframe(
        path,
        layer="soilarea_soilunit"
    )

    units = pyogrio.read_dataframe(
        path,
        layer="soil_units"
    )

    # Find the soil units linked to this polygon.
    links = links[
        links["maparea_id"] == maparea_id
    ]

    # Always return all expected fields, even if no soil
    # classification is linked to the polygon.
    if links.empty:
        return {
            "maparea_id": maparea_id,
            "soil_classifications": [],
            "topsoil_characteristics": [],
            "bottom_layer_characteristics": []
        }, None

    # Join soil unit codes to their descriptions.
    classifications = links.merge(
        units,
        left_on="soilunit_code",
        right_on="code",
        how="left"
    )

    # Read topsoil characteristic tables.
    top_links = pyogrio.read_dataframe(
        path,
        layer="soilarea_soilunit_soilcharacteristicstoplayer"
    )

    top_codes = pyogrio.read_dataframe(
        path,
        layer="soilcharacteristics_toplayer"
    )

    # Read bottom-layer characteristic tables.
    bottom_links = pyogrio.read_dataframe(
        path,
        layer="soilarea_soilunit_soilcharacteristicsbottomlayer"
    )

    bottom_codes = pyogrio.read_dataframe(
        path,
        layer="soilcharacteristics_bottomlayer"
    )

    # Keep characteristics belonging to this polygon.
    top_links = top_links[
        top_links["maparea_id"] == maparea_id
    ].merge(
        top_codes,
        left_on="soilcharacteristics_code",
        right_on="code",
        how="left"
    )

    bottom_links = bottom_links[
        bottom_links["maparea_id"] == maparea_id
    ].merge(
        bottom_codes,
        left_on="soilcharacteristics_code",
        right_on="code",
        how="left"
    )

    # Format soil classifications.
    soil_results = []

    for _, row in classifications.iterrows():
        soil_results.append({
            "soilunit_code": row.get("soilunit_code"),
            "soil_classification": row.get("soilclassification"),
            "main_classification": row.get(
                "mainsoilclassification"
            ),
            "reference_url": row.get("url"),
            "sequence_number": row.get(
                "soilunit_sequencenumber"
            )
        })

    # Format the final result.
    result = {
        "maparea_id": maparea_id,
        "soil_classifications": soil_results,
        "topsoil_characteristics": top_links[
            ["soilcharacteristics_code", "description"]
        ].to_dict("records"),
        "bottom_layer_characteristics": bottom_links[
            ["soilcharacteristics_code", "description"]
        ].to_dict("records")
    }

    return result, None


# ---------------------------------------------------------
# 5. READ GROUNDWATER INFORMATION (GHG, GLG AND GVG)
# ---------------------------------------------------------

def normalise_name(name):
    """Convert a column/layer name to lowercase letters and numbers only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def groundwater_variable_from_name(name):
    """Identify whether a layer, column or raster subdataset is GHG, GLG or GVG."""
    name = normalise_name(name)
    # Check GVG first because it contains the letters 'gvg', but avoid
    # relying on a particular use of underscores, hyphens or capitals.
    for variable in ("ghg", "glg", "gvg"):
        if name == variable or name.startswith(variable) or name.endswith(variable):
            return variable.upper()
        if variable in name:
            return variable.upper()
    return None


def numeric_value(value):
    """Return a finite float, or None for null/non-numeric values."""
    try:
        if value is None:
            return None
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def point_rows_covering_location(gdf, point):
    """Return rows whose geometry covers the point, using a spatial index when possible."""
    try:
        indices = gdf.sindex.query(point, predicate="intersects")
        candidates = gdf.iloc[list(indices)]
        return candidates[candidates.geometry.covers(point)]
    except Exception:
        # Fallback for older GeoPandas/spatial-index versions.
        return gdf[
            gdf.geometry.notna()
            & gdf.geometry.covers(point)
        ]


def sample_groundwater_from_vector(path, latitude, longitude):
    """Try to read groundwater values from vector layers in a GeoPackage/GML."""
    values = {"GHG": None, "GLG": None, "GVG": None}
    source_names = {}
    diagnostic = []

    if not os.path.isfile(path):
        return values, f"File not found: {path}", source_names

    try:
        layers = pyogrio.list_layers(path)
    except Exception as error:
        return values, f"Could not list vector layers: {error}", source_names

    for layer_info in layers:
        layer_name = str(layer_info[0])
        try:
            gdf = gpd.read_file(path, layer=layer_name)
        except Exception as error:
            diagnostic.append(f"Layer '{layer_name}' could not be read: {error}")
            continue

        if gdf.empty or gdf.crs is None or "geometry" not in gdf.columns:
            continue

        # Convert input GPS coordinates to the layer's coordinate system.
        point = gpd.GeoSeries(
            [Point(longitude, latitude)], crs="EPSG:4326"
        ).to_crs(gdf.crs).iloc[0]
        rows = point_rows_covering_location(gdf, point)
        if rows.empty:
            continue

        # First look for columns named after the variables, e.g. ghg_mediaan.
        column_variables = {}
        for column in gdf.columns:
            variable = groundwater_variable_from_name(column)
            if variable and column != gdf.geometry.name:
                column_variables[column] = variable

        for _, row in rows.iterrows():
            for column, variable in column_variables.items():
                if values[variable] is None:
                    value = numeric_value(row.get(column))
                    if value is not None:
                        values[variable] = value
                        source_names[variable] = f"{layer_name}.{column}"

        # Some datasets store one groundwater indicator per layer, with a
        # generic field such as 'value' or 'waarde'. Only use this fallback
        # when the layer name clearly identifies GHG, GLG or GVG.
        layer_variable = groundwater_variable_from_name(layer_name)
        if layer_variable and values[layer_variable] is None:
            non_geometry_columns = [
                c for c in gdf.columns if c != gdf.geometry.name
            ]
            preferred = [
                c for c in non_geometry_columns
                if any(token in normalise_name(c) for token in
                       ("value", "waarde", "depth", "diepte", "median", "mediaan"))
            ]
            candidate_columns = preferred or non_geometry_columns
            for _, row in rows.iterrows():
                for column in candidate_columns:
                    value = numeric_value(row.get(column))
                    if value is not None:
                        values[layer_variable] = value
                        source_names[layer_variable] = f"{layer_name}.{column}"
                        break
                if values[layer_variable] is not None:
                    break

        if all(value is not None for value in values.values()):
            break

    return values, None if any(v is not None for v in values.values()) else (
        "The GeoPackage did not expose GHG/GLG/GVG values in readable vector "
        "layers. The model may be stored as raster/NetCDF data instead."
    ), source_names


def sample_groundwater_from_rasters(path, latitude, longitude):
    """Try groundwater rasters stored directly in a GeoPackage or NetCDF file."""
    values = {"GHG": None, "GLG": None, "GVG": None}
    source_names = {}

    try:
        with rasterio.open(path) as root:
            subdatasets = list(root.subdatasets or [])
            if not subdatasets:
                # A single raster is usable only if its file/layer name clearly
                # identifies which groundwater variable it contains.
                variable = groundwater_variable_from_name(path)
                if variable:
                    value, note = sample_raster_source(path, latitude, longitude)
                    if value is not None:
                        values[variable] = value
                        source_names[variable] = path
                return values, source_names
    except Exception:
        return values, source_names

    for subdataset in subdatasets:
        variable = groundwater_variable_from_name(subdataset)
        if not variable or values[variable] is not None:
            continue
        value, note = sample_raster_source(subdataset, latitude, longitude)
        if value is not None:
            values[variable] = value
            source_names[variable] = subdataset

    return values, source_names


def extract_netcdf_files(zip_path, extract_dir):
    """Safely extract NetCDF files from the downloaded ZIP archive."""
    if not os.path.isfile(zip_path):
        return [], f"File not found: {zip_path}"

    os.makedirs(extract_dir, exist_ok=True)
    root = os.path.realpath(extract_dir)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                destination = os.path.realpath(os.path.join(extract_dir, member))
                if not (destination == root or destination.startswith(root + os.sep)):
                    continue
                if member.lower().endswith((".nc", ".nc4", ".cdf")):
                    archive.extract(member, extract_dir)
    except Exception as error:
        return [], f"Could not extract NetCDF ZIP: {error}"

    files = []
    for directory, _, filenames in os.walk(extract_dir):
        for filename in filenames:
            if filename.lower().endswith((".nc", ".nc4", ".cdf")):
                files.append(os.path.join(directory, filename))
    return files, None


def sample_groundwater_from_netcdf_zip(zip_path, latitude, longitude):
    """Read GHG/GLG/GVG from raster variables inside the NetCDF ZIP, if supported by GDAL."""
    values = {"GHG": None, "GLG": None, "GVG": None}
    source_names = {}
    files, error = extract_netcdf_files(zip_path, GROUNDWATER_EXTRACT_DIR)
    if error:
        return values, error, source_names
    if not files:
        return values, "No NetCDF (.nc/.nc4/.cdf) files were found inside the ZIP.", source_names

    for netcdf_path in files:
        try:
            with rasterio.open(netcdf_path) as root:
                subdatasets = list(root.subdatasets or [])
                if not subdatasets:
                    # Some files expose their data as a normal single-band raster.
                    variable = groundwater_variable_from_name(netcdf_path)
                    if variable and values[variable] is None:
                        value, _ = sample_raster_source(netcdf_path, latitude, longitude)
                        if value is not None:
                            values[variable] = value
                            source_names[variable] = netcdf_path
                    continue
        except Exception:
            continue

        for subdataset in subdatasets:
            # Check both the NetCDF filename and variable/subdataset name.
            variable = groundwater_variable_from_name(subdataset)
            if variable is None:
                variable = groundwater_variable_from_name(netcdf_path)
            if variable is None or values[variable] is not None:
                continue
            value, _ = sample_raster_source(subdataset, latitude, longitude)
            if value is not None:
                values[variable] = value
                source_names[variable] = subdataset

    note = None if any(value is not None for value in values.values()) else (
        "The NetCDF files were found, but GDAL/rasterio could not identify "
        "GHG, GLG or GVG raster variables. The file's variable names need to be checked."
    )
    return values, note, source_names


def sample_groundwater(latitude, longitude):
    """Find GHG, GLG and GVG, trying the GeoPackage first and NetCDF ZIP second."""
    combined = {"GHG": None, "GLG": None, "GVG": None}
    sources = {}
    messages = []

    # The GeoPackage may contain vector values or raster subdatasets.
    if os.path.isfile(GROUNDWATER_GPKG):
        vector_values, vector_note, vector_sources = sample_groundwater_from_vector(
            GROUNDWATER_GPKG, latitude, longitude
        )
        for variable, value in vector_values.items():
            if value is not None:
                combined[variable] = value
                sources[variable] = vector_sources.get(variable, GROUNDWATER_GPKG)

        raster_values, raster_sources = sample_groundwater_from_rasters(
            GROUNDWATER_GPKG, latitude, longitude
        )
        for variable, value in raster_values.items():
            if combined[variable] is None and value is not None:
                combined[variable] = value
                sources[variable] = raster_sources.get(variable, GROUNDWATER_GPKG)
        if vector_note:
            messages.append(vector_note)
    else:
        messages.append(f"GeoPackage not found: {GROUNDWATER_GPKG}")

    # Fill any missing indicators using NetCDF raster variables.
    if any(value is None for value in combined.values()):
        netcdf_values, netcdf_note, netcdf_sources = sample_groundwater_from_netcdf_zip(
            GROUNDWATER_NETCDF_ZIP, latitude, longitude
        )
        for variable, value in netcdf_values.items():
            if combined[variable] is None and value is not None:
                combined[variable] = value
                sources[variable] = netcdf_sources.get(variable, GROUNDWATER_NETCDF_ZIP)
        if netcdf_note:
            messages.append(netcdf_note)

    missing = [variable for variable, value in combined.items() if value is None]
    if not missing:
        note = None
    else:
        note = (
            "Could not retrieve: " + ", ".join(missing) + ". " + " ".join(messages)
        )

    return combined, note, sources


# ---------------------------------------------------------
# 6. COMBINE ALL DATA SOURCES
# ---------------------------------------------------------

def get_location_data(latitude, longitude):

    # Ground elevation.
    elevation, elevation_note = sample_raster(
        ELEVATION_TIF,
        latitude,
        longitude
    )

    peat, peat_note = sample_raster(
        PEAT_TIF,
        latitude,
        longitude,
        extra_nodata={-1.0}
    )

    # A negative number is not a meaningful peat thickness.
    # Do not interpret it as a valid measurement without
    # checking the dataset's value conventions.
    if peat is not None and peat < 0:
        peat_note = (
            f"Unexpected negative value ({peat}). "
            "Check the dataset's value conventions."
        )
        peat = None

    # Soil information.
    soil, soil_note = sample_soil(
        SOIL_GPKG,
        latitude,
        longitude
    )

    # Groundwater depth model (GHG, GLG and GVG).
    groundwater, groundwater_note, groundwater_sources = sample_groundwater(
        latitude,
        longitude
    )

    return {
        "latitude": latitude,
        "longitude": longitude,

        "ground_elevation": elevation,
        "ground_elevation_note": elevation_note,

        "peat_thickness": peat,
        "peat_thickness_note": peat_note,

        "soil_attributes": soil,
        "soil_note": soil_note,

        "groundwater_depths": groundwater,
        "groundwater_note": groundwater_note,
        "groundwater_sources": groundwater_sources
    }


# ---------------------------------------------------------
# 7. PRINT THE RESULTS
# ---------------------------------------------------------

def print_location_data(data):

    print("\n--- Location data ---")

    print(f"Latitude: {data['latitude']}")
    print(f"Longitude: {data['longitude']}")

    # Ground elevation.
    print("\nGround elevation:")

    if data["ground_elevation"] is not None:
        print(
            f"  {data['ground_elevation']:.3f}"
        )
    else:
        print(
            f"  Unavailable: "
            f"{data['ground_elevation_note']}"
        )

    # Peat thickness.
    print("\nPeat thickness:")

    if data["peat_thickness"] is not None:
        print(
            f"  {data['peat_thickness']:.3f}"
        )
    else:
        print(
            f"  Unavailable: "
            f"{data['peat_thickness_note']}"
        )

    # Groundwater depths. The WDM values are normally in centimetres below
    # ground level (maaiveld), not relative to NAP.
    print("\nGroundwater depth model (estimated cm below ground level):")
    groundwater = data.get("groundwater_depths", {})
    groundwater_sources = data.get("groundwater_sources", {})

    for variable, description in [
        ("GHG", "Average highest groundwater level (wet season)"),
        ("GLG", "Average lowest groundwater level (dry season)"),
        ("GVG", "Average spring groundwater level")
    ]:
        value = groundwater.get(variable)
        print(f"  {variable} - {description}:")
        if value is None:
            print("    Unavailable")
        else:
            print(f"    {value:.1f} cm below ground level ({value / 100:.2f} m)")
            source = groundwater_sources.get(variable)
            if source:
                print(f"    Source: {source}")

    if data.get("groundwater_note"):
        print("  Note:", data["groundwater_note"])

    print("  These are model estimates of groundwater depth, not direct measurements.")

    # Soil map.
    print("\nSoil-map results:")

    if data["soil_attributes"] is None:
        print(f"  Unavailable: {data['soil_note']}")
        return

    soil = data["soil_attributes"]

    print("  Map area:", soil["maparea_id"])

    print("\n  Soil classifications:")

    if soil["soil_classifications"]:
        for item in soil["soil_classifications"]:

            print("   - Code:", item["soilunit_code"])

            print(
                "     Classification:",
                item["soil_classification"]
            )

            print(
                "     Main class:",
                item["main_classification"]
            )

            print(
                "     Reference:",
                item["reference_url"]
            )
    else:
        print("   No soil classifications linked.")

    print("\n  Topsoil characteristics:")

    if soil["topsoil_characteristics"]:
        for item in soil["topsoil_characteristics"]:
            print(
                "   -",
                item["soilcharacteristics_code"],
                ":",
                item["description"]
            )
    else:
        print("   No characteristics recorded.")

    print("\n  Bottom-layer characteristics:")

    if soil["bottom_layer_characteristics"]:
        for item in soil["bottom_layer_characteristics"]:
            print(
                "   -",
                item["soilcharacteristics_code"],
                ":",
                item["description"]
            )
    else:
        print("   No characteristics recorded.")


# ---------------------------------------------------------
# 8. START THE PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Enter GPS coordinates in decimal degrees.")
    print("Groundwater files are optional; the script tries the GeoPackage and then the NetCDF ZIP.")

    try:
        latitude = float(
            input("Latitude(52.40345): ").strip()
        )

        longitude = float(
            input("Longitude(5.42396): ").strip()
        )

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        data = get_location_data(
            latitude,
            longitude
        )

        print_location_data(data)

    except ValueError as error:
        print(f"\nInvalid input: {error}")

    except Exception as error:
        print(f"\nCould not retrieve the data: {error}")