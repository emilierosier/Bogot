import os
import warnings

import numpy as np
import rasterio
import requests
import geopandas as gpd
import pyogrio
import streamlit as st

from pyproj import Transformer
from shapely.geometry import Point



#ELEVATION_TIF = "M_25GN2.tif"
ELEVATION_TIF = "flevoland.tif"

#PEAT_TIF = "veendikte_kaart.tiff"
PEAT_TIF = "Veendikte_Ned_Versie2022/Veendikte_Ned_Versie2022.tif"

SOIL_GPKG = "BRO_DownloadBodemkaart.gpkg"

GHG_TIF = "groundwaterdepth/bro-wdm-version-4.0.0_no_netcdf/ghg-mediaan.tif"
GLG_TIF = "groundwaterdepth/bro-wdm-version-4.0.0_no_netcdf/glg-mediaan.tif"
GVG_TIF = "groundwaterdepth/bro-wdm-version-4.0.0_no_netcdf/gvg-mediaan.tif"


SOIL_LAYER = "soilarea"



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



def sample_raster(path, latitude, longitude, extra_nodata=None):
    """
    Read a raster value at a GPS coordinate.

    Returns:
        value, None              if a valid value is found
        None, error_message      if data is unavailable
    """

    if not os.path.isfile(path):
        return None, f"File not found: {path}"

    with rasterio.open(path) as src:

        if src.crs is None:
            return None, "Raster has no coordinate reference system"

        # Convert GPS coordinates to the raster's CRS.
        transformer = Transformer.from_crs(
            "EPSG:4326",
            src.crs,
            always_xy=True
        )

        x, y = transformer.transform(longitude, latitude)

        # Find the pixel containing the coordinate.
        row, col = src.index(x, y)

        # Check whether the pixel exists in this raster.
        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            return None, (
                f"Coordinate is outside "
                f"{os.path.basename(path)}"
            )

        # Read the pixel with NoData masking enabled.
        pixel = next(
            src.sample([(x, y)], masked=True)
        )[0]

        if np.ma.is_masked(pixel):
            return None, (
                f"NoData at this coordinate "
                f"(raster NoData value: {src.nodata})"
            )

        value = float(pixel)

        if not np.isfinite(value):
            return None, "Raster returned an invalid numeric value"

        # Handle dataset-specific missing-value markers.
        if extra_nodata is not None:
            for marker in extra_nodata:
                if np.isclose(value, marker, rtol=0, atol=1e-6):
                    return None, (
                        f"Dataset-specific missing-value code "
                        f"({value})"
                    )

        # Check the NoData value declared by the raster.
        if src.nodata is not None:
            if np.isclose(
                value,
                float(src.nodata),
                rtol=1e-6,
                atol=0.0
            ):
                return None, (
                    f"NoData at this coordinate "
                    f"(raster value: {value})"
                )

        return value, None



def sample_soil(path, latitude, longitude):
    """
    Find the soil polygon covering the GPS coordinate
    and retrieve its soil classifications and characteristics.
    """

    if not os.path.isfile(path):
        return None, f"Soil file not found: {path}"

    polygons = gpd.read_file(
        path,
        layer=SOIL_LAYER
    )

    if polygons.crs is None:
        return None, "Soil layer has no coordinate reference system"

    if polygons.empty:
        return None, "The soil layer contains no polygons"

    point = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326"
    ).to_crs(polygons.crs).iloc[0]

    min_x, min_y, max_x, max_y = polygons.total_bounds

    if not (
        min_x <= point.x <= max_x
        and min_y <= point.y <= max_y
    ):
        return None, (
            "Coordinate is outside the overall extent "
            "of the soil dataset"
        )

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

    maparea_id = matches.iloc[0]["maparea_id"]

    links = pyogrio.read_dataframe(
        path,
        layer="soilarea_soilunit"
    )

    units = pyogrio.read_dataframe(
        path,
        layer="soil_units"
    )
    links = links[
        links["maparea_id"] == maparea_id
    ]

    if links.empty:
        return {
            "maparea_id": maparea_id,
            "soil_classifications": [],
            "topsoil_characteristics": [],
            "bottom_layer_characteristics": []
        }, None

    classifications = links.merge(
        units,
        left_on="soilunit_code",
        right_on="code",
        how="left"
    )

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



def sample_groundwater(latitude, longitude):
    """
    Read GHG, GLG and GVG values from the groundwater rasters.

    GHG = mean highest groundwater level
    GLG = mean lowest groundwater level
    GVG = mean spring groundwater level

    The values are returned in the raster's original units.
    Check raster metadata to confirm the units.
    """

    groundwater_files = {
        "GHG": GHG_TIF,
        "GLG": GLG_TIF,
        "GVG": GVG_TIF
    }

    results = {}

    for name, path in groundwater_files.items():

        value, note = sample_raster(
            path,
            latitude,
            longitude
        )

        results[name] = {
            "value": value,
            "note": note
        }

    return results



def sample_bgt_vegetation(latitude, longitude):
    """
    Retrieve BGT vegetation polygons around a GPS coordinate.
    """

    url = (
        "https://api.pdok.nl/kadaster/bgt/ogc/v1/"
        "collections/begroeidterreindeel/items"
    )


    delta_lat = 0.001
    delta_lon = 0.0015

    params = {
        "f": "json",
        "bbox": (
            f"{longitude - delta_lon},"
            f"{latitude - delta_lat},"
            f"{longitude + delta_lon},"
            f"{latitude + delta_lat}"
        ),
        "limit": 1000
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])

    if not features:
        return {
            "number_of_features": 0,
            "vegetation_types": [],
            "note": "No BGT vegetation objects found."
        }

    properties = [
        feature.get("properties", {})
        for feature in features
    ]


    print("\nBGT fields:", list(properties[0].keys()))

    possible_fields = [
        "plus_fysiekvoorkomen",
        "fysiekvoorkomen",
        "type",
        "naam"
    ]

    vegetation_types = set()

    for item in properties:
        for field in possible_fields:
            value = item.get(field)

            if value not in (None, ""):
                vegetation_types.add(str(value))

    return {
        "number_of_features": len(features),
        "vegetation_types": sorted(vegetation_types),
        "note": None
    }






def get_location_data(latitude, longitude):

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

    if peat is not None and peat < 0:
        peat_note = (
            f"Unexpected negative value ({peat}). "
            "Check the dataset's value conventions."
        )
        peat = None

    soil, soil_note = sample_soil(
        SOIL_GPKG,
        latitude,
        longitude
    )

    groundwater = sample_groundwater(
        latitude,
        longitude
    )
    bgt = sample_bgt_vegetation(
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
        
    }



def print_location_data(data):

    print("\n --- Location data ---")

    print(f"Latitude: {data['latitude']}")
    print(f"Longitude: {data['longitude']}")

    print("\n--- Ground elevation ---")

    if data["ground_elevation"] is not None:
        print(f"  {data['ground_elevation']:.3f}")
        print("  Check the elevation dataset for its vertical datum.")
    else:
        print(
            f"  Unavailable: "
            f"{data['ground_elevation_note']}"
        )

    print("\n--- Peat thickness ---")

    if data["peat_thickness"] is not None:
        print(f"  {data['peat_thickness']:.3f}")
        print("  Check the dataset metadata for units.")
    else:
        print(
            f"  Unavailable: "
            f"{data['peat_thickness_note']}"
        )

    print("\n--- Groundwater information ---")
    print("  Values are shown in the original raster units.")

    for name in ["GHG", "GLG", "GVG"]:

        item = data["groundwater"][name]
        value = item["value"]
        note = item["note"]

        print(f"\n  {name}:")

        if value is not None:
            print(f"    Value: {value:.2f}")
        else:
            print(f"    Unavailable: {note}")

    print(
        "\n  GHG: mean highest groundwater level "
        "(typically the wetter period)."
    )
    print(
        "  GLG: mean lowest groundwater level "
        "(typically the drier period)."
    )
    print(
        "  GVG: mean spring groundwater level."
    )

    # Soil map.
    print("\n--- Soil-map results ---")

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




if __name__ == "__main__":

    
# ---------------- STREAMLIT WEB INTERFACE ----------------

    st.set_page_config(
        page_title="BOGOT - Peatland Data Tool",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 BOGOT - Peatland Data Tool")
    
    st.write(
        "Enter GPS coordinates to retrieve information "
        "about elevation, peat thickness, soil and groundwater."
    )
    
    st.info(
        "Coordinates must be entered in decimal degrees. "
        "For example: 52.40345, 5.42396"
    )
    
    # Coordinate input
    with st.form("coordinate_form"):
    
        col1, col2 = st.columns(2)
    
        with col1:
            latitude = st.number_input(
                "Latitude",
                value=52.40345,
                format="%.6f"
            )
    
        with col2:
            longitude = st.number_input(
                "Longitude",
                value=5.42396,
                format="%.6f"
            )
    
        submitted = st.form_submit_button(
            "Retrieve location data",
            type="primary"
        )
    
    
    # Retrieve and display results
    if submitted:
    
        if not -90 <= latitude <= 90:
            st.error("Latitude must be between -90 and 90.")
    
        elif not -180 <= longitude <= 180:
            st.error("Longitude must be between -180 and 180.")
    
        else:
            with st.spinner("Retrieving location data..."):
    
                try:
                    data = get_location_data(
                        latitude,
                        longitude
                    )
    
                    st.success("Location data retrieved!")
    
                    st.subheader("Location")
                    st.write(f"Latitude: {latitude}")
                    st.write(f"Longitude: {longitude}")
    
                    # Elevation and peat thickness
                    st.subheader("Elevation and peat thickness")
    
                    col1, col2 = st.columns(2)
    
                    with col1:
                        if data["ground_elevation"] is not None:
                            st.metric(
                                "Ground elevation",
                                f'{data["ground_elevation"]:.3f}'
                            )
                        else:
                            st.warning(
                                data["ground_elevation_note"]
                            )
    
                    with col2:
                        if data["peat_thickness"] is not None:
                            st.metric(
                                "Peat thickness",
                                f'{data["peat_thickness"]:.3f}'
                            )
                        else:
                            st.warning(
                                data["peat_thickness_note"]
                            )
    
                    # Groundwater
                    st.subheader("Groundwater information")
                    st.caption(
                        "Values are shown in the original raster units."
                    )
    
                    groundwater = data["groundwater"]
    
                    cols = st.columns(3)
    
                    for col, name in zip(
                        cols, ["GHG", "GLG", "GVG"]
                    ):
                        item = groundwater[name]
    
                        with col:
                            if item["value"] is not None:
                                st.metric(
                                    name,
                                    f'{item["value"]:.2f}'
                                )
                            else:
                                st.warning(
                                    f'{name}: {item["note"]}'
                                )
    
                    st.caption(
                        "GHG = mean highest groundwater level | "
                        "GLG = mean lowest groundwater level | "
                        "GVG = mean spring groundwater level"
                    )
    
                    # Soil information
                    st.subheader("Soil information")
    
                    if data["soil_attributes"] is None:
                        st.warning(data["soil_note"])
    
                    else:
                        soil = data["soil_attributes"]
    
                        st.write(
                            f'Map area: {soil["maparea_id"]}'
                        )
    
                        with st.expander(
                            "Soil classifications",
                            expanded=True
                        ):
                            st.json(
                                soil["soil_classifications"]
                            )
    
                        with st.expander(
                            "Topsoil characteristics"
                        ):
                            st.json(
                                soil["topsoil_characteristics"]
                            )
    
                        with st.expander(
                            "Bottom-layer characteristics"
                        ):
                            st.json(
                                soil["bottom_layer_characteristics"]
                            )
    
                    # BGT vegetation
                    st.subheader("BGT vegetation information")
    
                    bgt = data["bgt"]
    
                    st.write(
                        f'Number of vegetation objects: '
                        f'{bgt["number_of_features"]}'
                    )
    
                    if bgt["vegetation_types"]:
                        st.write("Vegetation types:")
                        for vegetation in bgt["vegetation_types"]:
                            st.write(f"- {vegetation}")
                    else:
                        st.info(bgt["note"])
    
                except Exception as error:
                    st.error(
                        f"Could not retrieve the data: {error}"
                    )
