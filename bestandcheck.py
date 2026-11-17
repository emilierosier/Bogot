
import warnings
import pyogrio

warnings.filterwarnings(
    "ignore",
    message=".*GeoPackage user_version.*",
    category=RuntimeWarning
)

path = "BRO_DownloadBodemkaart.gpkg"

layers_to_check = [
    "soilarea",
    "soilarea_soilunit",
    "soil_units",
    "soilarea_normalsoilprofile",
    "normalsoilprofiles",
    "soilarea_soilunit_soilcharacteristicstoplayer",
    "soilarea_soilunit_soilcharacteristicsbottomlayer",
    "soilcharacteristics_toplayer",
    "soilcharacteristics_bottomlayer",
]

for layer in layers_to_check:
    print("\n" + "=" * 65)
    print("LAYER:", layer)

    try:
        df = pyogrio.read_dataframe(path, layer=layer)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())

        # Show a few rows, including IDs and relationships.
        # Hide geometry because it is not needed for this inspection.
        if "geometry" in df.columns:
            df = df.drop(columns="geometry")

        print(df.head(5).to_string(index=False))

    except Exception as error:
        print("Could not read layer:", error)
