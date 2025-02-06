"""
Simple functions to load and analyze ground truth data.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import os


def load_ground_truth(
    filepath: str = "data/vector/groundtruth.geojson",
) -> gpd.GeoDataFrame:
    """Load ground truth data from a GeoJSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return gpd.read_file(filepath)


def get_point_info(gdf: gpd.GeoDataFrame) -> dict:
    """Get basic information about point features."""
    points = gdf[gdf.geometry.type == "Point"]

    return {
        "total_points": len(points),
        "columns": list(points.columns),
        "bounds": points.total_bounds.tolist(),
    }


def create_feature_dataframe(
    gdf: gpd.GeoDataFrame, tiles_dir: str = "data/raster/tiles"
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with features from the GeoDataFrame and orthomosaic information.
    Matches presence/absence tiles with appropriate orthomosaic dates.

    Args:
        gdf: GeoDataFrame containing ground truth data with Presence and Datetime columns
        tiles_dir: Directory containing orthomosaic tiles

    Returns:
        DataFrame with features and orthomosaic information, filtered by presence/absence and dates
    """
    # Convert GeoDataFrame to DataFrame, excluding geometry column
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))

    # Get list of orthomosaic directories
    orthomosaics = [
        d
        for d in os.listdir(tiles_dir)
        if os.path.isdir(os.path.join(tiles_dir, d)) and not d.startswith(".")
    ]
    orthomosaics.sort()  # Sort chronologically

    # Convert orthomosaic dates to timezone-naive datetime
    ortho_dates = pd.to_datetime(
        [d.split("_")[0] for d in orthomosaics], format="%y%m%d", utc=True
    ).tz_localize(None)

    # Duplicate rows for each orthomosaic based on conditions
    df_expanded = pd.DataFrame()

    for idx, row in df.iterrows():
        # Convert to timezone-naive datetime
        row_date = pd.to_datetime(row["Datetime"]).tz_localize(None)

        if row["Presence"] == 1:
            # For presence data, only include orthomosaics after the datetime
            valid_orthos = [
                ortho
                for ortho, date in zip(orthomosaics, ortho_dates)
                if date > row_date
            ]
        else:
            # For absence data, include all orthomosaics
            valid_orthos = orthomosaics

        if valid_orthos:
            temp_rows = pd.DataFrame([row.to_dict()] * len(valid_orthos))
            temp_rows["orthomosaic"] = valid_orthos
            df_expanded = pd.concat([df_expanded, temp_rows], ignore_index=True)

    # Add tile path column based on presence/absence with zero-padded idx
    df_expanded["tile_path"] = df_expanded.apply(
        lambda x: os.path.join(
            tiles_dir,
            x["orthomosaic"],
            "presence" if x["Presence"] == 1 else "absence",
            f"{int(x['idx']):04d}.tif",  # Zero-pad idx to 4 digits
        ),
        axis=1,
    )

    return df_expanded


if __name__ == "__main__":
    try:
        # Load and analyze ground truth data
        gdf = load_ground_truth()

        # Print basic information
        info = get_point_info(gdf)
        print("\nGround Truth Information:")
        for key, value in info.items():
            print(f"{key}: {value}")

        # Create and display feature DataFrame
        df = create_feature_dataframe(gdf)
        print("\nFeature DataFrame Info:")
        print(df.info())

        # Show samples of presence and absence rows
        print("\nSample of Presence Rows (Presence == 1):")
        presence_sample = df[df["Presence"] == 1].head(3)
        print(
            presence_sample[
                ["Presence", "Datetime", "orthomosaic", "tile_path"]
            ].to_string()
        )

        print("\nSample of Absence Rows (Presence == 0):")
        absence_sample = df[df["Presence"] == 0].head(3)
        print(
            absence_sample[
                ["Presence", "Datetime", "orthomosaic", "tile_path"]
            ].to_string()
        )

        # Print some statistics
        print("\nSummary:")
        print(f"Total rows: {len(df)}")
        print(f"Presence rows: {len(df[df['Presence'] == 1])}")
        print(f"Absence rows: {len(df[df['Presence'] == 0])}")
        print(f"Unique orthomosaics: {df['orthomosaic'].nunique()}")

    except Exception as e:
        print(f"Error: {str(e)}")
