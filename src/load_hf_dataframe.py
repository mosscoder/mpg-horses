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

    Args:
        gdf: GeoDataFrame containing ground truth data
        tiles_dir: Directory containing orthomosaic tiles

    Returns:
        DataFrame with features and orthomosaic information
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

    # Add orthomosaic column
    df["orthomosaic"] = orthomosaics[0]  # Default to first orthomosaic

    # Update orthomosaic based on datetime if available
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Create a mapping of dates to orthomosaics
        ortho_dates = pd.to_datetime(
            [d.split("_")[0] for d in orthomosaics], format="%y%m%d"
        )
        date_to_ortho = pd.Series(orthomosaics, index=ortho_dates)

        # Find the closest orthomosaic date for each point
        for idx in df.index:
            point_date = df.loc[idx, "Datetime"]
            closest_date = ortho_dates[abs(ortho_dates - point_date).argmin()]
            df.loc[idx, "orthomosaic"] = date_to_ortho[closest_date]

    return df


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
        print("\nFeature DataFrame Preview:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        print("\nUnique orthomosaics:", df["orthomosaic"].unique())

    except Exception as e:
        print(f"Error: {str(e)}")
