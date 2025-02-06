"""
Simple functions to load and analyze ground truth data.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import os
from io import BytesIO


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


def encode_tile(tile_path: str) -> bytes:
    """
    Read a GeoTIFF tile and encode it as bytes.

    Args:
        tile_path: Path to the GeoTIFF tile

    Returns:
        Bytes representation of the tile

    Raises:
        FileNotFoundError: If tile does not exist
        ValueError: If tile cannot be read
    """
    if not os.path.exists(tile_path):
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    try:
        with rasterio.open(tile_path) as src:
            # Read all bands
            data = src.read()

            # Get metadata
            meta = src.meta

            # Create BytesIO object to store the data
            bio = BytesIO()

            # Save arrays and metadata
            np.savez_compressed(bio, data=data, **meta)

            # Get the bytes
            bio.seek(0)
            return bio.read()

    except Exception as e:
        raise ValueError(f"Failed to read tile {tile_path}: {str(e)}")


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


def encode_sample_tiles(df: pd.DataFrame, sample_size: int = 5) -> pd.DataFrame:
    """
    Create a sample DataFrame with encoded tiles.

    Args:
        df: DataFrame containing tile_path column
        sample_size: Number of samples to encode (default: 5)

    Returns:
        DataFrame with encoded_tile column containing byte representations
    """
    # Take a small random sample
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Create a copy to avoid modifying the original
    sample_df = sample_df.copy()

    # Encode tiles
    encoded_tiles = []
    for tile_path in sample_df.tile_path:
        try:
            encoded_tiles.append(encode_tile(tile_path))
        except (FileNotFoundError, ValueError) as e:
            encoded_tiles.append(None)
            print(f"Warning: {str(e)}")

    # Add encoded tiles column
    sample_df["encoded_tile"] = encoded_tiles

    # Remove rows where encoding failed
    sample_df = sample_df.dropna(subset=["encoded_tile"])

    return sample_df


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

        # Create sample with encoded tiles
        print("\nEncoding sample tiles...")
        sample_df = encode_sample_tiles(df, sample_size=2)
        print(f"Successfully encoded {len(sample_df)} tiles")
        print("\nSample DataFrame Info:")
        print(sample_df.info())

    except Exception as e:
        print(f"Error: {str(e)}")
