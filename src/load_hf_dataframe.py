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
from tqdm import tqdm


# Constants for data directories relative to project root
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ENCODED_TILES_DIR = os.path.join(PROCESSED_DIR, "encoded_tiles")


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


def encode_all_tiles(df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """
    Encode all tiles in the DataFrame, processing in batches to manage memory.

    Args:
        df: DataFrame containing tile_path column
        batch_size: Number of tiles to process at once (default: 100)

    Returns:
        DataFrame with encoded_tile column containing byte representations
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize encoded_tile column with None
    result_df["encoded_tile"] = None

    # Calculate number of batches
    n_batches = (len(df) + batch_size - 1) // batch_size

    # Process in batches with progress bar
    failed_encodings = []
    with tqdm(total=len(df), desc="Encoding tiles") as pbar:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))

            # Process current batch
            batch_indices = df.index[start_idx:end_idx]
            for idx in batch_indices:
                tile_path = df.loc[idx, "tile_path"]
                try:
                    result_df.loc[idx, "encoded_tile"] = encode_tile(tile_path)
                except (FileNotFoundError, ValueError) as e:
                    failed_encodings.append((tile_path, str(e)))
                pbar.update(1)

    # Print summary of failures
    if failed_encodings:
        print(f"\nFailed to encode {len(failed_encodings)} tiles:")
        for path, error in failed_encodings[:10]:  # Show first 10 failures
            print(f"- {path}: {error}")
        if len(failed_encodings) > 10:
            print(f"... and {len(failed_encodings) - 10} more failures")

    # Create a mask for successfully encoded tiles
    success_mask = result_df["encoded_tile"].notna()

    print(f"\nSuccessfully encoded {success_mask.sum()} out of {len(df)} tiles")

    # Only return rows with successful encodings
    return result_df[success_mask].copy()


def save_chunked_parquet(
    df: pd.DataFrame, output_dir: str = ENCODED_TILES_DIR, target_size_mb: int = 500
) -> None:
    """
    Save DataFrame to parquet files in chunks of approximately target_size_mb.

    Following project directory structure:
    data/
    ├── raw/          # Original data
    ├── processed/    # Cleaned, transformed data (including encoded tiles)
    ├── vector/       # Vector data (GeoJSON, etc.)
    ├── raster/       # Image data (GeoTIFFs)
    └── tabular/      # CSV and other tabular data

    Args:
        df: DataFrame to save
        output_dir: Directory to save parquet files (default: data/processed/encoded_tiles)
        target_size_mb: Target size of each chunk in MB (default: 500)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate total size of DataFrame in memory
    df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Size in MB

    # Calculate number of chunks needed
    n_chunks = int(np.ceil(df_size / target_size_mb))
    chunk_size = len(df) // n_chunks

    print(f"\nSaving DataFrame (total size: {df_size:.1f} MB) in {n_chunks} chunks...")

    # Save in chunks with progress bar
    with tqdm(total=len(df), desc="Saving chunks") as pbar:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))

            # Get chunk
            chunk = df.iloc[start_idx:end_idx]

            # Generate chunk filename
            chunk_file = os.path.join(output_dir, f"tiles_chunk_{i:03d}.parquet")

            # Save chunk
            chunk.to_parquet(chunk_file, index=True)
            pbar.update(end_idx - start_idx)

    print(f"Saved {n_chunks} chunks to {output_dir}/")

    # Save metadata about the chunks
    metadata = {
        "n_chunks": n_chunks,
        "total_rows": len(df),
        "total_size_mb": df_size,
        "chunk_files": [f"tiles_chunk_{i:03d}.parquet" for i in range(n_chunks)],
    }

    # Save metadata as JSON
    import json

    with open(os.path.join(output_dir, "chunks_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


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

        # Encode all tiles
        print("\nEncoding all tiles...")
        encoded_df = encode_all_tiles(df)
        print("\nEncoded DataFrame Info:")
        print(encoded_df.info(show_counts=True))

        # Show sample of encoded tile sizes
        print("\nSample of encoded tile sizes (bytes):")
        encoded_sizes = encoded_df["encoded_tile"].apply(len)
        print(f"Mean size: {encoded_sizes.mean():.0f}")
        print(f"Min size: {encoded_sizes.min()}")
        print(f"Max size: {encoded_sizes.max()}")
        print(f"Total size: {encoded_sizes.sum() / (1024*1024*1024):.2f} GB")

        # Save encoded tiles to parquet chunks
        save_chunked_parquet(encoded_df)  # Using default output directory

    except Exception as e:
        print(f"Error: {str(e)}")
