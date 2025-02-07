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
import json


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
    """Read a GeoTIFF tile and encode it as bytes."""
    if not os.path.exists(tile_path):
        raise FileNotFoundError(f"Tile not found: {tile_path}")

    with rasterio.open(tile_path) as src:
        data = src.read()
        bio = BytesIO()
        np.savez_compressed(bio, data=data, **src.meta)
        bio.seek(0)
        return bio.read()


def create_feature_dataframe(
    gdf: gpd.GeoDataFrame, tiles_dir: str = "data/raster/tiles"
) -> pd.DataFrame:
    """Create a DataFrame with features from the GeoDataFrame and orthomosaic information."""
    # Convert GeoDataFrame to DataFrame
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))

    # Get and sort orthomosaics
    orthomosaics = sorted(
        [
            d
            for d in os.listdir(tiles_dir)
            if os.path.isdir(os.path.join(tiles_dir, d)) and not d.startswith(".")
        ]
    )

    # Convert dates
    ortho_dates = pd.to_datetime(
        [d.split("_")[0] for d in orthomosaics], format="%y%m%d", utc=True
    ).tz_localize(None)

    # Create rows for each valid orthomosaic
    rows = []
    for _, row in df.iterrows():
        row_date = pd.to_datetime(row["Datetime"]).tz_localize(None)
        valid_orthos = (
            [ortho for ortho, date in zip(orthomosaics, ortho_dates) if date > row_date]
            if row["Presence"] == 1
            else orthomosaics
        )

        for ortho in valid_orthos:
            new_row = row.to_dict()
            new_row["orthomosaic"] = ortho
            new_row["tile_path"] = os.path.join(
                tiles_dir,
                ortho,
                "presence" if row["Presence"] == 1 else "absence",
                f"{int(row['idx']):04d}.tif",
            )
            rows.append(new_row)

    return pd.DataFrame(rows)


def encode_all_tiles(df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """Encode all tiles in the DataFrame."""
    result_df = df.copy()
    result_df["encoded_tile"] = None
    failed_encodings = []

    for idx in tqdm(df.index, desc="Encoding tiles"):
        try:
            result_df.loc[idx, "encoded_tile"] = encode_tile(df.loc[idx, "tile_path"])
        except (FileNotFoundError, ValueError) as e:
            failed_encodings.append((df.loc[idx, "tile_path"], str(e)))

    if failed_encodings:
        print(f"\nFailed to encode {len(failed_encodings)} tiles")

    success_mask = result_df["encoded_tile"].notna()
    print(f"\nSuccessfully encoded {success_mask.sum()} out of {len(df)} tiles")
    return result_df[success_mask].copy()


def save_chunked_parquet(
    df: pd.DataFrame, output_dir: str = ENCODED_TILES_DIR, target_size_mb: int = 500
) -> None:
    """Save DataFrame to parquet files in chunks."""
    os.makedirs(output_dir, exist_ok=True)

    df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    n_chunks = int(np.ceil(df_size / target_size_mb))
    chunk_size = len(df) // n_chunks

    for i in tqdm(range(n_chunks), desc="Saving chunks"):
        chunk = df.iloc[i * chunk_size : min((i + 1) * chunk_size, len(df))]
        chunk.to_parquet(
            os.path.join(output_dir, f"tiles_chunk_{i:03d}.parquet"), index=True
        )

    # Save metadata
    metadata = {
        "n_chunks": n_chunks,
        "total_rows": len(df),
        "total_size_mb": df_size,
        "chunk_files": [f"tiles_chunk_{i:03d}.parquet" for i in range(n_chunks)],
    }

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
