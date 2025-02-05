import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.enums import Resampling


def create_validation_map(
    geotiff_path, groundtruth_path, output_path=None, downsample_factor=8
):
    """
    Create a validation map showing ground truth points overlaid on drone imagery.

    Args:
        geotiff_path (str): Path to the drone survey GeoTIFF
        groundtruth_path (str): Path to the ground truth GeoJSON file
        output_path (str, optional): Path to save the output figure. If None, displays the figure.
        downsample_factor (int): Factor by which to downsample the raster. Higher values = smaller image.
    """
    # Read the ground truth points
    gdf = gpd.read_file(groundtruth_path)

    # Create figure and axis with a larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract survey name from geotiff path
    survey_name = os.path.basename(geotiff_path).replace("-visible.tif", "")

    # Read and plot the drone imagery with decimation
    with rio.open(geotiff_path) as src:
        # Calculate new shape
        height = int(src.height // downsample_factor)
        width = int(src.width // downsample_factor)

        # Read decimated raster
        data = src.read(
            out_shape=(src.count, height, width), resampling=Resampling.average
        )

        # Scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        # Ensure ground truth points are in same CRS as raster
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Plot the raster
        show(data, transform=transform, ax=ax)

    # Plot points with different colors for presence/absence
    presence = gdf[gdf["Presence"] == 1]
    absence = gdf[gdf["Presence"] == 0]

    presence.plot(
        ax=ax, color="green", marker="o", markersize=5, label="Present", alpha=0.5
    )
    absence.plot(
        ax=ax, color="red", marker="o", markersize=5, label="Absent", alpha=0.5
    )

    # Customize the plot
    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Upper Partridge Survey: {survey_name}")
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Use the most recent drone survey
    geotiff_path = os.path.join(
        "data", "raster", "geotiffs", "240702_upperpartridge-visible.tif"
    )
    groundtruth_path = os.path.join("data", "vector", "groundtruth.geojson")

    # For quick development, use higher downsample factor and display instead of save
    create_validation_map(
        geotiff_path, groundtruth_path, output_path=None, downsample_factor=20
    )
