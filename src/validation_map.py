import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show


def create_validation_map(geotiff_path, groundtruth_path, output_path=None):
    """
    Create a validation map showing ground truth points overlaid on drone imagery.

    Args:
        geotiff_path (str): Path to the drone survey GeoTIFF
        groundtruth_path (str): Path to the ground truth GeoJSON file
        output_path (str, optional): Path to save the output figure. If None, displays the figure.
    """
    # Read the ground truth points
    gdf = gpd.read_file(groundtruth_path)

    # Create figure and axis with a larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Read and plot the drone imagery
    with rio.open(geotiff_path) as src:
        # Ensure ground truth points are in same CRS as raster
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Plot the raster
        show(src, ax=ax)

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
    ax.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Use the most recent drone survey
    geotiff_path = os.path.join(
        "data", "raster", "geotiffs", "240828_upperpartridge-visible.tif"
    )
    groundtruth_path = os.path.join("data", "vector", "groundtruth.geojson")

    # Create output directory if it doesn't exist
    output_path = os.path.join("../results/figures", "validation_map.png")

    create_validation_map(geotiff_path, groundtruth_path, output_path)
