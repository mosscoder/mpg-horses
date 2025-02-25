# mpg-horses

## Project Overview
This repository contains a collection of Jupyter notebooks and supporting code for processing and analyzing horse pile data at MPG Ranch. The workflow includes ground truth data processing, image tile extraction, visualization, and machine learning model data preparation.

## Notebooks

The analysis workflow is split between Google Colab notebooks for resource-intensive processing and local notebooks for data preparation and analysis.

### Google Colab Notebooks

These notebooks are designed to run in Google Colab to leverage cloud computing resources:

- [Ground Truth - Horse Piles](https://colab.research.google.com/drive/17YSHHRmFFNArP-f3--oos-DtAtWLHhJR?usp=sharing)
  - Purpose: Processing horse pile ground truth data collected in 2024
  - Input: Raw GNSS survey data
  - Output: Processed ground truth coordinates and metadata
  - Key Features: Coordinate transformation, data validation, metadata enrichment

- [Crop Tiles](https://colab.research.google.com/drive/15LFRMVOfEiF__FswVqTQstZC5ocC6Ur0?usp=sharing)
  - Purpose: Processing and cropping aerial imagery tiles
  - Input: Raw aerial imagery
  - Output: Cropped image tiles ready for analysis
  - Key Features: Image preprocessing, tile extraction, quality control

### Local Notebooks

These notebooks are designed to run on your local machine and follow a sequential workflow:

#### 1. Ground Truth Data Processing
**`01_concat_ground_truth.ipynb`**
- Purpose: Concatenates and processes ground truth data from multiple surveys
- Input: Raw survey data from GNSS rovers
- Output: Consolidated GeoJSON with standardized metadata
- Key Operations:
  - Merges multiple survey datasets
  - Standardizes coordinate systems
  - Validates data quality
  - Enriches with temporal and spatial metadata

#### 2. Data Visualization
**`02_figures.ipynb`**
- Purpose: Generates figures and visualizations for analysis and reporting
- Input: Processed ground truth data, image tiles
- Output: Publication-ready figures and plots
- Key Visualizations:
  - Spatial distribution of observations
  - Temporal patterns
  - Data quality assessments
  - Statistical summaries

#### 3. Image Processing
**`03_extract_tiles.ipynb`**
- Purpose: Extracts and processes image tiles from aerial imagery
- Input: Raw aerial imagery, ground truth coordinates
- Output: Processed image tiles for model training
- Key Features:
  - Tile extraction around ground truth points
  - Image preprocessing and normalization
  - Quality control checks
  - Metadata association

#### 4. Model Data Preparation
**`04_load_hf_dataframe.ipynb`**
- Purpose: Prepares and loads data for machine learning model training
- Input: Processed image tiles and ground truth data
- Output: Hugging Face compatible dataset
- Key Operations:
  - Data formatting for deep learning
  - Feature engineering
  - Dataset splitting and validation
  - Hugging Face dataset creation

## Notebook Dependencies
Each notebook requires specific Python packages. Key dependencies include:
- pandas: Data manipulation and analysis
- geopandas: Geospatial data processing
- rasterio: Raster data handling
- matplotlib/seaborn: Data visualization
- numpy: Numerical operations
- datasets: Hugging Face dataset management

For complete environment setup, see `environment.yml`.

## Data Locations

### Ground Truth Data
- Processed horse pile ground truth data: `gs://mpg-aerial-survey/ground_truth/horsepile/processed`

## Methods

### Ground Truth Collection
Ground truth data is collected using Emlid GNSS rovers in a global coordinate system (WGS84/EPSG:4326). Field technicians use the rovers to record precise locations of both horse piles (presence) and confirmed absence locations during ground surveys. This proximity-based data collection serves as the reference dataset for training and validating detection models.

The Emlid rovers record coordinates in latitude and longitude format (WGS84). For each survey location, technicians record either the presence of a horse pile or explicitly confirm its absence, with the date of collection captured during data wrangling.

#### Wrangled GeoJSON Ground Truth Data Structure
The processed ground truth dataset is exported as a GeoJSON vector file from `01_concat_ground_truth.ipynb` with the following key metadata columns:
- `idx`: Unique identifier for each observation point (zero-padded four-digit integer)
- `Presence`: Boolean indicator (1 for horse pile present, 0 for absent)
- `Zone`: Survey zone identifier
- `Period`: Time period of the survey
- `Recency`: Additional metadata about observation recency
- `Datetime`: UTC timestamp of data collection
- `Latitude`: WGS84 latitude coordinate
- `Longitude`: WGS84 longitude coordinate
- `Easting`: UTM Zone 11N easting coordinate (meters)
- `Northing`: UTM Zone 11N northing coordinate (meters)
- `Ellipsoidal_height`: Height above WGS84 ellipsoid
- `geometry`: GeoJSON Point geometry containing coordinates

#### Absence Data Collection
The absence dataset consists of locations where field technicians confirmed no horse piles were present. Date of collection is known and is set during wrangling.
