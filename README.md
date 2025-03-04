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
- Input: Raw survey data from GNSS rovers (CSV files from Google Cloud Storage)
  - `gs://mpg-aerial-survey/ground_truth/horsepile/processed/horse_pile_presence.csv`
  - `gs://mpg-aerial-survey/ground_truth/horsepile/processed/horse_pile_absence.csv`
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
- Dependencies:
  - pandas, geopandas, rasterio, concurrent.futures for parallel processing

#### 4. Model Data Preparation
**`04_load_hf_dataframe.ipynb`**
- Purpose: Prepares and loads data for machine learning model training
- Input: Processed image tiles and ground truth data
- Output: Hugging Face compatible dataset with train-test splits
- Key Operations:
  - Data formatting for deep learning
  - Feature engineering
  - Dataset splitting and validation
  - Hugging Face dataset creation
- Key Features:
  - Implements reproducible train-test splitting (using RANDOM_SEED=42)
  - Processes GeoJSON data and orthomosaic tiles
  - Creates structured directories for model training data
  - Exports data in formats compatible with Hugging Face datasets

## Train-Test Split Methodology

The dataset is split into training (80%) and testing (20%) sets using a stratified approach that preserves the distribution of presence and absence points while ensuring spatial and temporal representativeness. The split methodology is implemented in `04_load_hf_dataframe.ipynb` and follows these steps:

### For Absence Points (Negative Samples)
1. **Filtering**: Absence points within 2 meters of any presence point are excluded to ensure they are truly representative of areas without horse piles.
2. **Spatial Blocking**: K-means clustering is applied to create spatial blocks based on geographic coordinates, ensuring spatial representativeness.
3. **Stratified Sampling**: 20% of points are randomly sampled from each spatial block for the test set, maintaining the spatial distribution.

### For Presence Points (Positive Samples)
1. **Stratification by Zone and Period**: Presence points are grouped by their zone and time period attributes.
2. **Proportional Sampling**: 20% of points are randomly sampled from each zone-period group for the test set.
3. **Temporal Consistency**: This approach ensures that the test set includes samples from all survey zones and time periods.

### Implementation Details
- Random seed (RANDOM_SEED=42) is used for reproducibility
- The resulting split maintains approximately the same presence-to-absence ratio in both training and testing sets
- The split is performed on the original ground truth points before tile extraction, ensuring that all tiles derived from the same point are in the same split

### Dataset Statistics
- Total dataset: ~15,900 point-orthomosaic combinations
  - Training set: ~12,780 combinations (80%)
  - Testing set: ~3,120 combinations (20%)
- Each point may appear multiple times in the dataset, paired with different orthomosaic dates to capture temporal variations

This stratified splitting approach ensures that the model evaluation is robust across different spatial locations and temporal conditions, providing a more reliable assessment of model performance.

## Project Structure
```
mpg-horses/
├── data/                  # Data directory (typically gitignored)
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   │   └── splits/       # Train-test splits for ML models
│   ├── tabular/          # CSV and tabular data files
│   ├── hf/               # Hugging Face formatted datasets
│   └── external/         # Third-party data sources
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_concat_ground_truth.ipynb  # Ground truth data processing
│   ├── 02_figures.ipynb              # Data visualization
│   ├── 03_extract_tiles.ipynb        # Image tile extraction
│   └── 04_load_hf_dataframe.ipynb    # ML dataset preparation
├── results/              # Generated analysis results
│   ├── figures/          # Generated graphics and figures
│   ├── docs/             # Documentation files
│   └── outputs/          # Other outputs
└── docs/                # Documentation
```

## Notebook Dependencies
Each notebook requires specific Python packages. Key dependencies include:
- pandas: Data manipulation and analysis
- geopandas: Geospatial data processing
- rasterio: Raster data handling
- matplotlib/seaborn: Data visualization
- numpy: Numerical operations
- datasets: Hugging Face dataset management
- scikit-learn: For machine learning operations (KMeans clustering)
- tqdm: Progress bar visualization

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
- `idx`: Unique identifier for each point
- `Presence`: Binary indicator (1 for presence, 0 for absence)
- `Zone`: Zone identifier
- `Period`: Time period identifier
- `Recency`: Recency indicator
- `datetime_groundtruth`: Timestamp of the ground truth observation
- `datetime_aerialsurvey`: Timestamp when the aerial imagery was captured
- `Latitude`: Latitude coordinate
- `Longitude`: Longitude coordinate
- `Easting`: Easting coordinate (UTM)
- `Northing`: Northing coordinate (UTM)
- `Ellipsoidal_height`: Height above ellipsoid
- `geometry`: WKT representation of the point geometry
- `orthomosaic`: Identifier for the associated orthomosaic
- `observation_offset`: Days between observation and orthomosaic capture
- `encoded_tile`: Base64-encoded image tile

#### Absence Data Collection
The absence dataset consists of locations where field technicians confirmed no horse piles were present. Date of collection is known and is set during wrangling.
