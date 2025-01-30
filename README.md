# mpg-horses

## Notebooks

### Google Colab Notebooks
- [Ground Truth - Horse Piles](https://colab.research.google.com/drive/17YSHHRmFFNArP-f3--oos-DtAtWLHhJR?usp=sharing) - Notebook developed in 2024 for processing horse pile ground truth data
- [Crop Tiles](https://colab.research.google.com/drive/15LFRMVOfEiF__FswVqTQstZC5ocC6Ur0?usp=sharing) - Notebook for cropping and processing aerial imagery tiles

### Local Notebooks
- `01_concat_ground_truth.ipynb` - Concatenates ground truth data
- `02_extract_tiles.ipynb` - Extracts image tiles from the dataset
- `03_push_to_hugging_face.ipynb` - Pushes processed data to Hugging Face

## Data Locations

### Ground Truth Data
- Processed horse pile ground truth data: `gs://mpg-aerial-survey/ground_truth/horsepile/processed`

## Methods

### Ground Truth Collection
Ground truth data is collected using Emlid GNSS rovers in a global coordinate system (WGS84/EPSG:4326). Field technicians use the rovers to record precise locations of horse piles during ground surveys. This data serves as the reference dataset for training and validating detection models.

The Emlid rovers record coordinates in latitude and longitude format (WGS84).

#### Wrangled Ground Truth Data Structure
The processed ground truth dataset contains the following key metadata columns:
- `Point_Index`: Unique identifier for each observation point
- `Presence`: Boolean indicator (1 for horse pile present, 0 for absent)
- `Zone`: Survey zone identifier
- `Period`: Time period of the survey
- `Description`: Additional notes or observations
- `Datetime`: Timestamp of data collection
- `Latitude`: WGS84 latitude coordinate
- `Longitude`: WGS84 longitude coordinate
- `Ellipsoidal height`: Height above WGS84 ellipsoid

#### Absence Data Collection
The absence dataset consists of locations where field technicians confirmed no horse piles were present. Date of collection is known and is set during wrangling.
