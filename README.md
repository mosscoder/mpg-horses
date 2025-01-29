# mpg-horses

## Notebooks

### Google Colab Notebooks
- [Ground Truth - Horse Piles](https://colab.research.google.com/drive/17YSHHRmFFNArP-f3--oos-DtAtWLHhJR?usp=sharing) - Notebook developed in 2024 for processing horse pile ground truth data

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

The Emlid rovers record coordinates in latitude and longitude format (WGS84). While the output includes fields for Easting and Northing, these are not used and are marked as NA in the data. The primary spatial coordinates used for all analyses are latitude and longitude.

#### Absence Data Collection
The absence dataset consists of locations where field technicians confirmed no horse piles were present. These points do not include start datetime information.
