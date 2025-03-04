# Horse Detection Dataset

This dataset contains ground truth data for horse detection in aerial imagery. The data includes presence/absence points and associated orthomosaic tiles.

## Dataset Structure

The dataset is split into training and testing sets:

### Training Set
- Total points: 1278
- Presence points: 710
- Absence points: 568

### Testing Set
- Total points: 312
- Presence points: 174
- Absence points: 138

## Features

The dataset includes the following features:

1. `idx`: Unique identifier for each point
2. `Presence`: Binary indicator (1 for presence, 0 for absence)
3. `Zone`: Zone identifier
4. `Period`: Time period identifier
5. `Recency`: Recency indicator
6. `Datetime`: Timestamp of the observation
7. `Latitude`: Latitude coordinate
8. `Longitude`: Longitude coordinate
9. `Easting`: Easting coordinate (UTM)
10. `Northing`: Northing coordinate (UTM)
11. `Ellipsoidal_height`: Height above ellipsoid
12. `orthomosaic`: Identifier for the associated orthomosaic
13. `observation_offset`: Days between observation and orthomosaic capture
14. `encoded_tile`: Base64-encoded image tile

## Data Collection

The data was collected through aerial surveys conducted at the MPG Ranch. Each point represents a ground truth observation of horse presence or absence.

## Usage

The dataset can be loaded using the Hugging Face datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("path/to/dataset")
```

## License

[Add appropriate license information]
