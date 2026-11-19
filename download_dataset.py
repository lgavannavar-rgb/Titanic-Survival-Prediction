from pathlib import Path
from model_utils import download_dataset

destination = Path(__file__).parent / "dataset" / "train.csv"
download_dataset(destination)
print(f"Titanic dataset downloaded to: {destination}")
