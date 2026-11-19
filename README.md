# Titanic Survival Intelligence — Logistic Regression

An advanced educational Streamlit application built around the original Kaggle Titanic training dataset and a Logistic Regression classifier.

## Features
- Overview dashboard and historical survival statistics
- Individual passenger survival probability
- Bulk prediction from a CSV upload
- Survival analytics and interactive charts
- Hold-out test metrics: accuracy, precision, recall, F1-score
- Confusion matrix and ROC curve
- Logistic Regression coefficient insights
- Session-based prediction history and CSV export

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first dataset download

## Setup on Windows
1. Extract the ZIP file.
2. Open the extracted `Titanic_Survival_AI` folder.
3. Open Command Prompt in this folder.
4. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

5. Install libraries:

   ```bash
   pip install -r requirements.txt
   ```

6. Download the original Titanic dataset:

   ```bash
   python download_dataset.py
   ```

   The file is saved as `dataset/train.csv`. The app can also attempt this download automatically on first launch.

7. Start the application:

   ```bash
   streamlit run app.py
   ```

8. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

## Dataset
The app uses the Kaggle Titanic training dataset (891 passenger records). It downloads from:

https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv

If the automatic download does not work, download the original `train.csv` from Kaggle and place it in the `dataset` folder.

## CSV format for bulk prediction
Required columns:
`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`

Example values:
- `Pclass`: 1, 2, or 3
- `Sex`: male or female
- `Embarked`: S, C, or Q

The target `Survived` column is not required for new passenger predictions.

## Important notes
- The model is trained from the included Titanic training dataset after it is downloaded.
- A stratified 80/20 train/test split is used for evaluation.
- Missing numerical values are imputed with the training-fold median; missing categorical values use the most frequent value.
- Numeric features are standardized and categorical features are one-hot encoded inside a scikit-learn Pipeline.
- Predictions are educational estimates from historical patterns, not certainty about an individual's fate.
- Prediction history is kept only in the current Streamlit session; it is not a permanent database.

## Project structure
```text
Titanic_Survival_AI/
├── app.py
├── model_utils.py
├── download_dataset.py
├── requirements.txt
├── README.md
└── dataset/
    └── (train.csv downloaded here)
```
