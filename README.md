# EcoPack AI

AI-powered sustainable packaging recommendation system that helps users choose eco-friendly packaging solutions.

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- Joblib
- Matplotlib

## Setup Instructions

### 1. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Labeled Dataset

```bash
python label_dataset.py
```

### 5. Train the Model

```bash
python train_model.py
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

## Project Structure

- app.py – Main Streamlit application
- label_dataset.py – Dataset labeling script
- train_model.py – Model training script
- requirements.txt – Required dependencies
- knn_model.pkl – Trained ML model
- feature_encoders.pkl – Feature encoders
- target_encoder.pkl – Target encoder
- ecopack_kaggle_labeled.csv – Processed dataset

## Features

- Sustainable packaging recommendations
- Carbon footprint analysis
- Eco-score calculation
- Packaging comparison dashboard
- Interactive Streamlit UI
