# Extrovert vs Introvert Personality Classifier

## Overview

This project predicts whether a person is more likely an extrovert or an introvert from behavior-based features derived from the Big Five personality traits. The analysis, model, and web app remain the same; the repository has been standardized into a cleaner, reproducible layout.

## Problem Statement

Given a set of personality-related inputs, classify a person as either `Extrovert` or `Introvert` using a logistic regression model trained on the provided dataset.

## Dataset

Source: Kaggle dataset by Rakesh Kapilavayi, *Extrovert vs Introvert Behavior Data*.

The raw dataset is stored in [data/raw/personality_dataset.csv](data/raw/personality_dataset.csv).

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Gradio

## Project Structure

- [data/raw](data/raw) stores the original dataset.
- [notebooks](notebooks) contains the exploratory notebook.
- [src](src) contains the training and export script.
- [models](models) stores the serialized trained model.
- [WEB](WEB) contains the Gradio app entry point.
- [images](images) is reserved for README screenshots later.

## Workflow

1. Load the dataset from `data/raw`.
2. Inspect and clean the data.
3. Run exploratory data analysis and visualizations.
4. Train a logistic regression model.
5. Evaluate the model using classification metrics.
6. Export the trained model and Gradio app assets.

## Results

The existing model performance reported in the project is approximately `0.908` balanced accuracy.

## Future Improvements

- Add automated data validation before training.
- Separate training, evaluation, and export steps into individual commands.
- Add a lightweight test suite for the data-loading and prediction helpers.
- Expand documentation with example inputs and screenshots.

## License

See [LICENSE](LICENSE) for details.
