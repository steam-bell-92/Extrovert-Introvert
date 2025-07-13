# 🧠 Extrovert vs Introvert Personality Classifier

This project is a simple machine learning app that predicts whether a person is more likely an **Extrovert** or an **Introvert** based on their scores in the **Big Five Personality Traits**.

 - **`Balanced Accuracy Score` ≈ 0.91**<br>
 - **`Score` = 0.908**

⭐ If you find this useful, consider giving it a star!

 > The webapp is deployed using `Hugging Face`
---

## 🧰 Tech Stack

| Tool / Library | Purpose                          |
|----------------|----------------------------------|
| **`NumPy`**      | Numerical computations, arrays   |
| **`Pandas`**     | Data manipulation & cleaning     |
| **`Matplotlib`** | Plotting and visualizing data    |
| **`Seaborn`**    | Statistical data visualization   |
| **`Scikit-learn`** | Model training and evaluation  |
| **`Joblib`**     | Saving and loading the ML model  |
| **`Gradio`**     | Building the web-based interface |
| **`Hugging Face Spaces`** | Free app hosting        |

---

## 🚀 Try the App Live

👉 [**Click here to use the live app on Hugging Face**](https://huggingface.co/spaces/steam-bell-92/Extrovert-Introvert)

> 🔁 Adjust sliders for each personality trait and see your predicted type in real time!

---

## 🧪 Features Used in Prediction

- 🧠 Openness (Creativity & Curiosity)
- ✅ Conscientiousness (Discipline & Organization)
- 🔊 Extraversion (Sociability & Energy)
- 💖 Agreeableness (Kindness & Cooperativeness)
- 🌪️ Neuroticism (Stress Sensitivity)

---

## 🛠️ How It Works

1. The model was trained in a Google Colab notebook using logistic regression.
2. Exported using `joblib` and loaded into `app.py`.
3. Gradio provides the interactive interface with sliders and instant predictions.
4. Hosted for free via Hugging Face Spaces.

> Dataset is from: <a href="https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data">Kaggle (Rakesh Kapilavayi)</a><br>
Also the Dataset is uploaded on repo

---

## 📁 Project Structure
```
Extrovert-Introvert/
├── app.py                              🔹 Gradio app script
├── model.joblib                        🔹 Trained ML model
├── requirements.txt                    🔹 Dependencies
├── README.md                           🔹 You're reading it!
├── personality_dataset.csv             🔹 Dataset
├── extrovert_introvert.py              🔹 Python File of ML Workflow
└── Extrovert_Introvert.ipynb           🔹 ML Workflow
```

---

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
