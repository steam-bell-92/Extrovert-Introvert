# 🧠 Extrovert vs Introvert Personality Classifier

This project is a simple machine learning app that predicts whether a person is more likely an **Extrovert** or an **Introvert** based on their scores in the **Big Five Personality Traits**.

## 🧰 Tech Stack

| Tool / Library | Purpose                          |
|----------------|----------------------------------|
| **NumPy**      | Numerical computations, arrays   |
| **Pandas**     | Data manipulation & cleaning     |
| **Matplotlib** | Plotting and visualizing data    |
| **Seaborn**    | Statistical data visualization   |
| **Scikit-learn** | Model training and evaluation  |
| **Joblib**     | Saving and loading the ML model  |
| **Gradio**     | Building the web-based interface |
| **Hugging Face Spaces** | Free app hosting        |


## 🚀 Try the App Live

👉 [**Click here to use the live app on Hugging Face**](https://huggingface.co/spaces/steam-bell-92/Extrovert-Introvert)

> 🔁 Adjust sliders for each personality trait and see your predicted type in real time!

## 🧪 Features Used in Prediction

- 🧠 Openness (Creativity & Curiosity)
- ✅ Conscientiousness (Discipline & Organization)
- 🔊 Extraversion (Sociability & Energy)
- 💖 Agreeableness (Kindness & Cooperativeness)
- 🌪️ Neuroticism (Stress Sensitivity)

## 🛠️ How It Works

1. The model was trained in a Google Colab notebook using logistic regression.
2. Exported using `joblib` and loaded into `app.py`.
3. Gradio provides the interactive interface with sliders and instant predictions.
4. Hosted for free via Hugging Face Spaces.

## 📁 Project Structure
```
Extrovert-Introvert/
├── app.py                              🔹 # Gradio app script
├── model.joblib                        🔹 # Trained ML model
├── requirements.txt                    🔹 # Dependencies
├── README.md                           🔹 # You're reading it!
└── Extrovert_Introvert.ipynb           🔹 # Where code's present
```

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92
