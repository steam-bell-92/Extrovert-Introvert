
import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.joblib")

def predict_personality(openness, conscientiousness, extraversion, agreeableness, neuroticism):
    input_array = np.array([[openness, conscientiousness, extraversion, agreeableness, neuroticism]])
    prediction = model.predict(input_array)[0]
    label = "Extrovert" if prediction == 1 else "Introvert"
    return label

demo = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.Slider(0, 1, step=0.01, label="Openness"),
        gr.Slider(0, 1, step=0.01, label="Conscientiousness"),
        gr.Slider(0, 1, step=0.01, label="Extraversion"),
        gr.Slider(0, 1, step=0.01, label="Agreeableness"),
        gr.Slider(0, 1, step=0.01, label="Neuroticism")
    ],
    outputs=gr.Label(num_top_classes=2),
    title="Extrovert vs Introvert Classifier",
    description="Enter personality trait scores (0 to 1) to predict whether someone is more extroverted or introverted.",
    theme="default"
)

demo.launch()
