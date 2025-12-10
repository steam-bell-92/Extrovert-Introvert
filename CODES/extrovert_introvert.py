import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import zipfile

class PersonalityClassifier:
    def __init__(self, model_path='personality_model.joblib'):
        self.model = LogisticRegression()
        self.model_path = model_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Correct features based on your dataset
        self.features = [
            'Time_spent_Alone', 
            'Social_event_attendance', 
            'Going_outside', 
            'Friends_circle_size', 
            'Post_frequency'
        ]
        self.target = 'Personality'

    def load_and_clean(self, filepath):
        """
        Loads data, removes duplicates, and handles basic setup.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Drop duplicates
        initial_shape = df.shape
        df = df.drop_duplicates(keep='first')
        df.index = range(1, len(df) + 1)
        
        print(f"Duplicates dropped: {initial_shape[0] - df.shape[0]}")
        self.data = df
        
        # Check for class balance
        print("Class Distribution:")
        print(self.data[self.target].value_counts())

    def visualize_data(self):
        """
        Generates the pairplot and heatmaps.
        """
        if self.data is None: return
        
        # Pairplot
        print("Generating Pairplot...")
        sns.pairplot(data=self.data, hue=self.target)
        plt.show()
        
        # Correlation Heatmaps
        for p_type in ['Introvert', 'Extrovert']:
            subset = self.data[self.data[self.target] == p_type]
            plt.figure(figsize=(8, 6))
            sns.heatmap(subset.corr(numeric_only=True), annot=True, cmap='coolwarm')
            plt.title(f'Correlation Matrix: {p_type}')
            plt.show()

    def train_sklearn(self):
        """
        Trains the standard Scikit-Learn Logistic Regression.
        """
        X = self.data[self.features]
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(self.X_train, self.y_train)
        print("Scikit-Learn Model Trained.")
        
        # Save immediately
        joblib.dump(self.model, self.model_path)

    def train_manual_gradient_descent(self, learning_rate=0.01, iterations=20000):
        """
        Your manual implementation of Logistic Regression Math.
        Returns the weights and bias.
        """
        print("\n--- Starting Manual Gradient Descent ---")
        
        # Prepare data (Map Extrovert to 1, Introvert to 0)
        X_val = self.data[self.features].values.astype(float)
        y_val = (self.data[self.target].values == 'Extrovert').astype(float)
        
        m, n = X_val.shape
        w = np.zeros(n, dtype=float)
        b = 0.0
        
        for i in range(iterations):
            # 1. Linear step
            z = np.dot(X_val, w) + b
            
            # 2. Sigmoid activation
            p = 1 / (1 + np.exp(-z))
            
            # 3. Calculate Gradient
            error = p - y_val
            dw = (X_val.T @ error) / m
            db = error.mean()
            
            # 4. Update Weights
            w -= learning_rate * dw
            b -= learning_rate * db
            
        print(f"Manual Training Complete.")
        print(f"Weights: {w}")
        print(f"Bias: {b}")
        return w, b

    def evaluate(self):
        """
        Evaluates the Scikit-Learn model.
        """
        y_pred = self.model.predict(self.X_test)
        
        print("\n--- Evaluation Results ---")
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

    def create_gradio_app(self):
        """
        Generates the app.py file with the CORRECT feature inputs.
        """
        app_code = f"""
import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("{self.model_path}")

def predict_personality(alone, event, outside, friends, post):
    # Inputs must match training order: 
    # ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    
    input_array = np.array([[alone, event, outside, friends, post]])
    prediction = model.predict(input_array)[0]
    
    # Sklearn usually maps classes alphabetically if not specified.
    # Assuming standard behavior, but returning raw prediction is safer if string.
    return str(prediction)

demo = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.Slider(0, 10, label="Time Spent Alone (Hours)"),
        gr.Slider(0, 10, label="Social Event Attendance (Count)"),
        gr.Slider(0, 10, label="Going Outside (Frequency)"),
        gr.Slider(0, 100, label="Friends Circle Size"),
        gr.Slider(0, 10, label="Post Frequency (Per Week)")
    ],
    outputs="label",
    title="Introvert vs Extrovert Detector",
    description="Based on behavioral data, not Big 5 traits."
)

demo.launch()
"""
        # Write files
        with open("app.py", "w") as f:
            f.write(app_code)
            
        with open("requirements.txt", "w") as f:
            f.write("gradio\nscikit-learn\nnumpy\njoblib")
            
        # Zip it
        with zipfile.ZipFile("personality_app.zip", "w") as zipf:
            zipf.write(self.model_path)
            zipf.write("app.py")
            zipf.write("requirements.txt")
            
        print("App generated and zipped as 'personality_app.zip'")

# --- Usage ---
if __name__ == "__main__":
    classifier = PersonalityClassifier()
    
    # create dummy csv for testing if file doesn't exist
    try:
        classifier.load_and_clean("personality_dataset.csv")
        classifier.train_sklearn()
        classifier.evaluate()
        
        # Compare with manual math
        classifier.train_manual_gradient_descent()
        
        # Generate the corrected App
        classifier.create_gradio_app()
        
    except FileNotFoundError:
        print("Please upload 'personality_dataset.csv' to run.")