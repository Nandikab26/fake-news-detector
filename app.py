# app.py

from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Model or vectorizer files not found. Please run train_model.py first.")
    exit()

@app.route('/')
def home():
    """Renders the main page (we will create this HTML file next)."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives text and returns a prediction."""
    try:
        data = request.get_json()
        news_text = data['text']
        
        # Vectorize the input text using the loaded vectorizer
        vectorized_text = vectorizer.transform([news_text])
        
        # Make a prediction using the loaded model
        prediction = model.predict(vectorized_text)
        
        # Convert prediction to a human-readable label
        result = "FAKE" if prediction[0] == 1 else "RELIABLE"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)