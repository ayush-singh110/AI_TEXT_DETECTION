from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    """
    Renders the welcome/index page with developer information.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles the prediction logic.
    GET request: Shows the form to input text.
    POST request: Processes the text and returns the prediction.
    """
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            input_text = request.form.get('text_input')
            
            # Validate input
            if not input_text or input_text.strip() == '':
                return render_template('home.html', error="Please enter some text to analyze.")
            
            # Additional validation for minimum text length
            if len(input_text.strip()) < 10:
                return render_template('home.html', error="Please enter at least 10 characters for accurate analysis.")
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(input_text)
            
            if results == 1:
                prediction_text = "AI-Generated"
            else:
                prediction_text = "Human-Written"
                
            return render_template('home.html', results=prediction_text, original_text=input_text)
            
        except Exception as e:
            return render_template('home.html', error=f"An error occurred during analysis: {str(e)}")

@app.route('/about')
def about():
    """
    Renders an about page with information about the developer and the project.
    """
    return render_template('index.html')  # Redirects to main page for now

if __name__ == '__main__':  # Fixed: was **name** instead of __name__
    print("Starting Flask application...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print("Port 5000 is already in use. Trying port 5001...")
            app.run(host='0.0.0.0', port=5001, debug=True)
        else:
            print(f"Error starting server: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")