from flask import Flask, request, render_template
import os
from waitress import serve

app = Flask(__name__)

# Function to handle the uploaded image
@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML form to upload the image

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']  # Get the uploaded file
    if file:
        file_path = os.path.join('uploads', file.filename)  # Save the file
        file.save(file_path)

        # Get predictions for the car model and color
        car_model = predict_car_model(file_path)
        car_color = detect_car_color(file_path)

        return f"Car Model: {car_model} | Car Color: {car_color}"

if __name__ == "__main__":
    app.run(debug=True)
    # Start the app using Waitress server for local development
    serve(app, host='0.0.0.0', port=5000)
