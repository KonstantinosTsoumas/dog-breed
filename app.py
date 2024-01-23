from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.utils.auxiliary_functions import decode_image
from src.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training has been completed successfully."



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():

    print("Predict route was called")  # Debugging line
    if 'image' not in request.json:
        return jsonify({"error": "No image data found"}), 400

    image = request.json['image']
    decode_image(image, clApp.filename)

    # Debug: Check if the image file exists and its size
    if os.path.exists(clApp.filename):
        print(f"Image file {clApp.filename} exists, size: {os.path.getsize(clApp.filename)} bytes")
    else:
        print(f"Image file {clApp.filename} does not exist.")


    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    #app.run(host='0.0.0.0', port=800) #for AZURE