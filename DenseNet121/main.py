from DensenetPrediction import Prediction
from PIL import Image
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

predictObj = Prediction()

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/"+imagefile.filename
    imagefile.save(image_path)
    image = Image.open(image_path)
    
    prediction_idx = predictObj.get_prediction(image)
    print(f' Prediction id = {prediction_idx}')
    class_id, class_name = predictObj.render_prediction(prediction_idx)
    print(f'class id = {class_id} : class name = {class_name}')
    predict_result = "Class_id = "+str(class_id) + "  :: Class_name = "+class_name
    return render_template('index.html', prediction = predict_result)
 
if __name__ == '__main__':
    app.run(port=3000, debug=True)