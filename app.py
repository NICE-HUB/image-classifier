from flask import Flask, request, render_template
import model = load_model('C:\Users\HP\projects-/dogcatclassification') 

app = Flask(__name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_image = request.files['image']

    prediction = your_image_classifier_module.predict(uploaded_image)

    return prediction

if __name__ == '__main__':
    app.run(debug=True)







