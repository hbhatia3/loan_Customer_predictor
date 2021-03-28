import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if(prediction[0]==1):
    	return render_template('home.html', prediction_text='***This customer can be a potential buyer of our personal loan. ADD him/her to our campaign list!***')
    else:
    	return render_template('home.html', prediction_text='***This customer can NOT be a potential buyer of our personal loan. DON\'T add him/her to our campaign list!***')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

#if __name__ == "__main__":
#    app.run(debug=True)