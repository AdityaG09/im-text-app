import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
# app
app = Flask(__name__)

clf = pickle.load(open('clf.pkl','rb'))
loaded_vec = pickle.load(open('count_vect.pkl', 'rb'))

# routes
@app.route('/', methods=['GET', 'POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    result_pred = clf.predict(loaded_vec.transform([data]))
    print(result_pred)

    # send back to browser
    output = {'results': int(result_pred[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)