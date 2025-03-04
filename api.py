import sys
import traceback

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            if request.is_json:
                json_ = request.json
            else:
                form_data = request.form.to_dict()
                if 'Age' in form_data:
                    try:
                        form_data['Age'] = float(form_data['Age'])
                    except ValueError:
                        form_data['Age'] = 0
                json_ = [form_data]

            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(map(int, lr.predict(query)))

            if not request.is_json:
                survived = prediction[0] == 1
                return render_template('index.html',
                                       prediction=prediction[0],
                                       survived=survived,
                                       show_result=True,
                                       passenger_data=json_[0])

            return jsonify({'prediction': prediction})
        except Exception as e:
            print(traceback.format_exc())
            if not request.is_json:
                return render_template('index.html', error=str(e))
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    lr = joblib.load("model.pkl")
    print('Model loaded')
    model_columns = joblib.load("model_columns.pkl")
    print('Model columns loaded')
    app.run(port=port, debug=True)
