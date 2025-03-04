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
            # Handle both API requests and form submissions
            if request.is_json:
                # JSON API call
                json_ = request.json
                is_api_call = True
            else:
                # Form data from web UI
                is_api_call = False
                # Check if we have multiple passengers
                passenger_count = int(request.form.get('passenger_count', 1))
                json_ = []

                # Process each passenger
                for i in range(1, passenger_count + 1):
                    # Get passenger data
                    try:
                        passenger = {
                            'Age': float(request.form.get(f'Age_{i}', 0)),
                            'Sex': request.form.get(f'Sex_{i}', ''),
                            'Embarked': request.form.get(f'Embarked_{i}', '')
                        }
                        json_.append(passenger)
                    except ValueError:
                        # Skip invalid passengers
                        continue

            # No valid passengers
            if not json_:
                return render_template('index.html', error="Please enter valid passenger data")

            # Make predictions
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            predictions = list(map(int, lr.predict(query)))

            # Calculate survival statistics
            total_passengers = len(predictions)
            total_survived = sum(predictions)
            survival_rate = (total_survived / total_passengers) * 100 if total_passengers > 0 else 0

            # Return appropriate response based on request type
            if is_api_call:
                # For API calls, return JSON
                return jsonify({
                    'predictions': predictions,
                    'stats': {
                        'total_passengers': total_passengers,
                        'survived': total_survived,
                        'survival_rate': survival_rate
                    }
                })
            else:
                # For form submissions, render template with results
                passengers_with_predictions = []
                for i, (passenger, survived) in enumerate(zip(json_, predictions)):
                    passenger['id'] = i + 1
                    passenger['survived'] = survived == 1
                    passengers_with_predictions.append(passenger)

                return render_template('index.html',
                                      passengers=passengers_with_predictions,
                                      show_results=True,
                                      total_passengers=total_passengers,
                                      total_survived=total_survived,
                                      survival_rate=survival_rate)

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