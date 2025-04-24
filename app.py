from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


@app.route('/')
def home_page():
     return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
     # render the page
     if request.method == 'GET':
          return render_template('form.html')
     
     else:
          data = CustomData(
               Favorite_Color=(request.form.get('Favorite_Color')),
               Favorite_Music_Genre=request.form.get('Favorite_Music_Genre'),
               Favorite_Beverage=request.form.get('Favorite_Beverage'),
               Favorite_Soft_Drink=request.form.get('Favorite_Soft_Drink')
          )
          final_new_data = data.get_data_as_dataframe()
          predict_pipeline = PredictPipeline()
          pred = predict_pipeline.predict(final_new_data)
          label_map = {0: 'Female', 1: 'Male'}
          predicted_gender = label_map.get(pred[0], 'Unknown')
          return render_template('results.html', final_result=predicted_gender)
          
     



if __name__ == "__main__":
     app.run(host = '0.0.0.0', debug=True)