import pandas as pd
from flask import Flask, request, render_template
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask("__name__")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/predict", methods=['POST'])
def predict_route():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']

    custom_data = CustomData(
        gender=inputQuery4,
        SeniorCitizen=inputQuery1,
        Partner=inputQuery5,
        Dependents=inputQuery6,
        PhoneService=inputQuery7,
        MultipleLines=inputQuery8,
        InternetService=inputQuery9,
        OnlineSecurity=inputQuery10,
        OnlineBackup=inputQuery11,
        DeviceProtection=inputQuery12,
        TechSupport=inputQuery13,
        StreamingTV=inputQuery14,
        StreamingMovies=inputQuery15,
        Contract=inputQuery16,
        PaperlessBilling=inputQuery17,
        PaymentMethod=inputQuery18,
        MonthlyCharges=inputQuery2,
        TotalCharges=inputQuery3,
        tenure=inputQuery19
    )

    data = custom_data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    prediction, probability = predict_pipeline.predict(data)

    if prediction[0] == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
        
    return render_template('result.html', output1=o1, output2=o2)

if __name__ == "__main__":
    app.run(debug=True)