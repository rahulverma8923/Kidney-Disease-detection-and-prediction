#Flask application for kidney disease prediction


from flask import Flask, request, render_template,flash,redirect,session,abort
from models import Model
from writeCsv import write_to_csv
from datetime import datetime
import os
import pandas as pd

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'admin' and request.form['username'] == 'admin':
        session['logged_in'] = True
        return displayrecords()
    else:
        return render_template('loginerror.html')

@app.route('/displayrecords',methods=['GET'])
def displayrecords():
    if not session.get('logged_in'):
        return render_template('login.html')
        
    df = pd.read_csv('dataset/records.csv')
    
    # Format data for better readability
    # Rename columns to more readable names
    df = df.rename(columns={
        'time': 'Test Date & Time',
        'age': 'Age',
        'bloodPressure': 'Blood Pressure',
        'sugar': 'Sugar Level',
        'pusCell': 'Pus Cell',
        'pusCellClumps': 'Pus Cell Clumps',
        'sodium': 'Sodium',
        'hemoglobin': 'Hemoglobin',
        'hypertension': 'Hypertension',
        'diabetesMelitus': 'Diabetes',
        'result': 'Diagnosis'
    })
    
    # Sort the dataframe by date in descending order (newest first)
    # Convert the date string to datetime for proper sorting if needed
    # The format is assumed to be DD/MM/YYYY (HH:MM:SS AM/PM) based on the write_to_csv function
    try:
        # First, keep a copy of the original date format for display
        df['Original Date'] = df['Test Date & Time']
        # Convert to datetime for sorting
        df['Sort Date'] = pd.to_datetime(df['Test Date & Time'], format='%d/%m/%Y (%I:%M:%S %p)')
        # Sort by the datetime column in descending order
        df = df.sort_values(by='Sort Date', ascending=False)
        # Drop the sorting column
        df = df.drop(columns=['Sort Date'])
        # Restore original date format
        df['Test Date & Time'] = df['Original Date']
        df = df.drop(columns=['Original Date'])
    except Exception as e:
        print(f"Error sorting by date: {e}")
        # If sorting fails, continue with unsorted data
    
    # Convert binary values to Yes/No for better readability
    df['Hypertension'] = df['Hypertension'].map({0: 'No', 1: 'Yes'})
    df['Diabetes'] = df['Diabetes'].map({0: 'No', 1: 'Yes'})
    df['Pus Cell'] = df['Pus Cell'].map({0: 'Abnormal', 1: 'Normal'})
    df['Pus Cell Clumps'] = df['Pus Cell Clumps'].map({0: 'Not Present', 1: 'Present'})
    
    # Format the diagnosis results
    df['Diagnosis'] = df['Diagnosis'].map({0: '<span class="badge badge-danger">Kidney Disease</span>', 
                                         1: '<span class="badge badge-success">Healthy</span>'})
    
    # Apply better formatting to the table with additional CSS classes
    html_table = df.to_html(classes='data table table-striped table-hover', 
                           table_id='records-table',
                           index=False,
                           border=0,
                           justify='center',
                           escape=False)  # Allow HTML in table cells
    
    return render_template('displayrecords.html', tables=[html_table])


@app.route('/login',methods=['GET'])
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return root()


@app.route('/predict', methods=["POST"])
def predict():
    age = int(request.form['age'])
    bp = int(request.form['bp'])
    sugar = int(request.form['sugar'])
    pc = int(request.form['pc'])
    pcc = int(request.form['pcc'])
    sodium = int(request.form['sodium'])
    hemo = float(request.form['hemo'])
    htn = int(request.form['htn'])
    db = int(request.form['db'])

    values = [age, bp, sugar, pc, pcc, sodium, hemo, htn,db]
    print(values)
    model = Model()
    classifier = model.random_forest_classifier()
    prediction = classifier.predict([values])
    print(f"Kidney disease = {prediction[0]}")

    time = datetime.now().strftime("%d/%m/%Y (%I:%M:%S %p)")
    write_to_csv(time,age, bp, sugar, pc, pcc, sodium, hemo, htn,db,prediction[0])
    return render_template("result.html", result=prediction[0])

app.secret_key = os.urandom(12)
app.run(port=5000, host='0.0.0.0', debug=True)