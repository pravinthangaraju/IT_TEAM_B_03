
from flask import Flask, request, render_template, jsonify,url_for
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
app = Flask(__name__)
import webbrowser

# Load your machine learning model (replace 'modelweb.pkl' with your actual model file)
# modelweb = pickle.load(open('modelweb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('whatsappfrontend.html')

@app.route('/andre')
def andre():
    return render_template('andre.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        ipme = request.form['mss']
        data =[ipme]
        
        raw_message_data = pd.read_csv('message_project_data.csv',encoding="latin-1")
        message_data = raw_message_data.where((pd.notnull(raw_message_data)),'')

        message_data.loc[message_data['Category'] == 'nor', 'Category',] = 0
        message_data.loc[message_data['Category'] == 'imp', 'Category',] = 1

        X = message_data['Message']
        Y = message_data['Category']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

        feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)

        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

       # model = MultinomialNB()
        model = LogisticRegression()
        #model = SVC(kernel='linear', C=1.0)

        model.fit(X_train_features, Y_train)

        input_message = data
        # convert text to feature vectors
        input_data_features = feature_extraction.transform(input_message).toarray()

         # making prediction

        prediction = model.predict(input_data_features)
        #print(prediction)
        # Output the prediction result
        if(prediction[0]==1):
            f = "Important Message from jannu:"+ipme 
        else:
            f="" 
        return render_template('whatsappbackend.html', prediction_text="{}".format(f),res="{}".format(ipme))

    except Exception as e:
        return render_template('whatsappbackend.html', prediction_text="Error: {}".format(str(e)))
if __name__ == "__main__":
    
    app.run(debug=True)