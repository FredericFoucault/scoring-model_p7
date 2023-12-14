from flask import Flask, render_template, jsonify,request
import pandas as pd
import pickle
import json

app = Flask(__name__)
 
 
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 
@app.route('/')
def index():
    return jsonify(dictionnaire)
 

data = pd.DataFrame(dictionnaire)


#chk_id=0 # how to get the selected client ID frpù selectbox in streamlit ?



X=data.drop(['SK_ID_CURR','TARGET'],axis=1)



@app.route('/predict', methods=['GET','POST'])
def predict():
    ClientID = request.args.get('ClientID')
    #ClientID = request.form.to_dict('ClientID')
    score = model.predict_proba(X[X.index == int(ClientID)])[:,1].tolist()
    #score=score[0]*100
    
    score=float(score[0]*100)
    dct= {'ClientID': int(ClientID),
            'prediction':score}
    return jsonify(dct)



if __name__ == "__main__":

    modelfile= '../model_LGBM.pkl'
    model = pickle.load(open(modelfile, 'rb'))
    print("loaded ok")
    app.run(debug=True)
