from model_prediction import prediction
from read_clientdata import avail_client
from flask import Flask, request,render_template,redirect, url_for
import pandas as pd

app = Flask(__name__)

model_file = 'data/finalized_model.sav'
threshold=0.3
clientdata='data/id_client_test.csv'
X_test=pd.read_csv('data/test.csv')
[clientList,df_client]=avail_client(clientdata)
df1 = df_client.join(X_test, how='outer')

@app.route('/personalpage/<name>')
def personalpage(name):
    dff_pred = prediction(int(name),df1,model_file,threshold)
    if dff_pred==1 :
            polarity= " Refusé "
    else:
            polarity= " Accordé "
            
    return render_template("personalpage.html",name=name,polarity=polarity)

@app.route('/Unknownclient/<name>')
def Unknownclient(name):
    return 'Unknown user %s' % name

@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user= request.form["number"]
        if int(user) in clientList:
            return redirect(url_for('personalpage', name = user))
        else:
            return redirect(url_for('Unknownclient', name = user))
            
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)