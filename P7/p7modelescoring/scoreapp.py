
from gc import callbacks
import dash
import dash_bootstrap_components as dbc
from components.ini_layouts import ini_layout
from flask import Flask


from flask import render_template
#from flask import current_app as app
from flask import request,render_template,redirect, url_for
from components.data_read import available_clients


app = Flask(__name__)
dash_app = dash.Dash(__name__, server = app,url_base_pathname='/scoringapp/', suppress_callback_exceptions=True,
external_stylesheets =[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.9.0/css/all.css'])


#with app.app_context():
#    from components import routes
@app.route('/personalpage/<name>')
def personalpage(name):
    return redirect('/scoringapp/'+str(name))

@app.route('/Unknownclient/<name>')
def Unknownclient(name):
    return 'Unknown user %s' % name

@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user= request.form["number"]
        if int(user) in available_clients:
            return redirect(url_for('personalpage', name = user))
        else:
            return redirect(url_for('Unknownclient', name = user))
            
    return render_template("login.html")

ini_layout(dash_app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_reloader=True)



