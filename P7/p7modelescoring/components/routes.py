"""Routes for parent Flask app."""
from flask import render_template
from flask import current_app as app
from flask import request,render_template,redirect, url_for
from components.data_read import available_clients


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































    