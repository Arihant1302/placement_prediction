from logging import PlaceHolder
import modle
from flask import Flask, render_template, request
app = Flask(__name__,template_folder='templates')

@app.route('/') # To render Homepage
def home_page():
    return render_template('index.html')

@app.route('/submit',methods=['POST'])
def submit():
    if (request.method=='POST'):
        ssc_s = int(request.form.get("ssc_p"))
        hsc_s = int(request.form.get("hsc_p"))
        deg_s = int(request.form.get("deg_p"))
        etest_s = int(request.form.get("etest_p"))
        prediction = modle.placement(ssc_s,hsc_s,deg_s,etest_s)
        a = []
        a.append([prediction])
        for i in a:
            return "You are " + a[i]
    return render_template('submit.html')

if __name__ == '__main__':
    app.run(debug=True)


