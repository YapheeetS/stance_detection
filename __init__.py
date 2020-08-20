

from flask import Flask,request
from main import *

app = Flask(__name__)
app.secret_key = 'Israa_Qasim_89'

# if the button "showResults" is clicked , redirect the user to the results page carrying the textfield value in the request object


# direct the user to the main page when the app starts
@app.route("/", methods=['GET', 'POST'])
def checkClaim():
    print("recieved a request")
    return "Hi"
    #claim = request.form['claim']
    # claim = request.args['claim']
    # report = fact_check(claim)
    # return report

if __name__=="__main__" :
    app.run(host='0.0.0.0')






