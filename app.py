# import flast module
from flask import Flask,render_template

# instance of flask application
app = Flask(__name__)

# home route that returns below text when root url is accessed
@app.route("/")

def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/admin')
def owner():
    return render_template("admin.html")

if __name__ == '__main__':  
   app.run()  
