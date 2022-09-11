from flask import Blueprint, render_template

views = Blueprint('views', __name__)


@views.route('/')
def home():
    'this will run when on the / route for website'
    return render_template("home.html")


@views.route('/tool')
def tool():
    'this will run when on the / route for website'
    return render_template("tool.html")


@views.route('/about')
def about():
    'this will run when on the / route for website'
    return render_template("about.html", text="passing data")
