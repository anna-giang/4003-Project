from flask import Blueprint, render_template, request, redirect, url_for

views = Blueprint('views', __name__)


@views.route('/')
def home():
    # this will run when on the / route for website
    return render_template("home.html")


@views.route('/tool', methods=['GET', 'POST'])
def tool():
    # when user hits submit - post request to result page
    if request.method == 'POST':
        job_ad = request.form.get('job-ad')
        # -- do we have any requirements for text like certain length?
        # do some stuff with the job ad
        # ---
        # the 1 or 0 is true/false if recommendation met
        result = {'diversity': 1, 'list': 0, 'workHr': 1, 'roster': 0,
                  'encourage': 0, 'tech': 1, 'lang': 0, 'attributes': 0}
        # calculate how many recommendations satisfied %
        perc = 30
        return render_template("result.html", result=result, perc=perc)
    return render_template("tool.html")


@views.route('/about')
def about():
    return render_template("about.html", text="passing data")


@views.route('/result')
def result():
    return render_template("result.html")
