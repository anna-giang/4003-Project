from flask import Blueprint, render_template, request, redirect, url_for

from prototype.ruleBasedLearning.ruleBasedLearning import *


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
        results = process(job_ad)

        # number is the count of how many found
        data = {'diversity': 1, 'list': 0, 'workHr': results[0], 'roster': results[2],
                'encourage': 0, 'tech': results[5], 'fem': results[4], 'masc': results[3], 'attributes': results[1]}
        # calculate how many recommendations satisfied %
        perc = 30
        return render_template("result.html", result=data, perc=perc)
    return render_template("tool.html")


@views.route('/about')
def about():
    return render_template("about.html", text="passing data")


@views.route('/result')
def result():
    return render_template("result.html")
