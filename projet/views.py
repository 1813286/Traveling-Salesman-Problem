from flask import Blueprint, render_template, request
from heuristicTsp.TspHeuristic import TspHeuristic
from dynamicTsp.TspDynamic import Tsp
views = Blueprint(__name__, "views")


@views.route("/")
def Home():
    return render_template("home.html")


@views.route("/Dynamic", methods=['POST'])
def DynamicProgramme():
    # code of dynamic programming
    instance = request.form['checked_instance']
    result = Tsp(instance)

    res = [r+1 for r in result['result']  ]

    return render_template('Dynamique.html', result=res, cost=result['cost'], Mytime=result['Mytime'])


@views.route("/Heuristic", methods=['POST'])
def HeuristicProgramme():
    # code of dynamic programming
    instance = request.form['checked_instance2']
    result = TspHeuristic(instance)
    res = [r+1 for r in result['result']]
    return render_template('Heuristique.html', result=res, cost=result['cost'], Mytime=result['Mytime'])


@views.route("/calcule")
def CalculePage():
    return render_template('index.html')


@views.route("/contact")
def ContactPage():
    return render_template('contact.html')
