from flask import Flask,render_template,Response,request,redirect
from jinja2 import Template

import json
from flask_session import Session

from recommend import *
import pandas as pd
import numpy as np

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True

app.config['SESSION_TYPE'] = "filesystem"
app.config["SESSION_FILE_DIR"] = "sess"
app.config["SESSION_PERMANENT"] = True
app.secret_key = "secretkey"

Session(app)

@app.after_request
def after_request(response):

	response.headers['Cache-Control'] = 'no-cache,no-store,must-revalidate'
	response.headers['Expires'] = 0
	response.headers['Pragma'] = 'no-cache'
	return response


@app.route("/")
def home():
	# Renders demo home page
	return render_template("home.html")

def get_items( indices ):

	df = pd.read_csv('static/dataset/Movie_data_1k.csv')
	row_list = df.to_numpy()[ indices[0] ].tolist()

	return row_list


# Render categorical items
@app.route("/load_content",methods=["POST","GET"])
def load_content():

	# get list of items using recommendation method
	# items_list = get_recommend_items()

	indices = np.random.randint(0,900,size=(1,20))

	items_list = get_items(indices)

	return render_template("items.html",items_list=items_list)















