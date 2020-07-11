from flask import Flask,render_template,Response,request,redirect,session
from jinja2 import Template

import json
from flask_session import Session

from recommend import *
import pandas as pd
import numpy as np
import json

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

@app.route("/start_session",methods=["POST","GET"])
def start_session():

	# When new user is created,
	# clear old session
	session.clear()
	# Get given recommendation method
	method = request.form.get('recommend_method')
	# username to store interaction data
	username = request.form.get('username')

	# store in session
	session['method'] = method
	session['username'] = username

	with open('static/keyword.txt','r') as file:
		keyword_text = file.read()
		file.close()	

	keywords = keyword_text.split('|')

	user_dict = {}

	# ** "likely" system started
	# Selecting `MultiArmedBandit`
	lk_agent = MultiArmedBandit( len(keywords) )
	info = lk_agent.get_info()
	user_dict[ username ] = [info[0].tolist(),info[1].tolist()]

	with open('static/userdata.json','w') as file:
		json.dump(user_dict,file)
		file.close()	
	
	return redirect("/")


@app.route("/")
def home():

	username = session.get('username')
	method = session.get('method')

	# Renders demo home page
	return render_template("home.html",username=username,method=method)

def get_items( indices ):

	df = pd.read_csv('static/dataset/Movie_dataset.csv')
	row_list = df.to_numpy()[ indices ].tolist()

	return row_list

def get_recommend_categ():
	
	with open('static/userdata.json', 'r') as file: 
		# Reading from json file 
		user_obj = json.load(file) 
		file.close()

	username = session['username']

	lk_agent = MultiArmedBandit( len(user_obj[username][0]) , user_obj[username] )	
	preds , user_dict = lk_agent.recommend(20)

	return preds

def get_keywords(categ_list_int):

	with open('static/keyword.txt','r') as file:
		keyword_text = file.read()
		file.close()	

	keywords = keyword_text.split('|')

	categ_keys = [ keywords[i] for i in categ_list_int ]
	return categ_keys

def get_recommended_indices(categ_list):

	df = pd.read_csv('static/dataset/Movie_data_1k.csv')
	shuffle_df = df.sample(frac=1)

	picked_indices = []

	for categ in categ_list:

		# Search for rows with 'Genre' as given `categ`
		categ_df = shuffle_df[ shuffle_df['genre'].str.contains( categ ) ] 
		# This excludes already picked indices
		excluded_df = categ_df[~categ_df.index.isin( picked_indices )]
		idx = excluded_df.index[0]

		picked_indices.append(idx)	

	return picked_indices


# Render categorical items
@app.route("/load_content",methods=["POST","GET"])
def load_content():

	# get list of "likely" categories as int using recommendation method
	categ_list_int = get_recommend_categ()

	# convert recommended category int into word for searching
	categ_list = get_keywords( categ_list_int )

	# search for keyword in dataset and return there index ids
	indices = get_recommended_indices(categ_list)

	# Fetch complete info of `indices`
	items_list = get_items(indices)

	return render_template("items.html",items_list=items_list)


@app.route('/feedback',methods=["POST","GET"])
def feedback():

	# Obtaining feedback from frontend
	categ = request.form.get('categ')
	feedback_int = request.form.get('feedback')

	# Read all keyword from file
	with open('static/keyword.txt','r') as file:
		keyword_text = file.read()
		file.close()	

	# Make a list of it
	keywords = keyword_text.split('|')


	# Get index of feedback category
	feedback_categ = []
	for each_categ in categ.split("|"):
		index = keywords.index( each_categ )
		feedback_categ.append( index )

	# Read stored data of user to manipulate
	with open('static/userdata.json', 'r') as file: 
		# Reading from json file 
		user_obj = json.load(file) 
		file.close()

	# Kick start agent
	username = session.get('username')
	lk_agent = MultiArmedBandit( len(keywords) , user_obj[username] )

	# Based on 'feedback_int' decide if it is positive or negative
	if feedback_int=="1":
		
		user_info = lk_agent.feedback(positive_category=feedback_categ)

	else:
				
		user_info = lk_agent.feedback(negative_category=feedback_categ)


	# store the returned value
	user_obj = {}
	user_obj[ username ] = [user_info[0].tolist(),user_info[1].tolist()]

	# write it into file
	with open('static/userdata.json','w') as file:
		json.dump(user_obj,file)
		file.close()	


	return "success"







