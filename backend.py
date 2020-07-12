from flask import Flask,render_template,Response,request,redirect,session
from jinja2 import Template

import json
from flask_session import Session

from recommend import *
import pandas as pd
import numpy as np
import json
import pickle

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

# rec_switch = {
# 	0:'MultiArmedBandit'
# 	1:'SVD Collaborative Item-User'
# 	2:'SVD+KNN Item-Item'
# 	3:'KNN User-User'
# }

info_filename = None
recommender_global = None

@app.route("/start_session",methods=["POST","GET"])
def start_session():

	# When new user is created,
	# clear old session
	session.clear()
	# Get given recommendation method
	method = int( request.form.get('recommend_method') )
	# username to store interaction data
	username = request.form.get('username')

	# store in session
	session['method'] = method
	session['username'] = username

	global info_filename
	info_filename =  'static/user/{}_{}_info.pkl'.format(username,method)

	if method!=None:
		initialize_method( method )

	watch_json = {}
	watch_json[username] = []
	with open('static/user_watch.json','w') as file:
		json.dump(watch_json,file)
		file.close()

	return redirect("/")

def save_info(info):
	global info_filename

	file = open( info_filename ,'wb')
	pickle.dump( info , file ) 
	file.close()

def initialize_method(method):

	global recommender_global

	if method==0:
		recommender_global = MultiArmedBandit('static/dataset/Movie_dataset.csv','genre')
		info = recommender_global.get_info()
	elif method==1 or method==2:
		info = SVD(50,'dataset/User_dataset.csv','userId',
				   'movieId','rating')
	elif method==3:
		info = KNN()		 	
	
	save_info(info)

@app.route("/")
def home():

	username = session.get('username')
	method = session.get('method')

	# Renders demo home page
	return render_template("home.html",username=username,method=method)

def get_items( indices ):

	df = pd.read_csv('static/dataset/Movie_dataset.csv')
	row_list = df.loc[ indices , : ].to_dict('records')

	return row_list

# def get_recommend_categ():
	
# 	with open('static/userdata.json', 'r') as file: 
# 		# Reading from json file 
# 		user_obj = json.load(file) 
# 		file.close()

# 	username = session['username']

# 	lk_agent = MultiArmedBandit( len(user_obj[username][0]) , user_obj[username] )	
# 	preds , user_dict = lk_agent.recommend(20)

# 	return preds

# def get_keywords(categ_list_int):

# 	with open('static/keyword.txt','r') as file:
# 		keyword_text = file.read()
# 		file.close()	

# 	keywords = keyword_text.split('|')

# 	categ_keys = [ keywords[i] for i in categ_list_int ]
# 	return categ_keys

# def get_recommended_indices(categ_list):

# 	df = pd.read_csv('static/dataset/Movie_data_1k.csv')
# 	shuffle_df = df.sample(frac=1)

# 	picked_indices = []

# 	for categ in categ_list:

# 		# Search for rows with 'Genre' as given `categ`
# 		categ_df = shuffle_df[ shuffle_df['genre'].str.contains( categ ) ] 
# 		# This excludes already picked indices
# 		excluded_df = categ_df[~categ_df.index.isin( picked_indices )]
# 		idx = excluded_df.index[0]

# 		picked_indices.append(idx)	

# 	return picked_indices

def use_recommendation():

	method = session.get( 'method' )

	file = open('info_filename','wb')
	pickle.dump( info , file )

	with open('user_watch.json','r') as file:
		watch_obj = json.load(file) 
		file.close()

	watch_indices = watch_obj[ session.get('username') ]

	# if method==0:
		
	# elif method==1:

	# elif method==2:
	
	# elif method==3:
		 	

	return predicted_indices

# Render categorical items
@app.route("/load_content",methods=["POST","GET"])
def load_content():

	indices = use_recommendation()

	# Fetch complete info of `indices`
	items_list = get_items(indices)

	return render_template("items.html",items_list=items_list)


@app.route('/feedback',methods=["POST","GET"])
def feedback():

	method = session.get('method')

	movie_id = int(request.form.get('id'))
	feedback_int = int(request.form.get('feedback'))

	# Convert like into 5 star rating
	# and dislike into 0 star rating
	feedback_rating = feedback_int*5.0

	with open('static/user_watch.json', 'r') as file: 
		user_obj = json.load(file) 
		watch_array = user_obj[session.get('username')]
		file.close()

	with open('static/user_watch.json', 'w') as file: 
		watch_array.append( [movie_id,feedback_rating] )
		user_obj[session.get('username')] = watch_array
		json.dump(user_obj,file)
		file.close()

	if method==0:
		BanditFeedback(movie_id,feedback_int)

	return "success"	

def BanditFeedback(movie_id,feedback_int):

	# Read all keyword from file
	with open('static/keyword.txt','r') as file:
		keyword_text = file.read()
		file.close()	

	keywords = keyword_text.split('|')

	movie_df = pd.read_csv( 'static/dataset/Movie_dataset.csv' , converters={'genre': eval} )
	genre_list = movie_df[movie_df['movieId']==movie_id].iloc[0]['genre']

	# Get index of feedback category
	feedback_categ = []
	for each_categ in genre_list:
		print(each_categ)
		index = keywords.index( each_categ )
		feedback_categ.append( index )

	if feedback_int==1:
		user_info = recommender_global.feedback(positive_category=feedback_categ)
	else:
		user_info = recommender_global.feedback(negative_category=feedback_categ)

	save_info(user_info)

