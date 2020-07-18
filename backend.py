from flask import Flask,render_template,Response,request,redirect,session
from jinja2 import Template

import json
from flask_session import Session

from recommend import *
import pandas as pd
import numpy as np
import json
import pickle
import random

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
root = 'static/dataset/'
NUM_PREDS = 20

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
	session['user_id'] = random.randint( 2*(10**3),3*(10**3) )

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
		recommender_global = SVD(50,root+'User_dataset.csv','userId',
				   'movieId','rating')
		info = recommender_global.get_info()
	elif method==3:
		print("SAHI aaya")
		recommender_global = KNN()		 	
		# info = recommender_global.get_info()
	
	# save_info(info)

@app.route("/")
def home():

	username = session.get('username')
	method = session.get('method')

	message = get_method_message(method)

	# Renders demo home page
	return render_template("home.html",username=username,method=method,message=message)

def get_items( indices ):

	df = pd.read_csv(root+'Movie_dataset.csv')
	df = df[ df['movieId'].isin(indices) ]
	df_unique = df.drop_duplicates(subset = ["movieId"])
	row_list = df_unique.to_dict('records')

	return row_list

def get_user_watch_ids():

	with open('static/user_watch.json','r') as file:
		watch_obj = json.load(file) 
		file.close()

	watch_array = np.array( watch_obj[ session.get('username') ] )

	if watch_array.size==0:
		return watch_array,[]

	moviehash = {}
	for movie_tup in watch_array:
		moviehash[ movie_tup[1] ] = movie_tup

	unique_watch_array = np.array( list( (moviehash.values()) ) )

	all_user_watch_ids = unique_watch_array[:,1]
	all_user_watch_ids = all_user_watch_ids.astype(np.int) 
	
	pos_user_watch_ids = unique_watch_array[unique_watch_array[:,2]=="5.0"]
	pos_user_watch_ids = pos_user_watch_ids[:,1]
	pos_user_watch_ids = pos_user_watch_ids.astype(np.int) 

	return all_user_watch_ids.tolist(),pos_user_watch_ids.tolist()

def make_temp_csv():

	with open('static/user_watch.json','r') as file:
		watch_obj = json.load(file) 
		file.close()

	watch_array = np.array( watch_obj[ session.get('username') ] )

	user_df = pd.read_csv(root+'User_dataset.csv')
	current_df = pd.DataFrame( watch_array , columns=['userId','movieId','rating','timestamp'] )
	new_df = user_df.append( current_df , ignore_index=True )
	new_df.to_csv('static/temp/Current_User_dataset.csv',index=False)

	return watch_array

def use_recommendation(num_preds):

	global recommender_global

	method = session.get( 'method' )
	user_id = session.get( 'user_id' )

	user_watch_ids,pos_user_watch_ids = get_user_watch_ids()

	if len(user_watch_ids)==0:
		movie_df = pd.read_csv('static/dataset/Movie_dataset.csv')
		indices = movie_df.sample(num_preds)['movieId'].values
		return indices

	make_temp_csv()

	if method==0:

		predicted_indices = recommender_global.recommend(
				num_preds,
				root+'Movie_dataset.csv',
				'movieId',#movieId
				'genre',#genre
				'imdb score',#imdb
				user_watch_ids,
				order='DESC')
	elif method==1:
		recommender_global = SVD(50,'static/temp/Current_User_dataset.csv','userId',
		   'movieId','rating')
		predicted_indices = recommender_global.user_item_based(
							user_watch_ids,
							user_id,
							num_preds
							)
	elif method==2:

		if len(pos_user_watch_ids)==0:
			movie_df = pd.read_csv('static/dataset/Movie_dataset.csv')
			indices = movie_df.sample(num_preds).index
			return indices

		recommender_global = SVD(50,'static/temp/Current_User_dataset.csv','userId',
		   'movieId','rating')
		recommender_global.get_fit_knn( recommender_global.Item_Vector.transpose() )
		predicted_indices = recommender_global.item_based(
							pos_user_watch_ids,
							num_preds
							)
	elif method==3:
		print("SAHI aaya")
		recommender_global.fit_model(
				'static/temp/Current_User_dataset.csv',
				'userId',
				'static/temp/Current_User_dataset.csv',
				'movieId',
				'rating'
			)
		predicted_indices = recommender_global.get_user_similar(
			'static/temp/Current_User_dataset.csv',
			'movieId',
			'rating',
			 user_id,
			 num_preds,
			 user_watch_ids
			)		 	

	return predicted_indices

# Render categorical items
@app.route("/load_content",methods=["POST","GET"])
def load_content():

	indices = use_recommendation(NUM_PREDS)

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
		watch_array.append( [session.get('user_id'),movie_id,feedback_rating,""] )
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

	keywords = sorted( keyword_text.split('|') )

	movie_df = pd.read_csv( 'static/dataset/Movie_dataset.csv' , converters={'genre': eval} )
	genre_list = movie_df[movie_df['movieId']==movie_id].iloc[0]['genre']

	# Get index of feedback category
	feedback_categ = []
	for each_categ in genre_list:
		index = keywords.index( each_categ )
		feedback_categ.append( index )

	if feedback_int==1:
		user_info = recommender_global.feedback(positive_category=feedback_categ)
	else:
		user_info = recommender_global.feedback(negative_category=feedback_categ)

	save_info(user_info)


# -----------------------------------------------------------------------------------------
# `get_method_message()` shows gives summary message
# about the chosen method
def get_method_message(method):

	msg = ""
	if method==0:
		msg = ("MultiArmedBandit is non-contextual and give results "
			   "using most rewarding keyword. In this case, Movie Genre."
			   "Observe the genre you click more, will appear more in recommendation.")
	elif method==1:
		msg = ("SVD uses User and Item pattern to get latent factor. "
			   "These factors help us predict for new users. "
			   "Recommendation you observe are from similar pattern of other user.")
	elif method==2:
		msg = ("SVD gives us latent factors of each Items(i.e Movies). "
			   "Then Recommendations are based on Nearest Neighbors of items(i.e. Movie Liked) "
			   "by current user(you).")
	elif method==3:
		msg = ("Based on your movie pattern, finds similar users "
			   "and movie rated good by them are Recommended.")


	return msg















