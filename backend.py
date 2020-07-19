from flask import Flask,render_template,Response,request,redirect,session
from jinja2 import Template

import json
from flask_session import Session

import pandas as pd
import numpy as np
import json
import pickle
import random

from likely import *

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

# RECOMMENDATION METHOD OPTIONS
# 	0 : MultiArmedBandit
# 	1 : SVD Collaborative Item-User
# 	2 : SVD+KNN Item-Item
# 	3 : KNN User-User

# QUICK INFO :
# We have two csv dataset , one has movie titles, genre etc.(Movie_dataset.csv)
# and another is User rating for some of the movies. (User_dataset.csv)


# Note : This is not utilized in our implementation
# as we just declare the object global,
# Else it would be usefull to store object's state in a
# info file in case of discontinuity of initialization and consumption of object
info_filename = None

# The global object for recommendation ,
# object of one of the classes in 'likely.py'.
recommender_global = None

# root path of primary datasets
root = 'static/dataset'

# Determines num of recommended items to display
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
	# Generate a random id between 2k and 3k
	session['user_id'] = random.randint( 2*(10**3),3*(10**3) )

	# Assign filename which will store recommend object info
	global info_filename
	info_filename =  'static/user/{}_{}_info.pkl'.format(username,method)

	if method!=None:
		initialize_method( method )

	# Create a json file stores all watched,
	# in our case liked or disliked movie
	watch_json = {}
	watch_json[username] = []
	with open('static/user_watch.json','w') as file:
		json.dump(watch_json,file)
		file.close()

	return redirect("/")

# Auxillary method to write to `info_filename`
def save_info(info):
	global info_filename

	file = open( info_filename ,'wb')
	pickle.dump( info , file ) 
	file.close()

# Initialize the matching object from
# given choice
def initialize_method(method):

	global recommender_global

	# Check the 'likely.py' for required arguments
	# Bandit object for 0
	if method==0:
		recommender_global = MultiArmedBandit( root+'Movie_dataset.csv','genre' )
		info = recommender_global.get_info()

	# Both 1 & 2 method uses SVD
	elif method==1 or method==2:
		recommender_global = SVD( 50,
								  root+'User_dataset.csv',
								  'userId',
				   				  'movieId',
				   				  'rating' )
		info = recommender_global.get_info()

	# And KNN for 3
	elif method==3:
		recommender_global = KNN()		 	
	

@app.route("/")
def home():

	username = session.get('username')
	method = session.get('method')

	# Simple intuition message about selected method
	message = get_method_message(method)

	# Renders home page
	return render_template("home.html",username=username,method=method,message=message)

def get_items( ids ):

	# Return movie data like title,genre etc
	# for given movieId values

	df = pd.read_csv(root+'Movie_dataset.csv')
	df = df[ df['movieId'].isin(ids) ]

	# Our dataset has some duplicates,
	# I know i should have cleaned it but lets go
	# with it.
	# Drop any rows with duplicate values of 'movieId'
	df_unique = df.drop_duplicates(subset = ["movieId"])
	# Convert it into list of dictionaries
	row_list = df_unique.to_dict('records')

	return row_list

def get_user_watch_ids():

	# Return all and positive movieIds of 
	# watch movie ny our current session user

	with open('static/user_watch.json','r') as file:
		watch_obj = json.load(file) 
		file.close()

	watch_array = np.array( watch_obj[ session.get('username') ] )

	# If empty, means its a new user, avoid futher
	# processing and return empty lists
	if watch_array.size==0:
		return watch_array,[]

	# Hash table to store unique movie values
	moviehash = {}
	# This is done to avoid duplicate rating for same movie,
	# this has to employed as we don't perform any checks at time of feedback
	for movie_tup in watch_array:
		# at position 1 is movieId
		moviehash[ movie_tup[1] ] = movie_tup

	# Get back unique value as list of lists
	unique_watch_array = np.array( list( (moviehash.values()) ) )

	all_user_watch_ids = unique_watch_array[:,1]
	all_user_watch_ids = all_user_watch_ids.astype(np.int) 
	
	# Positive ids are the one with perfect 5.0 rating
	pos_user_watch_ids = unique_watch_array[unique_watch_array[:,2]=="5.0"]
	pos_user_watch_ids = pos_user_watch_ids[:,1]
	pos_user_watch_ids = pos_user_watch_ids.astype(np.int) 

	return all_user_watch_ids.tolist(),pos_user_watch_ids.tolist()

def make_temp_csv():

	# Create new CSV file with info of current
	# session user

	with open('static/user_watch.json','r') as file:
		watch_obj = json.load(file) 
		file.close()

	# Get all watched movies and rating in list of lists format
	watch_array = np.array( watch_obj[ session.get('username') ] )

	user_df = pd.read_csv(root+'User_dataset.csv')
	# Convert into dataframe
	current_df = pd.DataFrame( watch_array , columns=['userId','movieId','rating','timestamp'] )
	# Append to already stored ratings datset
	new_df = user_df.append( current_df , ignore_index=True )
	new_df.to_csv('static/temp/Current_User_dataset.csv',index=False)

	return

def use_recommendation(num_preds):

	# Function that uses global recommender object
	# to get recommendation for session user

	global recommender_global

	method = session.get( 'method' )
	user_id = session.get( 'user_id' )

	# Get all watch history
	user_watch_ids,pos_user_watch_ids = get_user_watch_ids()

	# If there are no watch ids means its a new user,
	# So we use display random sampled movies, as SVD and KNN will not work
	# though Bandit can still give recommendation , but
	# are as good as random
	if len(user_watch_ids)==0:
		movie_df = pd.read_csv(root+'Movie_dataset.csv')
		indices = movie_df.sample(num_preds)['movieId'].values
		return indices

	# Create a CSV if not empty
	make_temp_csv()

	# See 'likely.py' for neccesary arguments.
	# All these if-else statements ar just selecting,
	# correct functions to call, get recommendation ids
	# i.e `predicted_indices`

	if method==0:

		predicted_indices = recommender_global.recommend(
				num_preds,
				root+'Movie_dataset.csv',
				'movieId',
				'genre',
				'imdb score',
				user_watch_ids,
				order='DESC')
	elif method==1:
		latent_size = 50
		recommender_global = SVD(latent_size,
								'static/temp/Current_User_dataset.csv',
								'userId',
		   						'movieId',
		   						'rating')
		predicted_indices = recommender_global.user_item_based(
							user_watch_ids,
							user_id,
							num_preds)
	elif method==2:

		if len(pos_user_watch_ids)==0:
			movie_df = pd.read_csv(root+'Movie_dataset.csv')
			indices = movie_df.sample(num_preds).index
			return indices

		recommender_global = SVD(50,
								'static/temp/Current_User_dataset.csv',
								'userId',
		   						'movieId',
		   						'rating')

		recommender_global.get_fit_knn( recommender_global.Item_Vector.transpose() )
		predicted_indices = recommender_global.item_based(
							pos_user_watch_ids,
							num_preds
							)
	elif method==3:

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

# Render items requested async
@app.route("/load_content",methods=["POST","GET"])
def load_content():

	# Get recommended indices
	indices = use_recommendation(NUM_PREDS)

	# Get list of movie info of given `indices
	items_list = get_items(indices)

	return render_template("items.html",items_list=items_list)


@app.route('/feedback',methods=["POST","GET"])
def feedback():

	# Stores the user generated feedback,
	# which is either dislike or like,
	# But as we used dataset for rating between 0 to 5.0
	# We convert like to ==> 5.0
	# and dislike to ==> 0.0

	method = session.get('method')

	movie_id = int(request.form.get('id'))
	# like is feedback_int == 1
	# dislike is feedback_int == 0
	feedback_int = int(request.form.get('feedback'))

	feedback_rating = feedback_int*5.0

	# Read already stored data
	with open('static/user_watch.json', 'r') as file: 
		user_obj = json.load(file) 
		watch_array = user_obj[session.get('username')]
		file.close()

	# Append to previous data and save back again,
	# I didn't found any method to do read&write in one go.
	with open('static/user_watch.json', 'w') as file: 
		watch_array.append( [session.get('user_id'),movie_id,feedback_rating,""] )
		user_obj[session.get('username')] = watch_array
		json.dump(user_obj,file)
		file.close()

	# Now bandit method requires special treatment,
	# and we update its parameters according to reward.
	# I know we could have taken care of this when doing recommendation
	# like other method i.e doing updates in one go
	 # but lets stick to intuitive way of doing Bandits
	if method==0:
		BanditFeedback(movie_id,feedback_int)

	return "success"	

def BanditFeedback(movie_id,feedback_int):

	# Read all keyword from our auxillary file 'keywords'
	# It is just unique values of 'genre' column of movies dataset 
	# joined together with "|" string
	with open('static/keyword.txt','r') as file:
		keyword_text = file.read()
		file.close()	

	keywords = sorted( keyword_text.split('|') )

	movie_df = pd.read_csv( root+'Movie_dataset.csv' , converters={'genre': eval} )
	# Get genre of given movieId
	genre_list = movie_df[movie_df['movieId']==movie_id].iloc[0]['genre']

	# Get index of feedback genre
	feedback_categ = []
	for each_categ in genre_list:
		index = keywords.index( each_categ )
		feedback_categ.append( index )

	# Update that category of keyword
	if feedback_int==1:
		recommender_global.feedback(positive_category=feedback_categ)
	else:
		recommender_global.feedback(negative_category=feedback_categ)

	return

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















