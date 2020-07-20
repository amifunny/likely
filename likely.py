import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import ast

# Classes Implemented are :- 
#  1) Multi Armed Bandit
#  2) SVD
#  3) KNN

class MultiArmedBandit():

	def __init__(self,item_csv,key_column):
		
		# item_csv : contains info like genre or keywords
		# about items 
		# key_column : Column with genre

		item_df = pd.read_csv(item_csv)
		keywords_list = item_df[key_column].apply(lambda x: ast.literal_eval(x))
		keywords_list = keywords_list.explode().unique() 

		self.keywords = sorted(keywords_list)

		# `num_arms` are categories that our bandit can choose from
		self.num_arms = len(self.keywords)

		# `a` and `b` are parameters of beta distribution,
		# and not some naive naming
		self.a = np.ones( (1,self.num_arms) )
		self.b = np.ones( (1,self.num_arms) )


	def get_info(self):
		
		info = {}

		info['keywords'] = self.keywords
		info['num_arms'] = self.num_arms
		info['estimates'] = [self.a,self.b]

		return info

	def set_info(self,info):
		self.keywords = info['keywords']

		self.num_arms = info['num_arms']

		self.a = np.array( info['estimates'][0] )
		self.b = np.array( info['estimates'][1] )

	def get_avg_estimates():
		# Returns avg value of each arm/category
		return self.a/(self.a+self.b)

	def recommend(self,
				num_preds,
				item_csv,
				item_column,#movieId
				keyword_column,#genre
				weight_column,#imdb
				prev_watch_ids,
				order='DESC'):


		# Store keyword sampled
		predictions = []

		for i in range(num_preds):
			# Sample q-values
			sampled_q = np.random.beta(self.a[0,:],self.b[0,:])
			# Get index of highest q-value predicted
			key_choice = np.argmax( sampled_q )
			# Get string ofthat keyword
			categ_choice = self.keywords[ key_choice ]
			predictions.append( categ_choice )

		set_pred = set(predictions)

		# Get Sample on from dataframe on
		# based on weight columns and value
		item_df = pd.read_csv(item_csv)
		# convert "stringy" list of genre into proper list
		item_df[keyword_column] = item_df[keyword_column].apply(lambda x: ast.literal_eval(x))
		
		# It assign a counter for number of predicted keywords
		# in any row.
		# For example, if predicted keys are ["Mystery","Horror"],
		# 'inter_index' give high values to rows tih both the genre in it
		item_df = item_df.assign(
					inter_index=[len(set(each_genre) & set_pred) for each_genre in item_df.genre]
				)

		picked_ids = prev_watch_ids
		pred_ids = []
		# To add element of randomness, 
		# we shuffle top given number of rows
		shuffle_limit = 50
		for categ in predictions:
			# Search for rows with 'genre' for given `categ`
			categ_df = item_df[ item_df['genre'].str.join('|').str.contains(categ) ]

			# This excludes already picked indices
			excluded_df = categ_df[~categ_df['movieId'].isin( picked_ids )]

			selected_rows = excluded_df.sort_values(['imdb score','inter_index'],ascending=False)
			selected_rows = selected_rows[:shuffle_limit].sample(frac=1)
			
			if len(selected_rows.index)!=0:
				# Choose ony first that bubble up 
				# after shuffling
				selected_row = selected_rows.iloc[0]
				item_id = selected_row['movieId']
				pred_ids.append(item_id)
				picked_ids.append(item_id)


		return pred_ids

	def feedback(self,
				positive_category=None,
				positive_reward=1.0,
				negative_category=None,
				negative_reward=0.0,
				assume_negative=False
				):

		"""	
			Args :
				positive_category : (type list) Categories obtained from user interaction like 
					clicks to act as positive feedback to update `arm_estimates`
				positive_reward : (float [Default : 1.0]) Reward to assume for `positive_category` 
					should between 0 and 1
				negative_category : (type list) Category not interacted by user or disliked
					to act as negative feedback to update `arm_estimates`
				negative_reward : (float [Default : 1.0]) Reward to assume for `positive_category`
					should between 0 and 1
				assume_negative : Assumes that all categories except positive one are
					considered as negative and given default negative reward. To be
					used in cases when one category was selected other were implicitly
					rejected.
		"""

		# clip rewards to be in [0,1]
		positive_reward = np.clip(positive_reward,0.0,1.0)
		negative_reward = np.clip(negative_reward,0.0,1.0)


		if assume_negative==True:

			# `assumed_matrix` has reward as 0 for each non-positive category
			assumed_matrix = np.zeros( (positive_category.shape[0],self.num_arms) )
			assumed_matrix[ positive_category ] = 1.0

			sum_reward = np.sum( assumed_matrix , 0 )
			self.a += sum_reward
			self.b += 1 - sum_reward

		else:

			if positive_category is not None:
				self.a[0,positive_category] += positive_reward
				self.b[0,positive_category] += 1 - positive_reward

			if negative_category is not None:
				self.a[0,negative_category] += negative_reward
				self.b[0,negative_category] += 1 - negative_reward

		return
			

class SVD():

	def __init__(self,
				latent_size,
				rating_csv,
				user_column,
				item_column,
				value_column):

		rating_df = pd.read_csv(rating_csv)
		# Create a pivot matrix with rating as values
		pivot_matrix = rating_df.pivot(index=user_column, columns=item_column, values=value_column).fillna(0)

		self.pivot_matrix = pivot_matrix

		all_values = pivot_matrix.values
		user_avg_value = np.mean( all_values , axis=-1 )

		# Subtract the average rating of user from each rating
		normalised_values = all_values - user_avg_value.reshape(-1,1)

		# Standard Singular Vector Decomposition (SVD)
		self.User_Vector,self.Weight,self.Item_Vector = svds( normalised_values , k = latent_size )

	def get_info(self):

		info = {}
		info['User'] = self.User_Vector
		info['Item'] = self.Item_Vector
		info['Weight'] = self.Weight
		info['Pivot_matrix'] = self.pivot_matrix
		return info

	def get_fit_knn(self,matrix,num_neighbors=20):

		self.knn_model = NearestNeighbors( metric='cosine' , algorithm='brute',
									  n_neighbors=num_neighbors, n_jobs=-1)
		self.knn_model.fit( matrix )

	def item_based(self,all_prev_watch_ids,pos_prev_watch_ids,num_preds):

		# When dealing with pivot matrix,
		# we need indexes of given ids
		all_movie_ids = self.pivot_matrix.columns
		item_idx = []
		for i,movie_id in enumerate(all_movie_ids):
			if movie_id in pos_prev_watch_ids:
				item_idx.append(i)

		items = (self.Item_Vector.transpose())[item_idx]

		predictions = []

		# If num of movies watched are few , we use 
		# fetch more neighbor and not otherwise
		if len(pos_prev_watch_ids)*(num_preds/2)<num_preds:
			num_of_neighbors = num_preds
		else:
			num_of_neighbors = int(num_preds/2)

		print(num_of_neighbors)

		# Get indices of similar
		weights,indices = self.knn_model.kneighbors( items , n_neighbors=num_of_neighbors )
		print(indices)
		# Convert indices into ids
		pred_ids = ( self.pivot_matrix.transpose() ).index[ indices.flatten() ]
		# exclude watched ids
		non_watched_idx = list(set(pred_ids) - set(all_prev_watch_ids))

		# Shuffle to add element of randomness
		print(non_watched_idx)
		np.random.shuffle(non_watched_idx)
		print(non_watched_idx)
		predictions.extend( non_watched_idx[:num_preds] )

		return predictions	

	def user_item_based(self,prev_watch_ids,user_id,num_preds):

		pivot_indices = self.pivot_matrix.index.values.tolist()
		user_idx = pivot_indices.index( user_id )

		# Using vectors got from SVd operation,
		# we can use simple dot product to get predicted ratings
		# by a user for items
		predicted_rating = np.dot( 
						   np.dot( np.reshape( self.User_Vector[user_idx] , (1,-1) ) , np.diag( self.Weight ) )
						   , self.Item_Vector )
		predicted_df = pd.DataFrame( predicted_rating.transpose() , 
						index = self.pivot_matrix.columns.values.tolist() , columns=['rating'] )

		# Exclude watched id rows
		excluded_df = predicted_df[ ~predicted_df.index.isin(prev_watch_ids)]
		# Note : 'rating' here is predicted and not determined one
		predicted_ids = excluded_df.sort_values('rating', ascending = False).index.values[:num_preds]

		return predicted_ids
			

class KNN(object):

	def __init__(self,num_neighbors=20):

		self.knn_model = NearestNeighbors( metric='cosine' , algorithm='brute',
									  n_neighbors=num_neighbors, n_jobs=-1)
	
	def fit_model(self,
				  user_csv,
				  user_column,
				  item_csv,
				  item_column,
				  value_column):

		user_df = pd.read_csv(user_csv)
		item_df = pd.read_csv(item_csv)

		pivot_matrix = user_df.pivot(
							    index=user_column,
							    columns=item_column,
							    values=value_column
							).fillna(0)

		# Get userIds for index to id conversion
		self.user_ids = pivot_matrix.index.values
		# Convert Sparse matrix consisting mostly zeros
		# into CSR matrix
		self.csr_mat = csr_matrix( pivot_matrix.values )
		self.knn_model.fit( self.csr_mat )

	def get_info(self):
		info = {}
		info['user_ids'] = self.user_ids
		info['knn_model'] = self.knn_model
		info['csr_matrix'] = self.csr_mat
		return info

	def set_info(self,info):
		self.user_ids = info['user_ids']
		self.knn_model = info['knn_model']
		self.csr_mat = info['csr_matrix']

	def get_user_similar(self,
						 user_csv,
						 item_column,
						 value_column,
						 user_id,
						 num_preds,
						 prev_watch_ids,
						 num_similar_users = 10):

		# Convert Current user_id into corresponding index
		current_user_idx = self.user_ids.tolist().index( user_id )
		
		# Get similar users' index
		distance,indexes = self.knn_model.kneighbors( self.csr_mat[current_user_idx] ,
													  n_neighbors=num_similar_users )

		# Convert Indices into Ids for users
		similar_user_ids = [ self.user_ids[i] for i in indexes[0] ]			

		user_df = pd.read_csv(user_csv)

		# Read user rated movies
		user_df = user_df[ user_df['userId'].isin( similar_user_ids ) ]
		# And return Top rated movies by users
		user_df = user_df.sort_values(
					[value_column], ascending=False
				  )[ user_df[value_column]>2.5 ][~user_df[item_column].isin(prev_watch_ids)]

		# Skim out Ids double the amount we require
		movie_ids = user_df['movieId'].values[:num_preds*2] 
		# Shuffle to add randomness
		np.random.shuffle(movie_ids)
		return movie_ids[:num_preds]
























