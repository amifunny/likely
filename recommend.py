# Library to help create recommendation agents.

import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Methods Implemented are :- 
#  1) Multi Armed Bandit



class MultiArmedBandit():

	def __init__(self,item_csv,item_column):
		
		item_df = pd.read_csv(item_csv)
		keywords_list = item_df[item_column].explode().unique() 

		self.keywords = {}
		for i,key in enumerate(keywords_list):
			self.keywords[key] = i

		self.num_arms = len(self.keywords)


		self.a = np.ones( (1,self.num_arms) )
		self.b = np.ones( (1,self.num_arms) )



	def get_info(self):
		info = {}

		self.info['keywords'] = self.keywords
		self.info['num_arms'] = self.num_arms
		self.info['estimates'] = [self.a,self.b]

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


		predictions = []

		for i in range(num_of_preds):
			sampled_q = np.random.beta(self.a[0,:],self.b[0,:])
			key_choice = np.argmax( sampled_q )
			categ_choice = self.keywords[ key_choice ]
			predictions.append( categ_choice )

		set_pred = set(predictions)

		# Get Sample on from dataframe on
		# based on weight colums and value
		item_df = pd.read_csv(item_csv)
		item_df = item_df.assign(inter_index=[len(set(each_genre) & set_pred) for each_genre in item_df.genre])

		picked_ids = prev_watch_ids
		for categ in predictions:

			# Search for rows with 'genre' for given `categ`
			categ_df = item_df[ item_df['genre'].str.join('|').str.contains(categ) ]

			# This excludes already picked indices
			excluded_df = categ_df[~categ_df['movieId'].isin( picked_indices )]

			selected_row = excluded_df.sort_values(['imdb score','inter_index'],ascending=False).iloc(0)
			item_id = item_id['movieId'].values[0]

			picked_ids.append(item_id)


		return predictions,self.get_info()

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

		return self.get_info()
			

class SVD():

	def __init__(self,
				latent_size,
				rating_csv,
				user_column,
				item_column,
				value_column):

		rating_df = pd.read_csv(rating_csv)
		pivot_matrix = rating_df.pivot(index=user_column, columns=item_column, values=value_column).fillna(0)

		all_values = pivot_matrix.values
		user_avg_value = np.mean( all_values , axis=-1 )

		normalised_values = all_values - user_avg_value.reshape(-1,1)

		self.User_Vector,self.Weight,self.Item_Vector = svds( normalised_values , k = latent_size )

	def get_info():

		info = {}
		info['User'] = self.User_Vector
		info['Item'] = self.Item_Vector
		info['Weight'] = self.Weight
		info['Pivot_matrix'] = self.pivot_matrix
		return info

	def get_fit_knn(matrix):

		self.knn_model = NearestNeighbors( metric='cosine' , algorithm='brute',
									  n_neighbors=20, n_jobs=-1)
		
		self.knn_model.fit( matrix )

	def item_based(prev_watch_ids,num_preds):

		items = (self.Item_Vector.T)[prev_watch_ids]

		predictions = []

		num_of_neighbors = 10
		weights,indices = self.knn.kneighbors( items , n_neighbors=num_of_neighbors )
		pred_ids = (pivot_matrix.T).index[ indices[0] ]
		non_watched_idx = list(set(pred_ids) - set(prev_watch_ids))
		predictions.extend( non_watched_idx[:num_preds] )

		return predictions	

	def user_item_based(prev_watch_ids,user_id,num_preds):

		predicted_rating = np.dot( np.dot(self.User_Vector[user_id],self.Weight) , self.Item_Vector )
		predicted_df = pd.DataFrame( predicted_rating.T , index = self.pivot_matrix.columns , columns='rating' )

		excluded_df = predicted_df[ ~predicted_df['movieId'].isin(prev_watch_ids)]
		predicted_ids = excluded_df.sort_values('rating', ascending = False).index.values[-num_preds:]

		return predicted_ids
			

class KNN(object):

	def __init__(self):

		self.knn_model = NearestNeighbors( metric='cosine' , algorithm='brute',
									  n_neighbors=20, n_jobs=-1)
	
	def fit_model(user_csv,
				  user_column,
				  item_csv,
				  item_column,
				  value_column):

		user_df = pd.read_csv(user_csv)
		item_df = pd.read_csv(item_csv)

		pivot_matrix = df.pivot(
							    index=user_column,
							    columns=item_column,
							    values=value_column
							).fillna(0)

		self.csr_matrix = csr_matrix( pivot_matrix.values )
		self.knn_model.fit( csr_matrix )

	def get_info():
		info = {}
		info['knn_model'] = self.knn_model
		info['csr_matrix'] = self.csr_matrix
		return info

	def set_info(info):
		self.knn_model = info['knn_model']
		self.csr_matrix = info['csr_matrix']

	def get_user_similar(user_csv,
						 item_column,
						 value_column,
						 user_id,
						 num_preds,
						 prev_watch_ids):

		num_similar_users = 5
		distance,indexes = knn_model.kneighbors( csr_matrix[user_id] , n_neighbors=num_similar_users )

		user_df = pd.read_csv(user_csv)

		user_df = (user_df[[indexes],:]).sort_values(
					[value_column], ascending=False
				  )[ user_df[value_column]>2.5 ][~user_df[item_column].isin(prev_watch_ids)]

		return user_df.index.values[:num_preds]
























