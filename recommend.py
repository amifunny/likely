# Library to help create recommendation agents.

import numpy as np

# Methods Implemented are :- 
#  1) Multi Armed Bandit



class MultiArmedBandit():

	def __init__(self,num_arms,init_estimates=None):
		
        # Number of categories to consider to recommend
		self.num_arms = num_arms

		# If no `init_estimates` are provided, Use 1.0
		# estimate for each arm which is optimistic
		# for exploration
		if init_estimates is None:
			self.a = np.ones( (1,num_arms) )
			self.b = np.ones( (1,num_arms) )

		else:
			self.a = np.array( init_estimates[0] )
			self.b = np.array( init_estimates[1] )


	def get_info(self):
		# Return parameters info or value
		return [self.a,self.b]

	def get_avg_estimates():
		# Returns avg value of each arm/category
		return self.a/(self.a+self.b)

	def recommend(self,
				num_of_preds=1):

		predictions = []

		for i in range(num_of_preds):
			sampled_q = np.random.beta(self.a[0,:],self.b[0,:])
			categ_choise = np.argmax( sampled_q )
			predictions.append( categ_choise )

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

	def __init__(self,A):

		self.A = A

	def __call__():

		U,sigma,V_T = np.linalg.svd(A)




				






















