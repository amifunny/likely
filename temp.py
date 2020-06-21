# import pandas as pd

# df = pd.read_csv( 'static/dataset/Movie_data_1k.csv' )

# df_list = df.to_numpy()
# all_keys = set()
# for row in df_list:
# 	if row[3]==row[3]:
# 		all_keys.update( row[3].split("|") )

# with open('static/keyword.txt','w') as file:

# 	file.write( "|".join(all_keys) )
# 	file.close()	


# with open('static/keyword.txt','r') as file:
# 	content = file.read()
# 	file.close()	

# print( content.split('|') )	

import numpy as np
tp = np.random.rand( 4,4 )
U , sigma , V = np.linalg.svd( tp )
print( U )
print( sigma )
print( V )

