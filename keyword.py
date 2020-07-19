import pandas as pd
import ast

# Script to write Auxillary file 'keyword.txt' for Bandits

df = pd.read_csv('Movie_dataset.csv')
df['genre'] = df['genre'].apply(lambda x: ast.literal_eval(x))
string_content = "|".join( sorted( df['genre'].explode().unique() ) )

with open('static/keyword.txt') as file:
	file.write( string_content )
	file.close()