





##### Hem de provar a veure si funciona el fet crear la matriu a pytho

import pandas as pd
import numpy as np

train = pd.read_parquet("multiome_train.parquet")
multi=pd.read_csv("multiome_var_meta.csv")
meta= pd.read_csv("multiome_obs_meta.csv")

nueva_train = train.iloc[232717:, train.columns != "normalized_count"]

#Eliminate rows where "location" doesn't start with "chr"
nueva_train = nueva_train[nueva_train['location'].str.startswith('chr')]

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

#Subset

#nueva_train_subset = nueva_train.head(n=100000)

#We create the matrix with pivot
mat = nueva_train.pivot(index='obs_id', columns='location', values='count')
mat = mat.fillna(0)
print(mat.head())

#Convert the dataframe to a numpy array
mat_array = mat.to_numpy()


#Get the column and row names
column_names = mat.columns
row_names = mat.index

mat_array_with_row_names = np.column_stack((row_names, mat_array))

# Save numpy to .csv with column and row names.
np.savetxt('mat2prova.csv', mat_array_with_row_names, delimiter=',', header=','.join(column_names), comments='', fmt='%s')
# Nota: header=','.join(column_names) creates a row with column names
# comments='' not any comments


import gzip

        
        # Save to csv
#np.savetxt('mat2prova.csv', mat_array, delimiter=',')

input_file = 'mat2prova.csv'
output_file = 'mat2prova.csv.gz'

# Zip the file
with open(input_file, 'rb') as f_in:
    with gzip.open(output_file, 'wb') as f_out:
        f_out.writelines(f_in)
