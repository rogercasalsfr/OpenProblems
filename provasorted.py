





##### Hem de provar a veure si funciona el fet crear la matriu a pytho

import pandas as pd
import numpy as np

train = pd.read_parquet("multiome_train.parquet")
multi=pd.read_csv("multiome_var_meta.csv")
meta= pd.read_csv("multiome_obs_meta.csv")




# Seleccionar todas las filas desde la fila 232718 hacia abajo en el DataFrame "train" sin la columna "normalized_count"
nueva_train = train.iloc[232717:, train.columns != "normalized_count"]

# Eliminar las filas en las que "location" no comienza con "chr"
nueva_train = nueva_train[nueva_train['location'].str.startswith('chr')]




import pandas as pd
from scipy.sparse import csr_matrix

# Supongamos que tienes un DataFrame llamado subset_train
# y quieres realizar la operación descrita en tu código R.

# Primero, crea una tabla pivote en Python
#mat =nueva_train.pivot(index='location', columns='obs_id', values='count')

# Luego, convierte la tabla pivote en una matriz dispersa (sparse matrix) con la biblioteca SciPy
#mat_sparse = csr_matrix(mat.values)

# Si necesitas una matriz en formato Matrix, puedes utilizar la biblioteca Matrix
# Asegúrate de tener la biblioteca Matrix instalada con: pip install Matrix
#from Matrix import Matrix
#mat = Matrix(mat_sparse)

# Si deseas establecer las ubicaciones como nombres de fila, puedes hacerlo de la siguiente manera:
#mat.set_rownames(mat_sparse.index)

# Ahora, 'mat' es una matriz dispersa en formato Matrix con las ubicaciones como nombres de fila.



#import dask.dataframe as dd

#nueva_train = dd.from_pandas(nueva_train, npartitions=10)  # Divide el DataFrame en particiones para el procesamiento en paralelo
#mat = nueva_train.pivot(index='location', columns='obs_id', values='count', fill_value=0)
#mat = mat.compute()  # Obtiene el resultado en un DataFrame de Pandas convencional

import pandas as pd
from scipy.sparse import save_npz

# Supongamos que tienes un DataFrame llamado nueva_train
# y deseas realizar una prueba con las primeras 1000 filas.

#nueva_train_subset = nueva_train.head(n=100000)

# Realiza la operación de pivotado en el subconjunto
mat = nueva_train.pivot(index='obs_id', columns='location', values='count')
mat = mat.fillna(0)
print(mat.head())

# Convierte el DataFrame a una matriz NumPy
mat_array = mat.to_numpy()


# Obtiene los nombres de columnas y filas
column_names = mat.columns
row_names = mat.index

mat_array_with_row_names = np.column_stack((row_names, mat_array))

# Guarda la matriz NumPy en un archivo CSV junto con los nombres de filas y columnas
np.savetxt('mat2prova.csv', mat_array_with_row_names, delimiter=',', header=','.join(column_names), comments='', fmt='%s')
# Nota: header=','.join(column_names) crea una línea de encabezado con los nombres de las columnas
# comments='' asegura que no haya caracteres de comentario en el archivo CSV
import gzip

        
        # Guarda la matriz NumPy en un archivo CSV
#np.savetxt('mat2prova.csv', mat_array, delimiter=',')

# Nombre del archivo de entrada (archivo CSV)
input_file = 'mat2prova.csv'

# Nombre del archivo de salida comprimido (archivo GZ)
output_file = 'mat2prova.csv.gz'

# Comprime el archivo CSV usando gzip
with open(input_file, 'rb') as f_in:
    with gzip.open(output_file, 'wb') as f_out:
        f_out.writelines(f_in)
