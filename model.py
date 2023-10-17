import pandas as pd
import numpy as np


de_train=pd.read_parquet("de_train.parquet")
id_map=pd.read_csv("id_map.csv")
sample_submission = pd.read_csv("sample_submission.csv")




# shuffle the data
de_train = de_train.sample(frac=1.0, random_state=42)






# Create features and and labels for reverse model 18211 features and 152 labels for true model
features_columns = ["cell_type", "sm_name"]
#Característiques, son les dues variables categòriques



labels_columns=["cell_type","sm_name","sm_lincs_id","SMILES","control"]
#Etiquetes



#Generem la columna que volem com a resposta "labels",
labels = de_train.drop(columns=labels_columns)

#i la que ens servirà per fer la predicció, "features"
features = pd.DataFrame(de_train, columns=features_columns)

features




# Get test data 
test_data = pd.DataFrame(id_map, columns=features_columns)

#Això és el què volem predir. 


from sklearn.preprocessing import OneHotEncoder

# Create an instance of the encoder
encoder = OneHotEncoder()

# Fit the encoder on features
encoder.fit(features)

# Transform the features into one-hot encoded format
one_hot_encode_features = encoder.transform(features)

# Transform the test data(id_map)
one_hot_test = encoder.transform(test_data)


one_hot_encode_features.toarray().shape, one_hot_test.toarray().shape
from sklearn.model_selection import train_test_split

# Split the data into 70% training, 15% validation, and 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(one_hot_encode_features, labels.values, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

#El de validació, ens serveix per ajustar hiperparàmetres i evaluar el rendiment durant les prediccions



# Printing the shapes of the data splits
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)





# We also get full features for final training
full_features = one_hot_encode_features.toarray()
full_labels = labels.values

print("full_features shape:", full_features.shape)
print("full_labels shape:", full_labels.shape)


####LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Crear el modelo de Regresión Lineal
regression_model = LinearRegression()

# Entrenar el modelo en los datos de entrenamiento
regression_model.fit(X_train, y_train)

# Realizar predicciones en los datos de validación
y_val_pred = regression_model.predict(X_val)

# Calcular el error cuadrático medio (MSE) en los datos de validación
mse = mean_squared_error(y_val, y_val_pred)
print(f'Error cuadrático medio en datos de validación: {mse:.2f}')



####RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Crear el modelo de Regresión de Bosques Aleatorios
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)

# Definir los hiperparámetros que deseas probar
param_grid = {
    'n_estimators': [100],  # Puedes ajustar el número de árboles
    'max_depth': [10],  # Puedes ajustar la profundidad máxima del árbol
    'min_samples_split': [2],  # Puedes ajustar el número mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1]  # Puedes ajustar el número mínimo de muestras en una hoja
}

# Crear un objeto GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Realizar la búsqueda de hiperparámetros en los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Obtener el modelo con los mejores hiperparámetros
best_rf_model = grid_search.best_estimator_

# Realizar predicciones en los datos de validación
y_val_pred_rf = best_rf_model.predict(X_val)

# Calcular el error cuadrático medio (MSE) en los datos de validación
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
print(f'Error cuadrático medio en datos de validación (Random Forest): {mse_rf:.2f}')







######NEURAL NETWORK

from sklearn.neural_network import MLPRegressor

# Crear el modelo de Red Neuronal de Dos Capas
nn_model = MLPRegressor(hidden_layer_sizes=(50, 10, 100, 50, 1000, 20, 2033, 10,18211), activation='tanh', solver='adam', random_state=42)

# Entrenar el modelo en los datos de entrenamiento
nn_model.fit(X_train, y_train)

# Realizar predicciones en los datos de validación
y_val_pred_nn = nn_model.predict(X_val)

# Calcular el error cuadrático medio (MSE) en los datos de validación
mse_nn = mean_squared_error(y_val, y_val_pred_nn)
print(f'Error cuadrático medio en datos de validación (Red Neuronal de Dos Capas): {mse_nn:.2f}')



















#Los mejores hiperparámetros son: {'activation': 'tanh', 'hidden_layer_sizes': (50, 10), 'solver': 'adam'}
#Error cuadrático medio en datos de validación con los mejores hiperparámetros: 2.59








####RIDGE REGRESSION





from sklearn.linear_model import Ridge

# Crear el modelo de Regresión Ridge
ridge_model = Ridge(alpha=1.0)  # Puedes ajustar el valor de alpha según tus necesidades

# Entrenar el modelo en los datos de entrenamiento
ridge_model.fit(X_train, y_train)

# Realizar predicciones en los datos de validación
y_val_pred_ridge = ridge_model.predict(X_val)

# Calcular el error cuadrático medio (MSE) en los datos de validación
mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)
print(f'Error cuadrático medio en datos de validación (Ridge): {mse_ridge:.2f}')









#####TRAIN THE FULL MODEL 


nn_model.fit(full_features, full_labels)


##PREDICTION


preds = nn_model.predict(one_hot_test.toarray())

preds.shape



sample_columns = sample_submission.columns
sample_columns= sample_columns[1:]
submission_df = pd.DataFrame(preds, columns=sample_columns)

submission_df.insert(0, 'id', range(255))

sample_submission



submission_df.to_csv("submission_df.csv", index=False)

#!zip submission_preds.zip /kaggle/working/submission_df.csv





