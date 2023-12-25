import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error
import os

TECdata = pd.read_csv('website\\model\\newdata(1).csv')
tecdata = pd.DataFrame(TECdata)

tecdata.columns = ['Year', 'Day', 'Month', 'HourofDay', 'Rz12', 'IG12', '3hapindex', '3hkpindex', 'TEC']

tecdata = tecdata.dropna()
tecdata.astype({'Year':'int'})
tecdata.astype({'Year':'int'})
tecdata.astype({'3hkpindex':'int'})
tecdata = tecdata.dropna()

features = ['Year','Day', 'Month', 'HourofDay','Rz12','IG12','3hapindex','3hkpindex']
X = tecdata[features]
y = tecdata['TEC']

X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.33, random_state=1) # 3 parts go to testing and 7 parts go to training for every 10 parts

# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y.values)
#     preds_val = model.predict(val_X.values)
#     mae = mean_absolute_error(val_y, preds_val)
#     return(mae)

# candidate_max_leaf_nodes = [3000, 3500, 4000, 4500, 5000]
# # Write loop to find the ideal tree size from candidate_max_leaf_nodes
# scores = {leaf_size: get_mae(leaf_size, y_train, y_test, X_train, X_test) for leaf_size in candidate_max_leaf_nodes}

# # Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
# best_tree_size = min(scores, key=scores.get)

# scores

# Fill in argument to make optimal size and uncomment
final_model = RandomForestRegressor(max_leaf_nodes=5000,random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X_train.values, y_train)

pred = final_model.predict(X_test.values)

print(pred[1:5])

import pickle
# print(final_model.predict(X_test))

with open("website\model\TEC_model3.pkl", "wb") as file:
    pickle.dump(final_model, file)

file_path = 'website\model\TEC_model3.pkl'
abs_file_path = os.path.abspath(file_path)
# print(abs_file_path)

# Opening saved model
with open(abs_file_path, "rb") as file:
    current_model = pickle.load(file)