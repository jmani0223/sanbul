import os, urllib, tarfile
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/jmani0223/sanbul/master/"
FIRES_PATH = os.path.join("datasets", "sanbul")
FIRES_URL = DOWNLOAD_ROOT + "datasets/sanbul/sanbul-5.csv"
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "sanbul_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
 path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
 print("Saving figure", fig_id)
 if tight_layout:
    plt.tight_layout()
 plt.savefig(path, format=fig_extension, dpi=resolution)

def fetch_fires_data(fires_url=FIRES_URL, fires_path=FIRES_PATH):
    if not os.path.isdir(fires_path):
        os.makedirs(fires_path)
    csv_path = os.path.join(fires_path, "sanbul-5.csv")
    urllib.request.urlretrieve(fires_url, csv_path)
    fires_tgz = tarfile.open(csv_path)
    fires_tgz.extractall(path=fires_path)
    fires_tgz.close()

fetch_fires_data()

def load_fires_data(fires_path=FIRES_PATH):
    csv_path = os.path.join(fires_path, "sanbul-5.csv")
    return pd.read_csv(csv_path)

fires = load_fires_data()

print("### fires.head():\n")
print(fires.head())

print("\n### fires.info():\n")
print(fires.info())

print("\n### fires.describe()\n")
print(fires.describe())

print("\n### fires['month'].value_counts():\n")
print(fires["month"].value_counts())

print("\n### fires['day'].value_counts():\n")
print(fires["day"].value_counts())

print("\n\nHistogram plots:\n")
fires.hist(bins=50, figsize=(20,15))
save_fig("01_attribute_histogram_plots")
plt.show()

# y = ln(burned_area+1)  왜곡된 burned_area 변환
fires['burned_area'] = np.log(fires['burned_area'] + 1)


print("\n\nHistogram plots:\n")
fires.hist(bins=50, figsize=(20,15))
save_fig("02_burned_area_attribute_histogram_plots")
plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()
fires["month"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]
print("\nMonth category proportion: \n",
    strat_test_set["month"].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n",
    fires["month"].value_counts()/len(fires))

from pandas.plotting import scatter_matrix
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attributes], figsize=(12, 8))
save_fig("03_scatter_matrix_plot")
plt.show()

fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=fires["max_temp"], label="max_temp",
c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
save_fig("04_fires_prices_scatterplot")
plt.show()

corr_matrix = fires.corr()
print("\nCORR_MATRIX:\n")
print(corr_matrix["burned_area"].sort_values(ascending=False))

#1-9
fires = strat_train_set.drop(["burned_area"], axis=1) # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
fires_cat = fires[["month"]]
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
print("\nfires_cat_1hot: \n", fires_cat_1hot)
print("\n\nAlternatively, you can set sparse=False when creating the OneHotEncoder:")
cat_encoder = OneHotEncoder(sparse=False)
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)
print("\ncat_month_encoder.categories_: \n", cat_encoder.categories_)

cat_encoder2 = OneHotEncoder()
fires_cat = fires[["day"]]
fires_cat_1hot_2 = cat_encoder2.fit_transform(fires_cat)
print("\nfires_cat_1hot_2: \n", fires_cat_1hot_2)
print("\n\nAlternatively, you can set sparse=False when creating the OneHotEncoder:")
cat_encoder2 = OneHotEncoder(sparse=False)
fires_cat_1hot_2 = cat_encoder2.fit_transform(fires_cat)
print("\ncat_day_encoder.categories_: \n", cat_encoder2.categories_)

#1-10
print("\n\n########################################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
 ])
fires_num_tr = num_pipeline.fit_transform(fires_num)
print("\nfires_num_tr: \n", fires_num_tr)

from sklearn.compose import ColumnTransformer
num_attribs = list(fires_num)
cat_attribs = ["month", "day"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
 ])
fires_prepared = full_pipeline.fit_transform(fires)

print("\nfires prepared:", fires_prepared)
print("\nfires prepared shape:", fires_prepared.shape)

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fires['longitude'] = scaler.fit_transform(fires['longitude'].values.reshape(-1,1))
fires['latitude'] = scaler.fit_transform(fires['latitude'].values.reshape(-1,1))

fires['avg_temp'] = scaler.fit_transform(fires['avg_temp'].values.reshape(-1,1))
fires['max_temp'] = scaler.fit_transform(fires['max_temp'].values.reshape(-1,1))
fires['max_wind_speed'] = scaler.fit_transform(fires['max_wind_speed'].values.reshape(-1,1))
fires['avg_wind'] = scaler.fit_transform(fires['avg_wind'].values.reshape(-1,1))

fires = pd.get_dummies(fires, columns=['longitude', 'latitude', 'month', 'day'])
print("\nScaled train dataset:\n", fires)

from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train, y_test = train_test_split(fires, fires_labels, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train, test_size=0.1, random_state=42)

print("\nTrain dataset:\n", X_train)
print("\nTrain target:\n", y_train)
 '''
#sgd
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(fires_prepared, fires_labels)

#svm
from sklearn.svm import SVR
svm_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_reg.fit(fires_prepared, fires_labels)

#dt
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(fires_prepared, fires_labels)

#rf
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(fires_prepared, fires_labels)

#2-1
from sklearn.model_selection import GridSearchCV
print("sgd_reg.get_params().keys(): ", sgd_reg.get_params().keys())
print("svm_reg.get_params().keys(): ", svm_reg.get_params().keys())
print("tree_reg.get_params().keys(): ", tree_reg.get_params().keys())
print("forest_reg.get_params().keys(): ", forest_reg.get_params().keys())

params_sgd = [{'alpha': [0.1, 0.5], 'epsilon':[0.1, 1]},
{'alpha': [0.5, 0.6], 'epsilon':[0.1, 0.7]},]

params_svm = {'kernel': ["linear", "poly", "rbf"],
'C': [0.1, 1, 10, 100],
'degree': [2, 3, 4],
'epsilon':[0.1, 1.0, 1.5]}

params_tree = [{'max_features':[2, 4, 6, 8]},
{'max_features':[2, 3, 4]},]

params_forest = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features':[2, 3, 4]},]

grid_search_cv = GridSearchCV(sgd_reg, params_sgd, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
sgd_best_model_cv = grid_search_cv.best_estimator_
print(sgd_best_model_cv)

grid_search_cv = GridSearchCV(svm_reg, params_svm, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
svm_best_model_cv = grid_search_cv.best_estimator_
print(svm_best_model_cv)

grid_search_cv = GridSearchCV(tree_reg, params_tree, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
tree_best_model_cv = grid_search_cv.best_estimator_
print(tree_best_model_cv)

grid_search_cv = GridSearchCV(forest_reg, params_forest, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
forest_best_model_cv = grid_search_cv.best_estimator_
print(forest_best_model_cv)

#2-2
#sgd
from sklearn.metrics import mean_squared_error
fires_predictions = sgd_reg.predict(fires_prepared)
sgd_mse = mean_squared_error(fires_labels, fires_predictions)
sgd_rmse = np.sqrt(sgd_mse)
# revert into the original value: y=ln(burned_area+1) => burned_area = exp(y)-1
sgd_rmse_reverted = np.exp(sgd_rmse)-1
print("\nSGD - RMSE(train set):\n", sgd_rmse_reverted)
#svm
fires_predictions = svm_reg.predict(fires_prepared)
svm_mse = mean_squared_error(fires_labels, fires_predictions)
svm_rmse = np.sqrt(svm_mse)
# revert into the original value: y=ln(burned_area+1) => burned_area = exp(y)-1
svm_rmse_reverted = np.exp(svm_rmse)-1
print("\nSVM - RMSE(train set):\n", svm_rmse_reverted)
#dt
fires_predictions = sgd_reg.predict(fires_prepared)
tree_mse = mean_squared_error(fires_labels, fires_predictions)
tree_rmse = np.sqrt(tree_mse)
# revert into the original value: y=ln(burned_area+1) => burned_area = exp(y)-1
tree_rmse_reverted = np.exp(tree_rmse)-1
print("\nDT - RMSE(train set):\n", tree_rmse_reverted)
#rf
fires_predictions = forest_reg.predict(fires_prepared)
forest_mse = mean_squared_error(fires_labels, fires_predictions)
forest_rmse = np.sqrt(forest_mse)
# revert into the original value: y=ln(burned_area+1) => burned_area = exp(y)-1
forest_rmse_reverted = np.exp(forest_rmse)-1
print("\nRF - RMSE(traRFin set):\n", forest_rmse_reverted)

#2-3
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
 train_errors, val_errors = [], []
 for m in range(1, len(X_train)):
     model.fit(X_train[:m], y_train[:m])
     y_train_predict = model.predict(X_train[:m])
     y_val_predict = model.predict(X_val)
     train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
     val_errors.append(mean_squared_error(y_val_predict, y_val))
 plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
 plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
 plt.legend(loc="upper right", fontsize=14) # not shown in the book
 plt.xlabel("Training set size", fontsize=14) # not shown
 plt.ylabel("RMSE", fontsize=14) # not shown

plot_learning_curves(sgd_reg, fires_prepared, fires_labels)
plt.axis([0, 80, 0, 3]) # not shown in the book
print("\n")
save_fig("sgd_learning_curves_plot") # not shown
plt.show()

plot_learning_curves(svm_reg, fires_prepared, fires_labels)
plt.axis([0, 80, 0, 3]) # not shown in the book
print("\n")
save_fig("svm_learning_curves_plot") # not shown
plt.show()

plot_learning_curves(tree_reg, fires_prepared, fires_labels)
plt.axis([0, 80, 0, 3]) # not shown in the book
print("\n")
save_fig("dt_learning_curves_plot") # not shown
plt.show()

plot_learning_curves(forest_reg, fires_prepared, fires_labels)
plt.axis([0, 80, 0, 3]) # not shown in the book
print("\n")
save_fig("forest_learning_curves_plot") # not shown
plt.show()

#2-4
from sklearn.model_selection import cross_val_score
sgd_scores = cross_val_score(sgd_reg, fires_prepared, fires_labels, scoring="neg_mean_squared_error", cv=10)
svm_scores = cross_val_score(svm_reg, fires_prepared, fires_labels, scoring="neg_mean_squared_error", cv=10)
tree_scores = cross_val_score(tree_reg, fires_prepared, fires_labels, scoring="neg_mean_squared_error", cv=10)
forest_scores = cross_val_score(forest_reg, fires_prepared, fires_labels, scoring="neg_mean_squared_error", cv=10)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

sgd_rmse_scores = np.sqrt(-sgd_scores)
print("\nSGD Regression scores (train set): \n")
display_scores(sgd_rmse_scores)

svm_rmse_scores = np.sqrt(-svm_scores)
print("\nSVM Regression scores (train set): \n")
display_scores(svm_rmse_scores)

tree_rmse_scores = np.sqrt(-tree_scores)
print("\nDT Regression scores (train set): \n")
display_scores(tree_rmse_scores)

forest_rmse_scores = np.sqrt(-forest_scores)
print("\nRF Regression scores (train set): \n")
display_scores(forest_rmse_scores)

#2-5
X_test = strat_test_set.drop("burned_area", axis=1)
y_test = strat_test_set["burned_area"].copy()
X_test_prepared = full_pipeline.transform(X_test)

sgd_scores = cross_val_score(sgd_reg, X_test_prepared, y_test, scoring="neg_mean_squared_error", cv=10)
sgd_rmse_scores = np.sqrt(-sgd_scores)
print("\nSGD Regression scores (test set): \n")
display_scores(sgd_rmse_scores)

svm_scores = cross_val_score(svm_reg, X_test_prepared, y_test, scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-svm_scores)
print("\nSVM Regression scores (test set): \n")
display_scores(svm_rmse_scores)

tree_scores = cross_val_score(sgd_reg, X_test_prepared, y_test, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
print("\nDT Regression scores (test set): \n")
display_scores(tree_rmse_scores)

forest_scores = cross_val_score(sgd_reg, X_test_prepared, y_test, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("\nRF Regression scores (test set): \n")
display_scores(forest_rmse_scores)

#keras model
import tensorflow as tf
from tensorflow import keras

X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2,
random_state=42)
X_test = X_test_prepared
y_test = y_test
np.random.seed(42)
tf.random.set_seed(42)
mlp_model = keras.models.Sequential([
keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
keras.layers.Dense(1)
])
mlp_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = mlp_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
mse_test = mlp_model.evaluate(X_test, y_test)
X_new = X_test[:10]
y_pred = mlp_model.predict(X_new)

# Learning curves: train set (blue line), validation set (orange line)
history = mlp_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
mse_test = mlp_model.evaluate(X_test, y_test)
X_new = X_test[:10]
y_pred = mlp_model.predict(X_new)
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
save_fig("learning curves") # not shown
print("learning curves")
plt.show()

model_version = "0001"
model_name = "my_sanbul_model"
model_path = os.path.join(model_name, model_version)
print("\nmodel_path: \n", model_path)

tf.saved_model.save(mlp_model, model_path)

for root, dirs, files in os.walk(model_name):
 indent = ' ' * root.count(os.sep)
 print('{}{}/'.format(indent, os.path.basename(root)))
 for filename in files:
    print('{}{}'.format(indent + ' ', filename))
