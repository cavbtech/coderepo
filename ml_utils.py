from os import X_OK
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier	
from sklearn.metrics import fbeta_score, f1_score,precision_score,recall_score,accuracy_score, roc_curve, auc


#model
model = LogisticRegression(solver='liblinear', class_weight='balanced')

X, y, cat_ix, num_ix = load_dataset()
# one hot encode categorical, normalize numerical
ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
# scale, then undersample, then fit model
pipeline = Pipeline(steps=[('t',ct), ('s', RepeatedEditedNearestNeighbours()), ('m',model)])
pipeline.fit(X, y)

classes = {0: "Bad risk",1: "Good risk"}

def load_dataset():
	df=read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
	last_ix = len(df.columns) - 1
	X, y = df.drop(last_ix, axis=1), df[last_ix]
	
	# select categorical features
	cat_ix = x.select_dtypes(include=['object', 'bool']).columns
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X.values, y, cat_ix, num_ix
	
def predict(query_data):
	print(f"query_data.dict().values()={query_data.dict().values()}")
	x = list(query_data.dict().values())
	prediction = pipeline.predict([x])[0] 
	print(f"prediction={prediction}")
	print(f"Model prediction: {classes[prediction]}")
	return classes[prediction]


def check_accurary(X,y,X_test, y_test):
	best_accuracy = 0.0
	right_number_neighbours = 1
	for i in range(1,10):
		knnmodel = KNeighborsClassifier(n_neighbors=i)
		knnmodel.fit(X,y)
		#Predicting for test data with logistic regression
		y_pred = knnmodel.predict(X_test)
		###Calculating results for various evaluation metric
		precision = precision_score(y_test,y_pred, average='micro')
		recall 	  = recall_score(y_test,y_pred, average='micro')
		accuracy  = accuracy_score(y_test,y_pred)
		f1        = f1_score(y_test,y_pred, average='macro')
		if best_accuracy < accuracy :
			best_accuracy           = accuracy
			right_number_neighbours = i
			
	print(f"---------------------if neighbors = {i}---------------------------------------")
	print(f"Accuracy: {accuracy}")
	print(f"Recall: {recall}")
	print(f"Precision: {precision}")
	print(f"F1-score: {f1}")
	print(f"""best accuracy {best_accuracy} 
	is coming for {right_number_neighbours} neighbours 	
	and hence right number of neighbours = {right_number_neighbours}""")
