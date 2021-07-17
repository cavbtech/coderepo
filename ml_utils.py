from os import X_OK
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier	
from sklearn.metrics import fbeta_score, f1_score,precision_score,recall_score,accuracy_score, roc_curve, auc


clf_lr = LogisticRegression(penalty='l2',C=1.0, max_iter=10000)


classes = {0: "Bad risk",1: "Good risk"}

def load_model():
	df=read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
	last_ix = len(df.columns) - 1
	x, y = df.drop(last_ix, axis=1), df[last_ix]
	# Categorical features has to be converted into integer values for the model to process. 
	#This is done through one hot encoding.
	# select categorical features
	cat_ix = x.select_dtypes(include=['object', 'bool']).columns
	# one hot encode categorical features only
	ct = ColumnTransformer([('o',OneHotEncoder(),cat_ix)], remainder='passthrough')
	X = ct.fit_transform(x)
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

	
	clf_lr.fit(X_train, y_train)
	accuracy = check_accurary(X,y,X_test,y_test)	

	print(f"accuracy={accuracy}")

def predict(query_data):
	print(f"query_data.dict().values()={query_data.dict().values()}")
	x = list(query_data.dict().values())
	prediction = clf_lr.predict([x])[0] 
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