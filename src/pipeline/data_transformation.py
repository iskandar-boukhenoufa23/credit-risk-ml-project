import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score
)
from sklearn.model_selection import train_test_split

import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder




mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit-risk")



def load_preprocess():
	df = pd.read_csv("data/credit_risk.csv")

	print(df.head())

	# encoder person_home_ownership
	print(df["person_home_ownership"].value_counts())
	df["person_home_ownership"] = df["person_home_ownership"].str.strip().str.lower()
	df["person_home_ownership"] = df["person_home_ownership"].str.strip().str.lower()

	df = pd.get_dummies(
    		df,
    		columns=["person_home_ownership"],
    		drop_first=True   # important
	)
	print(df["person_home_ownership"].value_counts())
	print(df["person_emp_length"].describe())

	df["person_emp_length"] = df["person_emp_length"].clip(upper=50)
	df["person_emp_length"] = pd.cut(
		df["person_emp_length"],
		bins=[0, 2, 5, 10, 20, 50],
		labels=["0-2", "2-5", "5-10", "10-20", "20+"]
	)

	print(df["person_emp_length"])

	le = LabelEncoder()
	df["person_emp_length"] = le.fit_transform(df["person_emp_length"])



	# Ordinal Encoding for loan_grade
	grade_map = {
    	'A': 1,
    	'B': 2,
    	'C': 3,
    	'D': 4,
    	'E': 5,
    	'F': 6,
    	'G': 7
	}


	df["loan_grade"] = df["loan_grade"].map(grade_map).fillna(0)

	return df



	X = df.drop("loan_status", axis=1)
	y = df["loan_status"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	return X_train, y_train, X_test, y_test, le


def train_and_log_model(
	n_estimators: int = 100,
	max_depth=10,
	min_samples_split: int = 2,
	min_samples_leaf: int = 1
