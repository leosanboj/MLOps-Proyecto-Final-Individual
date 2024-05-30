# -*- coding: utf-8 -*-

import json
import joblib
import pandas

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class Model:
	def __init__(self):
		print("Model: init")
		self.models = {
			"RFR": {
				"module": "sklearn.ensemble",
				"class": "RandomForestRegressor",
				"hyperparameters": {
					"random_state": [0],
					"n_estimators": [10, 20, 30, 40, 50, 100],
					"verbose": [1]
				}
			},
			"XGB": {
				"module": "xgboost",
				"class": "XGBRegressor",
				"hyperparameters": {
					"random_state": [0],
					"max_depth": [3, 4, 5, 6, 7, 8],
					"tree_method": ["hist"],
					"verbosity": [1]
				}
			}
		}
	def run(self, df, path, settings):
		print(f"Model: run")
		df = self.featuresSelection(df, settings["featuresSelection"])
		df_train, df_test = self.splitData(df, settings["splitData"])
		bestPe = None
		bestModel = None
		for mName, s in self.models.items():
			print(f"\nModel: {mName}")
			print(json.dumps(s, indent = 2))
			module = __import__(s["module"], fromlist = [s["class"]])
			model = getattr(module, s["class"])
			model = self.train(df_train, model, s["hyperparameters"])
			pe = self.test(model, df_test, df)
			if not bestPe:
				bestPe = pe
				bestModel = model
		bestPe *= 100
		assert bestPe >= settings["maxPe"], f"Maximum Total-Stock-Known Percentage-Error not reached {pe:.3%} >= {settings['maxPe']}%"
		joblib.dump([
			bestModel,
			list(df.drop(columns = ["rented"]).columns)
		], path)
		print(f"Model: run done\n")
	def featuresSelection(self, df, settings):
		print(f"Model: featuresSelection")
		correlation_threshold = settings["correlation"] / 100
		df_c = df.corr()["rented"].sort_values(ascending = False)
		df_c = df_c[df_c.index != "rented"]
		df_c = df_c[abs(df_c) >= correlation_threshold]
		print(df_c)
		dFeatures = [c for c in df.columns if not c in df_c.index]
		df = df[df_c.index.to_list() + ["rented"]]
		print(f"Features dropped: {dFeatures}")
		print(f"Model: featuresSelection done\n")
		return df
	def splitData(self, df, settings):
		print(f"Model: splitData")
		rows_train = int(df.shape[0] * settings["train_window_percentage"])
		rows_test  = df.shape[0] - rows_train
		df_train = df.iloc[-rows_train:].copy()
		df_test  = df.iloc[ :rows_test ].copy()
		print(f'Training Window [{df_train["date"].max()}, {df_train["date"].min()}] {df_train["date"].max() - df_train["date"].min()} ({df_train.shape[0]} first rows)')
		print(f'Testing  Window [{df_test["date"].max()}, {df_test["date"].min()}] {df_test["date"].max() - df_test["date"].min()} ({df_test.shape[0]} last rows)')
		print(f"Model: splitData done\n")
		return df_train, df_test
	def train(self, df_train, model, hyperparameters):
		print(f"Model: train")
		df_train_x = df_train.drop(columns = ["date", "rented"]).copy()
		gsModel = GridSearchCV(model(), hyperparameters, n_jobs = -1, verbose = 1)
		gsModel.fit(df_train_x, df_train["rented"])
		print(pandas.DataFrame(gsModel.cv_results_))
		model = gsModel.best_estimator_
		pandas.DataFrame(
			{
				"feature": df_train_x.columns,
				"importance": model.feature_importances_
			}
		).sort_values(
			by = ["importance"], ascending = False, ignore_index = True
		).set_index(
			"feature"
		).plot(
			kind = "bar",
			figsize = (16, 4),
			title = "Features Importance"
		)
		plt.tight_layout()
		plt.savefig("features.png")
		print(f"Model: train done\n")
		return model
	def test(self, model, df_test, df_evaluate):
		print(f"Model: test")
		df_test["predicted"] = model.predict(df_test.drop(
			columns = ["date", "rented", "predicted"],
			errors = "ignore"
		))
		if df_evaluate is None:
			print(f"Model: test done\n")
			return df_test
		df_test[["date", "rented", "predicted"]].set_index("date")
		df_test.set_index("date")[
			["rented", "predicted"]
		].plot(
			kind = "line",
			figsize = (16, 4),
			title = "Rented"
		)
		plt.tight_layout()
		plt.savefig("test.png")
		mae = (df_test["predicted"] - df_test["rented"]).abs().mean()
		pe = mae / df_evaluate['rented'].max()
		print(f"Mean Absolute Error: {mae:.3f}")
		print(f"Total-Stock-Known Percentage-Error: {pe:.3%}")
		print(f"Model: test done\n")
		return pe

