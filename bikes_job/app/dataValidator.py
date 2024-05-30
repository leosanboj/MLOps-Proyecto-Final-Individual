# -*- coding: utf-8 -*-

import pandas
import json

class DataValidator:
	def __init__(self):
		print("DataValidator: init\n")
	def run(self, jobName, df, settings):
		print(f"DataValidator: run")
		self.dNullsChecker(df, settings['maxNulls%'])
		self.dTypesChecker(jobName, df, settings['dtypes'])
		df["Date"] = pandas.to_datetime(df["Date"], format = "%d/%m/%Y")
		df = self.dRangesChecker(df, settings['dranges'])
		print(f"DataValidator: run done shape {df.shape}\n")
		return df
	def dTypesChecker(self, jobName, df, settings):
		print(f"DataValidator: dTypesChecker")
		errors = ""
		df_dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
		if jobName == "test":
			del settings["Rented Bike Count"]
		for c, t in settings.items():
			if c in df_dtypes:
				if df_dtypes[c] != t:
					errors +=  f"\nColumn '{c}' data type is '{df_dtypes[c]}'"
					errors +=  f" but '{t}' is expected."
			else:
				errors +=  f"\nColumn '{c}' not found on data"
		assert errors == "", errors
		print(f"DataValidator: dTypesChecker done\n")
	def dNullsChecker(self, df, settings):
		print(f"DataValidator: dNullsChecker")
		df_nulls = ((100 * df.isna().sum()) / df.shape[0]).to_dict()
		df_nulls = pandas.DataFrame({
			"column": df_nulls.keys(),
			"nulls found %": df_nulls.values(),
			"max allowed %": [float(settings[c]) for c in df_nulls.keys()]
		})
		print(df_nulls)
		df_nulls_error = df_nulls[
			df_nulls["nulls found %"] > df_nulls["max allowed %"]
		]
		assert df_nulls_error.shape[0] == 0, f"\nMax nulls % reached:\n{df_nulls_error}"
		print(f"DataValidator: dNullsChecker done\n")
	def dRangesChecker(self, df, settings):
		print(f"DataValidator: dRangesChecker")
		errors = ""
		n = 0
		for c, r in settings.items():
			if type(r) == int:
				count = df[c].nunique()
				if count > r:
					errors +=  f"\nColumn '{c}' max distinct values {r}"
					errors +=  f" but {count} found"
			else:
				print(f"Filtering column {r[0]} <= {c} <= {r[1]}", end = "")
				if r[0]:
					df_r = df[df[c] >= r[0]]
				if r[1]:
					df_r = df_r[df_r[c] <= r[1]]
				n += df_r.shape[0] - df.shape[0]
				print(f" ({df_r.shape[0] - df.shape[0]} rows removed)")
				df = df_r
		assert errors == "", errors
		print(f"DataValidator: dRangesChecker done [{n} total rows removed]\n")
		return df.copy()

