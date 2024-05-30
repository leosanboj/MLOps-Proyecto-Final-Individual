# -*- coding: utf-8 -*-

import pandas

pandas.set_option('display.max_columns', None)

class DataPreparator:
	def __init__(self):
		print("DataPreparator: init")
	def run(self, df, settings):
		print(f"DataPreparator: run [{df.shape} in shape]")
		df.rename(columns = settings["renameColumns"], inplace = True)
		df = self.removeNulls(df)
		df = self.datetimePreparation(df)
		df = self.categoricalPreparation(df)
		df.sort_values(
			["date"], ascending = False, ignore_index = True, inplace = True
		)
		print(df.head(3))
		df.info(verbose = True)
		print(df.describe())
		print(f"DataPreparator: run done [{df.shape} out shape]\n")
		return df
	def removeNulls(self, df):
		print("DataPreparator: removeNulls")
		n = df.shape[0]
		df = df.dropna()
		print(f"DataPreparator: removeNulls done ({df.shape[0] - n} rows removed)\n")
		return df.copy()
	def datetimePreparation(self, df):
		print("DatetimePreparation: datetimePreparation")
		df["date"] = df["date"] + pandas.to_timedelta(df["hour"], unit = "h")
		df["dW"] = df["date"].dt.weekday
		df["dD"] = df["date"].dt.day
		df["dM"] = df["date"].dt.month
		print("DatetimePreparation: datetimePreparation done\n")
		return df
	def categoricalPreparation(self, df):
		print("DatetimePreparation: categoricalPreparation")
		df["seasons"] = df["seasons"].str.lower()
		df["holiday"] = (df["holiday"] == "No Holiday").astype(int)
		df["open"] = (df["open"] == "Yes").astype(int)
		df = pandas.get_dummies(
			df,
			columns = ["seasons"],
			prefix = ["s"],
			dtype = int
		)
		print("DatetimePreparation: categoricalPreparation done\n")
		return df

