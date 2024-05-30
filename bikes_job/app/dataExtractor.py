# -*- coding: utf-8 -*-

import pandas

class DataExtractor:
	def __init__(self):
		print("DataExtractor: init\n")
	def run(self, path, settings):
		print(f"DataExtractor: run")
		df = pandas.read_csv(
			path,
			encoding = settings['encoding'],
			dtype = settings['dtypes']
		)
		print(f"DataExtractor: run done [shape {df.shape}]\n")
		return df

