# -*- coding: utf-8 -*-

import sys
import json
import joblib

from dataExtractor import DataExtractor
from dataValidator import DataValidator
from dataPreparator import DataPreparator
from model import Model

class Pipeline:
	def __init__(self, settingsPath):
		print(f"\nPipeline: init [settings path '{settingsPath}']")
		with open(settingsPath, 'r') as f:
			self.settings = json.load(f)
		print(json.dumps(self.settings, indent = 2))
		print("Pipeline: init done")
	def run(self, jobName, pathData, path):
		print(f"\nPipeline: run")
		assert jobName in ["train", "test"]
		df = DataExtractor().run(pathData, self.settings["dataExtractor"])
		df = DataValidator().run(jobName, df, self.settings["dataValidator"])
		df = DataPreparator().run(df, self.settings["dataPreparator"])
		if jobName == "train":
			Model().run(df, path, self.settings["modelTraining"])
		elif jobName == "test":
			model, features = joblib.load(path)
			df = Model().test(model, df[features].copy(), None)
			df.to_csv(f"{pathData[:-4]}_predicted.csv")
		print(f"Pipeline: run done")
if __name__ == '__main__':
	assert len(sys.argv) == 5, """

Please use like this:
	
python3 pipeline.py train <settingsPath>.json <dataPath>.csv <modelPath>.pkl

python3 pipeline.py test <settingsPath>.json <dataPath>.csv <modelPath>.pkl to get <dataPath>_predicted.csv file

"""
	Pipeline(sys.argv[2]).run(sys.argv[1], sys.argv[3], sys.argv[4])

