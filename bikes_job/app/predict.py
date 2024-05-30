# -*- coding: utf-8 -*-

import sys
import json

from dataExtractor import DataExtractor
from dataValidator import DataValidator
from dataPreparator import DataPreparator
from modelTraining import ModelTraining

class Pipeline:
	def __init__(self, settingsPath):
		print(f"\nPipeline: init [settings path '{settingsPath}']")
		with open(settingsPath, 'r') as f:
			self.settings = json.load(f)
		print(json.dumps(self.settings, indent = 2))
		print("Pipeline: init done")
	def run(self, pathData, pathModel):
		print(f"\nPipeline: run")
		df = DataExtractor().run(pathData, self.settings["dataExtractor"])
		df = DataValidator().run(df, self.settings["dataValidator"])
		df = DataPreparator().run(df, self.settings["dataPreparator"])
		df = ModelTraining().run(df, pathModel, self.settings["modelTraining"])
		print(f"Pipeline: run done")

if __name__ == '__main__':
	assert len(sys.argv) == 4, """

Please use like this:
	
python3 predict.py <settingsPath>.json <dataPath>.csv <modelPath>.pkl
"""
	Pipeline(sys.argv[1]).run(sys.argv[2], sys.argv[3])

