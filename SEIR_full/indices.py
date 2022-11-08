#!/usr/bin/env python3
import itertools
import numpy as np
import pandas as pd
from .utils import *

class Indices:

	def __init__(self, cell_name= '30'):
		self.cell_name = cell_name
		# Age groups
		self.A = {
			0: '0-4',
			1: '5-9',
			2: '10-19',
			3: '20-29',
			4: '30-39',
			5: '40-49',
			6: '50-59',
			7: '60-69',
			8: '70+',
		}

		# Risk groups
		self.R = {
			0: 'High',
			1: 'Low',
		}

		self.cell = {}  	# holds '250' value
		self.G = {}  		# Regions dict
		self.N = {}  		# A dictionary of all of the possible combinations of intervention, region, risk and age (2x241X2x9 = 8676)
		self.GA = {}  		# All Regions & Age combinations dict
		self.MI = {}  		# All Age & Region & Age combinations dict (keys length of 9x241x9 = 19,521)
		self.GRA = {}  		# All Regions & Risk & Age combinations dict
		self.region_gra_dict = {}
		self.age_gra_dict = {}  		# For each Age group, specifies it's locations (indexes of columns)
		self.risk_dict = {}
		self.region_age_dict = {}
		self.region_risk_age_dict = {}
		self.age_dict = {}  		# For each Age group, specifies it's locations (indexes of columns)
		self.risk_age_dict = {}
		self.age_ga_dict = {}
		self.region_dict = {}
		self.region_ga_dict = {}
		self.region_risk_dict = {}
		self.make_new_indices()


	def make_new_indices(self, empty_list=None):
		#define cells
		self.cell = pd.read_excel(
			'../Data/division_choice/' + self.cell_name + '/county_int_2name_county_string.xlsx', engine='openpyxl')
		self.cell['county_id'] = self.cell['county_id'].astype(str)

		# remove empty cells from indices
		if not(empty_list is None):
			self.cell = self.cell[
				self.cell['county_id'].apply(lambda x: x not in empty_list.values)]
		# set area indices
		self.G = {i: str(k) for i, k in enumerate(list(self.cell['county_id'].values))}
		# set cell names dict
		self.cell.set_index('county_id', inplace=True)
		self.cell.index = self.cell.index.astype(str)
		self.cell = self.cell.to_dict()['cell_name']

		# All combination:
		# itertools.product() gives the cartesian product of the lists (gives all the combinations of the lists content)
		# i is an auto-increment number (received by the enumerate() function) and group is the intersection of all 4 lists
		self.N = {i: group for i, group in enumerate(itertools.product(
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		}

		# Region and age combination - for beta_j
		self.GA = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.G.values(),
				self.A.values(),
			))
		}

		self.MI = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.A.values(),
				self.G.values(),
				self.A.values(),
			))
		}

		self.GRA = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		}

		self.region_gra_dict = get_opposite_dict(
			self.GRA,
			[[x] for x in list(self.G.values())],
		)

		self.age_gra_dict = get_opposite_dict(
			self.GRA,
			[[x] for x in list(self.A.values())],
		)

		self.risk_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.R.values())],
		)

		self.region_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.G.values(),
				self.A.values(),
			)),
		)

		self.region_risk_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		)

		self.age_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.A.values())],
		)

		self.risk_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.R.values(),
				self.A.values(),
			)),
		)

		self.region_risk_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.G.values(),
				self.R.values(),
			)),
		)

		self.age_ga_dict = get_opposite_dict(
			self.GA,
			[[x] for x in list(self.A.values())],
		)

		self.region_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.G.values())],
		)

		self.region_ga_dict = get_opposite_dict(
			self.GA,
			[[x] for x in list(self.G.values())],
		)

	def update_empty(self):
		empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
			'cell_id'].astype(str)
		self.make_new_indices(empty_cells)
