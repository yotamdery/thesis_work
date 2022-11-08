## Imports
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
from scipy import optimize
import sys
sys.path.append('../SEIR_full/')
sys.path.append('..')
import SEIR_full as mdl
import SEIR_full.model as mdl
import SEIR_full.calibration as mdl
import datetime as dt
from scipy.stats import poisson
from scipy.stats import binom
import copy

#################################################
# Generating Calibration Parameters with deaths  #
#################################################

with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

for scen in [mdl.num2scen(i) for i in [4]]:
	for phase in [60]:
		#         mdl.show_calibration(
		#             ind,
		#             int(scen[-1]),
		#             phase,
		#         )
		# #         run calibration for model parameters
		mdl.make_calibration(
			scen,
			phase,
			ind,
		)

		# #         run calibration for model tracking states
		#         mdl.make_track_calibration(
		#             scen,
		#             phase,
		#             ind,
		#         )