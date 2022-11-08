#!/usr/bin/env python3
import pickle

import numpy
import pandas as pd
import scipy.sparse
from SEIR_full.utils import *

########################
# -- Set Parameters -- #
########################
MAIN_DIR = '/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main'

# hospitalizations probabilities for age-, risk-groups :
with open(MAIN_DIR + '/Data/parameters/hospitalizations.pickle', 'rb') as pickle_in:
	hospitalizations = pickle.load(pickle_in)

# ventilation probabilities for age-, risk-groups :
with open(MAIN_DIR + '/Data/parameters/vents_proba.pickle', 'rb') as pickle_in:
	vents_proba = pickle.load(pickle_in)

# Asymptomatic fraction
with open(MAIN_DIR + '/Data/parameters/f0_full.pickle', 'rb') as pickle_in:
	f0_full = pickle.load(pickle_in)

# Contact matrices of routine
full_home_matrix = scipy.sparse.load_npz(MAIN_DIR + '/Data/base_contact_mtx/full_home.npz')
full_leisure_routine_matrix = scipy.sparse.load_npz(MAIN_DIR + '/Data/base_contact_mtx/full_leisure_routine.npz')
full_work_routine_matrix = scipy.sparse.load_npz(MAIN_DIR + '/Data/base_contact_mtx/full_work_routine.npz')
# Calculate the routine contact matrix of February - a fixed matrix that the model uses the whole operation time
# Taking the mean of all 3 matrices
final_contact_matrix = (1/3) * (full_work_routine_matrix + full_leisure_routine_matrix + full_home_matrix)

# Contact matrix dic
# with open('../Data/parameters/C_calibration.pickle', 'rb') as pickle_in:
# 	C_calibration = pickle.load(pickle_in)
# Contact matrix dic
# with open('../Data/parameters/C_const.pickle', 'rb') as pickle_in:
# 	C_const = pickle.load(pickle_in)

# Orthodox distribution
with open(MAIN_DIR + "/Data/parameters/orthodox_dist.pickle", 'rb') as pickle_in:
	is_haredi = pickle.load(pickle_in)

# Arab distribution
with open(MAIN_DIR + "/Data/parameters/arab_dist.pickle", 'rb') as pickle_in:
	is_arab = pickle.load(pickle_in)

# stay home index for behavior model
# with open('../Data/parameters/stay_home_idx.pickle', 'rb') as pickle_in:
# 	stay_home_idx = pickle.load(pickle_in)

# routine vector behavior model
# with open('../Data/parameters/routine_t.pickle', 'rb') as pickle_in:
# 	not_routine = pickle.load(pickle_in)

# Population size
with open(MAIN_DIR + '/Data/parameters/init_pop.pickle', 'rb') as pickle_in:
	population_size = pickle.load(pickle_in)

# Initial model (Starting conditions for the start date - currently 15th May
with open(MAIN_DIR + '/Data/parameters/init_model.pickle', 'rb') as pickle_in:
	init_model = pickle.load(pickle_in)

# We don't use it in our model - because we initiate the model from 15th May
# esp - model without sectors
# with open(MAIN_DIR + '/Data/parameters/eps_by_region.pickle', 'rb') as pickle_in:
# 	eps_sector = pickle.load(pickle_in)

# Yotam's addon - s_to_v1_transition parameter
with open(MAIN_DIR + '/Data/parameters/s_to_v1_transition.pickle', 'rb') as pickle_in:
	s_1_to_v1_transition_t = pickle.load(pickle_in)

# v2_to_v3_transition parameter
with open(MAIN_DIR + '/Data/parameters/v2_to_v3_transition.pickle', 'rb') as pickle_in:
	v2_to_v3_transition_t = pickle.load(pickle_in)

# Alpha variant vector
with open(MAIN_DIR + '/Data/parameters/alpha_variant_vec.pickle', 'rb') as pickle_in:
	alpha_variant_vec = pickle.load(pickle_in)

# Delta variant vector
with open(MAIN_DIR + '/Data/parameters/delta_variant_vec.pickle', 'rb') as pickle_in:
	delta_variant_vec = pickle.load(pickle_in)

# Omicron variant vector
with open(MAIN_DIR + '/Data/parameters/omicron_variant_vec.pickle', 'rb') as pickle_in:
	omicron_variant_vec = pickle.load(pickle_in)
	omicron_variant_vec = numpy.asarray(omicron_variant_vec[:529])

# Beta_lockdown vector
with open(MAIN_DIR + '/Data/parameters/lockdown_vec.pickle', 'rb') as pickle_in:
	lockdown_vec = pickle.load(pickle_in)

# Beta_school vector
with open(MAIN_DIR + '/Data/parameters/school_vec.pickle', 'rb') as pickle_in:
	school_vec = pickle.load(pickle_in)

# Activity vector
with open(MAIN_DIR + '/Data/parameters/activity_vec.pickle', 'rb') as pickle_in:
	visitors_covid = pickle.load(pickle_in)

# Isolation morbidity ratio vector
with open(MAIN_DIR + '/Data/parameters/isolation_morbidity_ratio_vector.pickle', 'rb') as pickle_in:
	isolation_morbidity_ratio_vector = pickle.load(pickle_in)

# index_t_to_date_mapper - {key == increment num : value == date} - for 10 years run
with open(MAIN_DIR + '/Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
	index_t_to_date_mapper = pickle.load(pickle_in)

# calib_index_t_to_date_mapper - {key == increment num : value == date} - only for the calibration dates
with open(MAIN_DIR + '/Data/parameters/calib_index_t_to_date_mapper.pickle', 'rb') as pickle_in:
	calib_index_t_to_date_mapper = pickle.load(pickle_in)

# Reading the dictionary of the calibration's results:
with open(MAIN_DIR + '/Data//calibration/calibration_dict.pickle', 'rb') as pickle_in:
	cal_parameters = pickle.load(pickle_in)

# Beta_home - home transmissibility:
# with open('../Data/parameters/beta_home.pickle', 'rb') as pickle_in:
# 	beta_home = pickle.load(pickle_in)
# beta_home = 0.38/9

#  gama - transition rate between Is,Ia to R
gama = 1. / 9.5

# delta - transition rate E to Is, Ia
delta = 1. / 5.

# # sigma - transition rate E to Ie
# sigma = 1. / 3.5

# fixing parameter for beta_home
psi = 1

# nu - Transition rate out of H
nu = 0.07247553272296633

# mu - Transition rate out of Vents
mu = 0.07142857142857142

# eta - Transition rate H_latent to H
eta = 0.8676140711537818

# xi - Transition rate Vents_latent to Vents
xi = 0.23638829809201067

# alpha - factor for the probability for hospitalizations, according to the paper
alpha = 1 - 0.87

## Waining time
# omega - waining rate from recovered to susceptible (expectation of exponential dist. of 9 months)
omega = 1/270
# tau - waining rate from V2 to S2
tau = 1/180

# fertility rates:
fertility_rate_orthodox = 7.1/3.68		# From past calibration
fertility_rate_arabs = 3.33/2.68		# From past calibration
fertility_rate_sacular = 1

# Initializing a dictionary of events
events_dict = {}
events_dict['second_lockdown'] = ('2020-09-13', '2020-10-15')
events_dict['third_lockdown'] = ['2020-12-27', '2021-02-04']

events_dict['school_opening_1'] = ['2020-05-15', '2020-07-01']
events_dict['school_opening_2'] = ['2020-09-01', '2020-09-18']
events_dict['school_opening_3'] = ['2020-11-15', '2020-12-26']
events_dict['school_opening_4'] = ['2021-05-18', '2021-07-01']

events_dict['british_variant_alpha'] = ('2020-12-15', '2021-04-15')
events_dict['indian_variant_delta'] = ['2021-04-15', '2021-08-15']
# Omicron dates (from first discovery in israel until emergence of 2 subvariants of omicron - BA4 & BA5:
events_dict['african_variant_omicron'] = ['2022-01-01', '2022-04-01']


# Israel's population size
pop_israel = 9345000

## Transition rates to/from the vaccinations compartments
# all transitions from the vaccination compartments to E_2 are defined in the model's script (we account for the force of infection in the calculations)
v1_to_v2_transition = 1/21		# 21 days expectation
#v2_to_v3_transition = 1/183		# taken from the analysis

v1_to_e_transition = 0.0   		# Initialization. We update this value later on model_class module, in the predict method. This value is dependent in the force of infection (lambda_t int the model class)
v2_to_e_transition = 0.0		# Initialization. same explanation as v1_to_e_transition
v3_efficiency = 0.95				# Initialization. We want to control this variable when simulating the results, Hence the difference from the previous two transitions.
v3_to_e_transition = 0.0		# Initialization. same explanation as v1_to_e_transition


