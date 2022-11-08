#!/usr/bin/env python
# coding: utf-8

# ## This notebook produces cases per year for the morbidity based uniform simulation

# In[1]:


import numpy as np
import pandas as pd
import itertools
import pickle
import datetime, random, copy
import sys, os

import SEIR_full as mdl
import SEIR_full.model as mdl
import SEIR_full.calibration as mdl
from SEIR_full.indices import *
from SEIR_full.utils import *
from SEIR_full.parameters import *


# * Fixing the ranges of the uncertainty factors that we're interested in:

# In[2]:


# Creating a list of tuples, each tuple is a realization of randomly picked values for the factors that we change every iteration of the simulation
np.random.seed(82)
list_of_tuples = []
for i in range(1000):
    booster_eff = round(np.random.uniform(low=29.5001, high=95.5)) / 100     # Discrete uniform dist, booster_efficiency ~ U(30, 95), Updated in the model's object
    inv_level = round(np.random.uniform(low=9.5001, high=20.5)) / 100       # Discrete uniform dist, booster_efficiency ~ U(10, 20), Updated in the simulation's settings
    hosp_proba_scalar = round(np.random.uniform(low=0.8, high=1.2), 2)      # Continuous uniform dist, booster_efficiency ~ U(0.8, 1.2), Updated in the model's object
    years_for_model_run = round(np.random.uniform(low=2.5001, high=10.5))         # Discrete uniform dist, booster_efficiency ~ U(3, 10), Updated in the simulation's settings
    list_of_tuples.append((booster_eff, inv_level, hosp_proba_scalar, years_for_model_run))


# * Fixing the static settings for all runs of the big loop

# In[3]:


## DEFINING STATIC SETTINGS (INDEPENDENT IN THE DYNAMIC FACTORS) FOR THE SIMULATION RUN
# Short path for the data directory
DATA_DIR = r'/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/Data'
# Reading the indices of the model - adding an hint to declare it's from type Indices!
with(open(DATA_DIR + '/parameters/indices.pickle', 'rb')) as openfile:
    ind:Indices = pickle.load(openfile)

# Create indices for the age groups for easy & automatic access to the results of the operation in the script:
age_groups_ind = {'5-9':0, '10-29':1, '30-59':2, '60+':3}

# Reading the neutralized vectors:
# Alpha variant vector
with open(DATA_DIR + '/parameters/neutralized_alpha_variant_vec.pickle', 'rb') as pickle_in:
	neutralized_alpha_variant_vec = pickle.load(pickle_in)

# Delta variant vector
with open(DATA_DIR + '/parameters/neutralized_delta_variant_vec.pickle', 'rb') as pickle_in:
	neutralized_delta_variant_vec = pickle.load(pickle_in)

# Beta_lockdown vector
with open(DATA_DIR + '/parameters/neutralized_lockdown_vec.pickle', 'rb') as pickle_in:
	neutralized_lockdown_vec = pickle.load(pickle_in)

# Beta_school vector
with open(DATA_DIR + '/parameters/neutralized_school_vec.pickle', 'rb') as pickle_in:
	neutralized_school_vec = pickle.load(pickle_in)

# Isolation morbidity ratio vector
with open(DATA_DIR + '/parameters/neutralized_isolation_morbidity_ratio_vector.pickle', 'rb') as pickle_in:
	neutralized_isolation_morbidity_ratio_vector = pickle.load(pickle_in)

# zero vector to remove any transition from V_2 and S_2 to V_3
with open(DATA_DIR + '/parameters/neutralized_transition_rate_to_V_3.pickle', 'rb') as pickle_in:
	neutralized_transition_vector = pickle.load(pickle_in)


# * Utils functions:

# In[4]:


# Getting the reported unreported ratio
def get_reported_unreported_ratio(scen):
    if scen == 'Scenario2':
        reported = 1
        unreported = 2
        reported_unreported = unreported / (reported + unreported)
    return reported_unreported

# Getting all region,age combinations as a list
def get_all_region_age_combinations():
    """This function returns a list of tuples of all region,risk,4-age_group combinations"""
    counties_list = list(ind.region_dict.keys())
    age_groups_list = list(age_groups_ind.keys())
    return list(itertools.product(counties_list, age_groups_list))

def region_age_groups_mapping(wanted_mapping: str):
    """ This function gets a wanted mapping (4 examined age groups to 9 original model's age groups or vice versa),
    and returns a dictionary consist of the wanted mapping
    """
    # Initializing the final mapping to return and it's keys:
    mapping_dict = {}
    all_comb_4_age_groups_list = get_all_region_age_combinations()
    if wanted_mapping == '4-to-9':
        # Iterating over all possible combinations with the 4-age_groups:
        for combination in all_comb_4_age_groups_list:
            # The list that the current combination will map to
            curr_res_list = []
            # A list to iterate when building the curr_res_list - depending on the current and new age-group:
            if combination[1] == '5-9':
                correspondent_age_groups = ['5-9']
            elif combination[1] == '10-29':
                correspondent_age_groups = ['10-19', '20-29']
            elif combination[1] == '30-59':
                correspondent_age_groups = ['30-39', '40-49', '50-59']
            else:
                correspondent_age_groups = ['60-69', '70+']

            # Appending to the curr_res_list to finalize the mapping:
            for age_group in correspondent_age_groups:
                curr_res_list.append((combination[0], age_group))

            # Updating the final mapping_dict:
            mapping_dict[combination] = curr_res_list
    return mapping_dict

# Getting all region,risk,age combinations as a list
def get_all_region_risk_age_combinations():
    """This function returns a list of tuples of all region,risk,4-age_group combinations"""
    counties_list = list(ind.region_dict.keys())
    risk_list = ['High', 'Low']
    age_groups_list = list(age_groups_ind.keys())
    return list(itertools.product(counties_list, risk_list, age_groups_list))

def region_risk_age_groups_mapping(wanted_mapping: str):
    """ This function gets a wanted mapping (4 examined age groups to 9 original model's age groups or vice versa),
    and returns a dictionary consist of the wanted mapping
    """
    # Initializing the final mapping to return and it's keys:
    mapping_dict = {}
    all_comb_4_age_groups_list = get_all_region_risk_age_combinations()
    if wanted_mapping == '4-to-9':
        # Iterating over all possible combinations with the 4-age_groups:
        for combination in all_comb_4_age_groups_list:
            # The list that the current combination will map to
            curr_res_list = []
            # A list to iterate when building the curr_res_list - depending on the current and new age-group:
            if combination[2] == '5-9':
                correspondent_age_groups = ['5-9']
            elif combination[2] == '10-29':
                correspondent_age_groups = ['10-19', '20-29']
            elif combination[2] == '30-59':
                correspondent_age_groups = ['30-39', '40-49', '50-59']
            else:
                correspondent_age_groups = ['60-69', '70+']

            # Appending to the curr_res_list to finalize the mapping:
            for age_group in correspondent_age_groups:
                curr_res_list.append((combination[0], combination[1], age_group))

            # Updating the final mapping_dict:
            mapping_dict[combination] = curr_res_list
    return mapping_dict

def get_indexes_region_risk_age_combination(region_risk_age_group: tuple):
    """This function gets a tuple of (region, risk, 4_age_group) and returns the indexes of that combination as they are in the original model (including transformation to the 9 original age groups. e.g.: for (1101, 'High', '10-29'), returns the indexes for (1101, 'High', '10-19') + (1101, 'High', '20-29')"""
    # Initializing the result list and the age groups mapper:
    indexes_list_of_lists = []
    four_to_nine_age_map = region_risk_age_groups_mapping('4-to-9')
    for val in four_to_nine_age_map[region_risk_age_group]:
        indexes_list_of_lists.append(ind.region_risk_age_dict[val])
    # Merging the list of lists to one list:
    region_risk_age_indexes_list = [item for sublist in indexes_list_of_lists for item in sublist]
    return sorted(region_risk_age_indexes_list)

# Getting the current lambda_t for the current combination of (region, 4-age_group):
def get_current_lambda_t(curr_res_mdl_1_ml: dict, target_region_risk_4age_group: tuple):
    # Getting the lambda of the last day of the current model, and the region,4-age_group mapping:
    lambda_last_day = np.reshape(curr_res_mdl_1_ml['L_2'][-1, :], newshape=(540))
    four_to_nine_age_map = region_age_groups_mapping('4-to-9')
    # Initializing the final proportion for the current compartment that we want to calculate:
    final_prop_for_comp = 0
    # Iterating on each correspondent combination of (region, 9-age_group) to aggregate on:
    for val in four_to_nine_age_map[target_region_risk_4age_group]:
        # Getting the index of the current (region, risk, 9-age_group)
        correspondent_index = ind.region_age_dict[val]
        # Adding the value of current (region, risk, 9-age_group)
        final_prop_for_comp += lambda_last_day[correspondent_index].sum()

    return final_prop_for_comp

def calc_comp_prop_for_region_risk_age(comp: str, curr_res_mdl_1_ml: dict, target_region_risk_4age_group: tuple):
    """This function gets the target compartment, the current predictions of the current model, and the wanted combination of (region, risk, 4-age_group),
    and returns the compartment aggregated by the wanted combination of (region, risk, 4-age_group) - (aggregated from (region, risk, 9-age_group))"""
    # Getting the last day of the compartment, and the needed age-groups mapping:
    comp_last_day = np.reshape(curr_res_mdl_1_ml[comp][-1, :], newshape=(540))
    four_to_nine_age_map = region_risk_age_groups_mapping('4-to-9')
    # Initializing the final proportion for the current compartment that we want to calculate:
    final_prop_for_comp = 0
    # Iterating on each correspondent combination of (region, risk, 9-age_group) to aggregate on:
    for val in four_to_nine_age_map[target_region_risk_4age_group]:
        # Getting the index of the current (region, risk, 9-age_group)
        correspondent_index = ind.region_risk_age_dict[val]
        # Adding the value of current (region, risk, 9-age_group)
        final_prop_for_comp += float(comp_last_day[correspondent_index][0])

    return final_prop_for_comp

def calc_rho_or_f_prop_for_region_risk_age(rho_or_f_j: np.array, target_region_risk_4age_group: tuple):
    """This function gets the model_1_ml.rho and the wanted combination of (region, risk, 4-age_group),
    and returns rho aggregated by the wanted combination of (region, risk, 4-age_group) - (aggregated from (region, risk, 9-age_group))"""
    four_to_nine_age_map = region_risk_age_groups_mapping('4-to-9')
    ## Calculating the final proportion of rho or f_j as a weighted sum using the distribution of S(0) (population_size):
    # Initializing the proportions for the weighted sum:
    S0_list = []
    rho_or_f_j_list = []
    # Iterating on each correspondent combination of (region, risk, 9-age_group) and append to the relevant list:
    for val in four_to_nine_age_map[target_region_risk_4age_group]:
        # Getting the index of the current (region, risk, 9-age_group)
        correspondent_index = ind.region_risk_age_dict[val]
        # Adding the value of current (region, risk, 9-age_group)
        S0_list.append(population_size[correspondent_index][0])
        rho_or_f_j_list.append(rho_or_f_j[correspondent_index][0])

    ## Calculating the weighted sum:
    # Init the lists to calc the expression
    numerator = []
    denominator = []
    for i in range(len(S0_list)):
        numerator.append(S0_list[i]*rho_or_f_j_list[i])
        denominator.append(S0_list[i])
    # Returning the final and weighted probability value:
    final_weighted_sum_proba = np.sum(numerator) / np.sum(denominator)
    return final_weighted_sum_proba

def update_compartment(curr_res_mdl_1_ml: dict, compartment: str, updated_array: np.array, model_1_ml_: any):
    """This function gets the current models' predictions, the compartment to update and the updated array, and updates the model object with the updated array"""
    # Initiate settings:
    curr_res_mdl_1_ml[compartment][-1] = updated_array
    comp_as_list = []
    # Changing the updated arrays to be lists of arrays (to allow the update of the model object):
    for element in curr_res_mdl_1_ml[compartment]:
        comp_as_list.append(element)

    # Performing the proper model update
    model_1_ml_.update({
        compartment : comp_as_list
    })

# Defining the model - initializing it for t=0:
def defining_model(scen):
    model_1_ml_ = mdl.Model_behave(
        ind=ind,
        beta_j=cal_parameters[scen]['beta_j'],
        beta_activity=cal_parameters[scen]['beta_activity'],
        beta_school=cal_parameters[scen]['beta school'],
        scen=scen
    )
    # Predicting without an assignment (there's no need for that)
    model_1_ml_.predict(
        days_in_season= 529,    # Num of days between 15/05/20 - 25/10/21
        shifting_12_days= True
    )
    # Updating the vectors to be able to run the model furthermore:
    model_1_ml_.update({
        'alpha_variant_vec':  neutralized_alpha_variant_vec,
        'delta_variant_vec': neutralized_delta_variant_vec,
        'isolation_morbidity_ratio_vector': neutralized_isolation_morbidity_ratio_vector,
        'is_lockdown': neutralized_lockdown_vec,
        'is_school': neutralized_school_vec,
        'v2_to_v3_transition_t' : neutralized_transition_vector,
        's_2_to_v3_transition_t' : neutralized_transition_vector
    })
    return model_1_ml_

# Update model for current variables of uncertainty:
def update_models_object(mdl_1_ml: mdl.Model_behave, realization: tuple):
    new_rho = mdl_1_ml.rho * realization[2]
    mdl_1_ml.update({
        'booster_efficiency': realization[0],
        'rho': new_rho
    })


# #### Getting the vaccinations priority queue:

# In[5]:


def get_vaccination_pq(curr_res_mdl_1_ml: dict, model_1_ml_: any):
    """This function gets the current model and returns the final priority queue for all combinations of (county, risk, 4-age_group)"""
    # Initialize settings: final priority queue to return (as a dictionary, not sorted), the 4-9 age-groups mapper, and the reported/unreported ratio:
    vaccination_que = {}
    reported_unreported = get_reported_unreported_ratio('Scenario2')
    comp_list = ['S_2', 'V_2', 'R_1', 'R_2', 'L_2']
    mapping_4_to_9 = region_risk_age_groups_mapping('4-to-9')

    for key, val in mapping_4_to_9.items():
        # Getting the proportion value for each compartment[region,risk,4-age_group], save it as a list such that the order is exactly - [S_2, V_2, R_1, R_2, L_2]
        current_comp_prop_list = []
        # For each compartment - calculate its value of (region, risk, 4-age_group):
        for comp in comp_list:
            # If we assess the proportion of the force of infection, the calculation is different (lambda_t depends only on (region,age))
            if comp != 'L_2':
                current_comp_prop_list.append(calc_comp_prop_for_region_risk_age(comp, curr_res_mdl_1_ml, key))
            else:
                current_comp_prop_list.append(get_current_lambda_t( curr_res_mdl_1_ml, (key[0], key[2]) ))
        # getting the proportion value for model_1_ml.rho:
        hosp_proba = calc_rho_or_f_prop_for_region_risk_age(model_1_ml_.rho , key)
        # getting the proportion value for model_1_ml.f[model_1_ml.scen]:
        f_j = calc_rho_or_f_prop_for_region_risk_age(model_1_ml_.f[model_1_ml_.scen] , key)
        # Calculating the hospitalization probability measurement:
        comps = current_comp_prop_list      # To shorten the code
        V_2_eff = 1-0.94        # Effectiveness of second dose to prevent infection
        scoring_index = ( (comps[0] + (comps[2]+comps[3])*reported_unreported + (comps[1]*(V_2_eff))) * comps[4] * ((1-f_j)*model_1_ml_.delta) * hosp_proba )                         / (comps[0] + comps[1] + (comps[2]+comps[3])*reported_unreported)
        # Assigning the scoring measurement to the correspondent combination of (region,risk,4-age_group)
        vaccination_que[key] = scoring_index
    # Sorting the dict to get a "priority queue" - receiving a list of tuples:
    vaccination_que = sorted(vaccination_que.items(), key=lambda x: x[1], reverse= True)

    return vaccination_que


# #### calculating the transferring proportion:

# In[6]:


def calc_trans_prop(vaccination_pq_: list, curr_res_mdl_1_ml: dict, curr_inv_: float):
    # Initializing the accumulated amount used and the priority queue:
    curr_inventory_used = 0
    vaccination_pq_copy = vaccination_pq_.copy()
    # Saving the combinations that we will vaccinate:
    vaccinated_que_ = []
    # Iterating until we cross the prior inventory level:
    while (curr_inventory_used <= curr_inv_):
        # Getting the current combination of (age_group, risk) from the pq:
        current_county_risk_age = vaccination_pq_copy.pop(0)[0]
        # Getting the correspondent reported/unreported ratio for the Scenario:
        reported_unreported = get_reported_unreported_ratio('Scenario2')
        # Getting R2_(j,r), R1_(j,r), V2_(j,r), S2_(j,r) proportions:
        R_2_risk_age, R_1_risk_age, S_2_risk_age, V_2_risk_age = calc_comp_prop_for_region_risk_age('R_2', curr_res_mdl_1_ml, current_county_risk_age),                                                                  calc_comp_prop_for_region_risk_age('R_1', curr_res_mdl_1_ml, current_county_risk_age),                                                                  calc_comp_prop_for_region_risk_age('S_2', curr_res_mdl_1_ml, current_county_risk_age),                                                                  calc_comp_prop_for_region_risk_age('V_2', curr_res_mdl_1_ml, current_county_risk_age)
        # Calculating the formula for num of needed vaccines:
        count_vaccines_in_use = ( ((R_1_risk_age + R_2_risk_age) * reported_unreported) + S_2_risk_age + V_2_risk_age) * pop_israel
        # Updating the accumulated used vaccines:
        curr_inventory_used += count_vaccines_in_use
        # Updating the vaccinated_que:
        vaccinated_que_.append(current_county_risk_age)

    final_trans_prop = curr_inv_ / curr_inventory_used
    return final_trans_prop, vaccinated_que_


# #### Vaccinating in booster dose:

# In[7]:


### Vaccinating in booster (moving from S_2 and V_2 to V_3, including the model update):
def vaccinate(trans_prop_: float, curr_res_mdl_1_ml: dict, vaccination_que: list, model_1_ml__):
    ## initialize settings:
    vaccination_que_copy = vaccination_que.copy()
    trans_prop_copy = trans_prop_
    reported_unreported = get_reported_unreported_ratio('Scenario2')
    # Getting the last day of the compartments and V_3, saving it to a dictionary:
    last_day_dict = {'S_2_last_day': np.reshape(curr_res_mdl_1_ml['S_2'][-1, :], newshape=(540) ),
                     'V_2_last_day': np.reshape(curr_res_mdl_1_ml['V_2'][-1, :], newshape=(540) ),
                     'V_3_last_day': np.reshape(curr_res_mdl_1_ml['V_3'][-1, :], newshape=(540) )}

    # Iterating over the combinations of (risk, age_group) that we need to vaccinate, and moving the population, and updating the model:
    for curr_risk_age in vaccination_que_copy:
        # Getting the relevant indexes to operate in the correspondent locations of the compartment:
        curr_risk_age_indexes = get_indexes_region_risk_age_combination(curr_risk_age)

        ## Vaccinating from S_2 to V_3
        S_2_last_day_updated, V_3_last_day_updated = vaccinating_from_V_2_S_2(last_day_dict['V_3_last_day'], last_day_dict['S_2_last_day'], trans_prop_copy, curr_risk_age_indexes)
        # updating the results:
        last_day_dict['V_3_last_day'], last_day_dict['S_2_last_day'] = V_3_last_day_updated, S_2_last_day_updated

        ## Vaccinating from V_2 to V_3
        V_2_last_day_updated, V_3_last_day_updated = vaccinating_from_V_2_S_2(last_day_dict['V_3_last_day'], last_day_dict['V_2_last_day'], trans_prop_copy, curr_risk_age_indexes)
        # updating the results:
        last_day_dict['V_3_last_day'], last_day_dict['V_2_last_day'] = V_3_last_day_updated, V_2_last_day_updated

    update_compartment(curr_res_mdl_1_ml, 'V_2', last_day_dict['V_2_last_day'], model_1_ml__)
    update_compartment(curr_res_mdl_1_ml, 'S_2', last_day_dict['S_2_last_day'], model_1_ml__)
    update_compartment(curr_res_mdl_1_ml, 'V_3', last_day_dict['V_3_last_day'], model_1_ml__)


# ### Main - 1000 iterations of the Sim:

# In[ ]:

#
# # A list to hold 1000 realizations of cases per year:
# cases_per_year_list = []
#
# # Big loop run:
# for tup in list_of_tuples:
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario2')
#     # Settings:
#     update_models_object(model_1_ml, tup)       # Updating the v3_efficiency and the proba for hosp
#     inv_level = tup[1] * pop_israel             # Updating the inventory level
#     years_for_model_run = tup[3]                # Updating the years for the model to run for
#
#     # Running the model for X years, in 6 months resolution
#     for i in range(years_for_model_run*2):
#         # Running until the vaccination month:
#         for j in range(6):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#             # Getting the vaccination priority_que:
#             vaccination_pq = get_vaccination_pq(res_mdl_1_ml)
#             # Getting the proportion of transition, and the combinations that we vaccinate:
#             trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, inv_level*1/6)
#             # Vaccinating (including the model update):
#             vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#     ## Calculating the cases per year index for every measurement:
#     tot_new_Is = np.add(res_mdl_1_ml['new_Is_1'].sum(axis=1) * pop_israel, res_mdl_1_ml['new_Is_2'].sum(axis=1) * pop_israel)
#     tot_new_H = np.add(res_mdl_1_ml['new_H_1'].sum(axis=1) * pop_israel, res_mdl_1_ml['new_H_2'].sum(axis=1) * pop_israel)
#     # Transferring to cumulative sum:
#     cumsum_tot_new_Is, cumsum_tot_new_H = np.cumsum(tot_new_Is), np.cumsum(tot_new_H)
#     # Performing the calculation itself:
#     cases_per_year_new_Is, cases_per_year_new_H = (cumsum_tot_new_Is[-1] - cumsum_tot_new_Is[528]) / years_for_model_run, (cumsum_tot_new_H[-1] - cumsum_tot_new_H[528]) / years_for_model_run
#     cases_per_year_list.append((cases_per_year_new_Is, cases_per_year_new_H))
#
#
# # In[ ]:
#
#
# cases_per_year_list
#
#
# # In[ ]:
#
#
# with open('cases_per_year_list.pickle', 'wb') as handle:
#     pickle.dump(cases_per_year_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

