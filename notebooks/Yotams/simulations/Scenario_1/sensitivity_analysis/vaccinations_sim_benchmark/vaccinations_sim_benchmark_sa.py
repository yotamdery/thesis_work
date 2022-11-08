#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing
import itertools
import sys
import os
import numpy as np
import pandas as pd
import pickle
import SEIR_full as mdl
from SEIR_full.indices import *
from SEIR_full.parameters import *
from SEIR_full.utils import *
from SEIR_full.plot_utils import  *
#from SEIR_full.model import mdl


# In[2]:


# Creating a directory for the images:
sys.path.append(os.getcwd())
if not os.path.exists("images"):
    os.mkdir("images")


# ### Utils:

# In[3]:


# Short path for the data directory
DATA_DIR = r'/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/Data'


# In[4]:


# Reading the indices of the model - adding an hint to declare it's from type Indices!
with(open(DATA_DIR + '/parameters/indices.pickle', 'rb')) as openfile:
    ind:Indices = pickle.load(openfile)


# In[5]:


# Create indices for the age groups for easy & automatic access to the results of the operation in the script:
age_groups_ind = {'5-9':0, '10-29':1, '30-59':2, '60+':3}


# In[6]:


# Defining the time periods for a "single run" of six months:
time_periods_single_run = [i for i in range(1,7)]
# Vaccination inventory - 3 options: equal to 25% of the israels' total population, 50%, and 75%:
vaccination_inventory = [9345000 * 0.1, 9345000 * 0.15, 9345000 * 0.2]
# Creating a dict of (key = time_period_t, value= inventory):
all_t_inv_dict = {}
for t in time_periods_single_run:
    for inv in vaccination_inventory:
        all_t_inv_dict[t] = vaccination_inventory


# In[7]:


# Vaccination priority queue:
vaccination_pq = [('High', '60+'), ('High', '30-59'), ('High', '10-29'), ('High', '5-9'),
                   ('Low', '60+'), ('Low', '30-59'), ('Low', '10-29'), ('Low', '5-9')
                  ]


# In[8]:


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


# ###Functions:
# 

# In[9]:
def get_reported_unreported_ratio(scen):
    if scen == 'Scenario1':
        reported = 1
        unreported = 1
        reported_unreported = unreported / (reported + unreported)
    return reported_unreported

# In[10]:


def risk_age_groups_mapping(wanted_mapping: tuple):
    """ This function gets a wanted mapping (4 examined age groups to 9 original model's age groups or vice versa),
    and returns a dictionary consist of the wanted mapping
    """
    age_groups_mapping_dict = {}
    # If the wanted mapping is 4 examined age groups to 9 original model's age groups:
    if wanted_mapping == ('High', '4-to-9'):
        age_groups_mapping_dict[('High', '5-9')] = [('High', '5-9')]
        age_groups_mapping_dict[('High', '10-29')] = [('High', '10-19'), ('High', '20-29')]
        age_groups_mapping_dict[('High', '30-59')] = [('High','30-39'), ('High','40-49'), ('High','50-59')]
        age_groups_mapping_dict[('High', '60+')] = [('High', '60-69'), ('High', '70+')]

    elif wanted_mapping == ('Low', '4-to-9'):
        age_groups_mapping_dict[('Low', '5-9')] = [('Low', '5-9')]
        age_groups_mapping_dict[('Low', '10-29')] = [('Low', '10-19'), ('Low', '20-29')]
        age_groups_mapping_dict[('Low', '30-59')] = [('Low','30-39'), ('Low','40-49'), ('Low','50-59')]
        age_groups_mapping_dict[('Low', '60+')] = [('Low', '60-69'), ('Low', '70+')]

    # If the wanted mapping is 9 original model's age groups to 4 examined age groups:
    elif wanted_mapping == ('High', '9-to-4'):
        age_groups_mapping_dict[('High', '0-4')] = np.nan     # There's no corresponding age group among the 4 examined age-groups
        age_groups_mapping_dict[('High', '5-9')] = ('High', '5-9')
        age_groups_mapping_dict[('High', '10-19')] = ('High', '10-29')
        age_groups_mapping_dict[('High', '20-29')] = ('High', '10-29')
        age_groups_mapping_dict[('High', '30-39')] = ('High', '30-59')
        age_groups_mapping_dict[('High', '40-49')] = ('High', '30-59')
        age_groups_mapping_dict[('High', '50-59')] = ('High', '30-59')
        age_groups_mapping_dict[('High', '60-69')] = ('High', '60+')
        age_groups_mapping_dict[('High', '70+')] = ('High', '60+')

    else:
        age_groups_mapping_dict[('low', '0-4')] = np.nan     # There's no corresponding age group among the 4 examined age-groups
        age_groups_mapping_dict[('low', '5-9')] = ('low', '5-9')
        age_groups_mapping_dict[('low', '10-19')] = ('low', '10-29')
        age_groups_mapping_dict[('low', '20-29')] = ('low', '10-29')
        age_groups_mapping_dict[('low', '30-39')] = ('low', '30-59')
        age_groups_mapping_dict[('low', '40-49')] = ('low', '30-59')
        age_groups_mapping_dict[('low', '50-59')] = ('low', '30-59')
        age_groups_mapping_dict[('low', '60-69')] = ('low', '60+')
        age_groups_mapping_dict[('low', '70+')] = ('low', '60+')

    return age_groups_mapping_dict


# In[11]:


def get_wanted_df(wanted_age_group):
    """ Gets the age-groups that we want to address our calculations to, and returns a sub-Dataframe containing only the relevant columns (the wanted age-groups columns)"""
    # Reading the pop per county file:
    county_pop = pd.read_csv(DATA_DIR + '/division_choice/30/population_per_county_age-group.csv')
    # Initializing the list with the county column, that will be picked anyway
    wanted_columns = [county_pop.columns[0]]
    # Splitting the given age_group, creating list of integers. try-expect block to catch the 60+ column as well:
    try:
        wanted_age_group_splitted = list(map(int, wanted_age_group.split('-')))
    except ValueError:
        wanted_age_group_splitted = [60, 70]  # In order to pass the logic condition successfully

    # Iterating on the columns of the file and performing the condition logic:
    for col in county_pop.columns[1:]:
        # Splitting the current column of the file. try-expect block to catch the 70+ column as well:
        try:
            ages_as_list = list(map(int, col.split('_')[1].split('-')))
        except ValueError:
            ages_as_list = [70]
        # If these conditions are met, take this column to be in the sub-df
        if ages_as_list[0] >= min(wanted_age_group_splitted) and ages_as_list[0] <= max(wanted_age_group_splitted):
            wanted_columns.append(col)

    return county_pop[wanted_columns]


# In[12]:


# Gets a dictionary to reduce (lambda, V_2 or S_2) and reduce its cardinality to consist only the relevant age-groups
def calc_sub_dict(dict_to_reduce: dict, wanted_age_group: str):
    sub_dict = {}
    # Splitting the given age_group, creating list of integers. try-expect block to catch the 60+ group as well (if wanted):
    try:
        wanted_age_group_splitted = list(map(int, wanted_age_group.split('-')))
    except ValueError:
        wanted_age_group_splitted = [60, 70]    # In order to pass the logic condition successfully

    for key, val in dict_to_reduce.items():
        # Splitting the age group part of the key. try-expect block to catch the 70+ column from the file as well:
        try:
            if isinstance(key, str):   # If the dict_to_reduce keys are only on the age level
                ages_as_list = list(map(int, key.split('-')))
            elif isinstance(key, tuple):    # If the dict_to_reduce keys are on the region&age level (as lambda_t does) - take the second element (the age)
                ages_as_list = list(map(int, key[1].split('-')))
        except ValueError:
            ages_as_list = [70]
        # If these conditions are met, take this key-value pair to be in the sub-dict
        if ages_as_list[0] >= min(wanted_age_group_splitted) and ages_as_list[0] <= max(wanted_age_group_splitted):
            sub_dict[key] = val    # Converting from ndarray to float

    return sub_dict


# In[13]:


def get_indexes_risk_age_combination(risk_age_group: tuple):
    """This function gets a tuple of (risk, 4_age_group) and returns the indexes of that combination as they are in the original model (including transformation to the 9 original age groups
    e.g.: for ('High', '10-29'), returns the indexes for ('High', '10-19') + ('High', '20-29') """
    # Initializing the result list and the age groups mapper:
    indexes_list_of_lists = []
    four_to_nine_age_map = risk_age_groups_mapping((risk_age_group[0], '4-to-9'))
    for val in four_to_nine_age_map[risk_age_group]:
            indexes_list_of_lists.append(ind.risk_age_dict[val])
    # Merging the list of lists to one list:
    risk_indexes_list = [item for sublist in indexes_list_of_lists for item in sublist]
    return sorted(risk_indexes_list)


# In[14]:


# Gets the compartment to aggregate, and returns the same compartment, stratified by the original age-groups
def calc_aggregated_compartment_by_age_risk(compartment: np.array):
    compartment_dict = {}
    for risk_age in ind.risk_age_dict.keys():
        compartment_dict[risk_age] = float(compartment[:, ind.risk_age_dict[risk_age]].sum(axis=1))
    return compartment_dict


# In[15]:


# Gets the compartment dictionary (the compartment stratified by the old age-groups), and aggregates the compartment to the wanted age-group by returning a dict (e.g agg. V_2 for age group 30-59, using the sub-age-groups 30-39, 40-49, 50-59)
def compartment_by_4_age_groups(compartment: dict, target_risk_age_group: tuple):
    # Initiate the wanted mapping:
    four_to_nine_age_map = risk_age_groups_mapping((target_risk_age_group[0], '4-to-9'))
    # Initiate result dict to return:
    agg_risk_4_groups_dict = {}
    for risk_4_age_group in four_to_nine_age_map.keys():     # Iterating on each combination of (risk, 9 age-groups):
        agg_risk_4_groups_dict[risk_4_age_group] = 0
    # Performing the aggregation to the new dict - (risk, 4 age groups):
    for key, val in four_to_nine_age_map.items():
        for tup in val:
            agg_risk_4_groups_dict[key] += compartment[tup]

    return agg_risk_4_groups_dict


# In[16]:


def calc_comp_prop_for_age_risk(comp: str, curr_res_mdl_1_ml: dict, target_risk_age_group: tuple):
    """This function gets the target compartment and the current predictions of the current model,
    and returns the compartment aggregated by the wanted combination of (age-group, risk)"""
    # Getting the last day of the compartment:
    comp_last_day = np.reshape(curr_res_mdl_1_ml[comp][-1, :], newshape=(1,540))
    # Aggregate the compartment by (9 age groups, risk):
    comp_agg_9_age_groups_risk = calc_aggregated_compartment_by_age_risk(comp_last_day)
    # Aggregate the compartment by (4 age groups, risk):
    comp_agg_4_age_groups_risk = compartment_by_4_age_groups(comp_agg_9_age_groups_risk, target_risk_age_group)
    # Gets the relevant age group out of the 4 age groups, and summing on it to get the final proportion:
    return comp_agg_4_age_groups_risk[target_risk_age_group]


# In[17]:


def update_compartment(curr_res_mdl_1_ml: dict, compartment: str, updated_array: np.array, model_1_ml__):
    """This function gets the current models' predictions, the compartment to update and the updated array, and updates the model object with the updated array"""
    # Initiate settings:
    curr_res_mdl_1_ml[compartment][-1] = updated_array
    comp_as_list = []
    # Changing the updated arrays to be lists of arrays (to allow the update of the model object):
    for element in curr_res_mdl_1_ml[compartment]:
        comp_as_list.append(element)

    # Performing the proper model update
    model_1_ml__.update({
        compartment : comp_as_list
    })


# In[18]:


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


# ### calculating the transferring proportion:

# In[19]:


def calc_trans_prop(vaccination_pq_: list, curr_res_mdl_1_ml: dict, curr_inv_: float):
    # Initializing the accumulated amount used and the priority queue:
    curr_inventory_used = 0
    vaccination_pq_copy = vaccination_pq_.copy()
    # Saving the combinations that we will vaccinate:
    vaccinated_que_ = []
    # Iterating until we cross the prior inventory level:
    while (curr_inventory_used <= curr_inv_):
        # Getting the current combination of (age_group, risk) from the pq:
        current_risk_age = vaccination_pq_copy.pop(0)
        # Getting the correspondent reported/unreported ratio for the Scenario:
        reported_unreported = get_reported_unreported_ratio('Scenario1')
        # Getting V2_j,r, S2_j,r proportions:
        S_2_age_risk, V_2_age_risk, R_2_age_risk, R_1_age_risk = calc_comp_prop_for_age_risk('S_2', curr_res_mdl_1_ml,current_risk_age),                                                                  calc_comp_prop_for_age_risk('V_2', curr_res_mdl_1_ml,current_risk_age),                                                                  calc_comp_prop_for_age_risk('R_2', curr_res_mdl_1_ml,current_risk_age),                                                                  calc_comp_prop_for_age_risk('R_1', curr_res_mdl_1_ml,current_risk_age)
        # Calculating the formula for num of needed vaccines:
        count_vaccines_in_use = (((R_1_age_risk + R_2_age_risk) * reported_unreported) + S_2_age_risk + V_2_age_risk) * pop_israel
        # Updating the accumulated used vaccines:
        curr_inventory_used += count_vaccines_in_use
        # Updating the vaccinated_que:
        vaccinated_que_.append(current_risk_age)

    final_trans_prop = curr_inv_ / curr_inventory_used
    return final_trans_prop, vaccinated_que_


# In[20]:


### Vaccinating in booster (moving from S_2 and V_2 to V_3, including the model update):
def vaccinate(trans_prop_: float, curr_res_mdl_1_ml: dict, vaccination_que: list, model_1_ml_):
    ## initialize settings:
    vaccination_que_copy = vaccination_que.copy()
    trans_prop_copy = trans_prop_
    # Getting the last day of the compartments and V_3, saving it to a dictionary:
    last_day_dict = {'S_2_last_day': np.reshape(curr_res_mdl_1_ml['S_2'][-1, :], newshape=(540) ),
                     'V_2_last_day': np.reshape(curr_res_mdl_1_ml['V_2'][-1, :], newshape=(540) ),
                     'V_3_last_day': np.reshape(curr_res_mdl_1_ml['V_3'][-1, :], newshape=(540) ),
                     'R_1_last_day': np.reshape(curr_res_mdl_1_ml['R_1'][-1, :], newshape=(540) ),
                     'R_2_last_day': np.reshape(curr_res_mdl_1_ml['R_2'][-1, :], newshape=(540) )}

    # Iterating over the combinations of (risk, age_group) that we need to vaccinate, and moving the population, and updating the model:
    for curr_risk_age in vaccination_que_copy:
        # Getting the relevant indexes to operate in the correspondent locations of the compartment:
        curr_risk_age_indexes = get_indexes_risk_age_combination(curr_risk_age)

        ## Vaccinating from S_2 to V_3
        S_2_last_day_updated, V_3_last_day_updated = vaccinating_from_V_2_S_2(last_day_dict['V_3_last_day'], last_day_dict['S_2_last_day'], trans_prop_copy, curr_risk_age_indexes)
        # updating the results:
        last_day_dict['V_3_last_day'], last_day_dict['S_2_last_day'] = V_3_last_day_updated, S_2_last_day_updated

        ## Vaccinating from V_2 to V_3
        V_2_last_day_updated, V_3_last_day_updated = vaccinating_from_V_2_S_2(last_day_dict['V_3_last_day'], last_day_dict['V_2_last_day'], trans_prop_copy, curr_risk_age_indexes)
        # updating the results:
        last_day_dict['V_3_last_day'], last_day_dict['V_2_last_day'] = V_3_last_day_updated, V_2_last_day_updated

    update_compartment(curr_res_mdl_1_ml, 'V_2', last_day_dict['V_2_last_day'], model_1_ml_)
    update_compartment(curr_res_mdl_1_ml, 'S_2', last_day_dict['S_2_last_day'], model_1_ml_)
    update_compartment(curr_res_mdl_1_ml, 'V_3', last_day_dict['V_3_last_day'], model_1_ml_)


# # Starting with the vaccination effort

# #### Vaccinating at the 1st month - running model until 2nd month

# In[21]:

#
# # Children's vaccinations of 5-9 started in 22/11/2021
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 1
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [], []
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[22]:
#
#
# ## Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_1.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_1.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_1.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_1.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # Reading results:
# with open('res_mdl_list_1.pickle', 'rb') as handle:
#  res_mdl_list = pickle.load(handle)
# with open('vaccination_days_t_1.pickle', 'rb') as handle:
#  vaccination_days_t =pickle.load(handle)
# with open('trans_prop_list_1.pickle', 'rb') as handle:
#  trans_prop_list = pickle.load(handle)
# with open('vaccinated_que_list_1.pickle', 'rb') as handle:
#  vaccinated_que_list = pickle.load(handle)


# In[23]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=1)
#
#
# # #### Vaccinating at the 2nd month - running model until 3rd month
# #
#
# # In[24]:
#
#
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 2
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [[],[],[]], [[],[],[]]
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[25]:
#
#
# # Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_2.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_2.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_2.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_2.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # # Reading results:
# # with open('res_mdl_list_2.pickle', 'rb') as handle:
# #     res_mdl_list = pickle.load(handle)
# # with open('vaccination_days_t_2.pickle', 'rb') as handle:
# #     vaccination_days_t =pickle.load(handle)
# # with open('trans_prop_list_2.pickle', 'rb') as handle:
# #     trans_prop_list = pickle.load(handle)
# # with open('vaccinated_que_list_2.pickle', 'rb') as handle:
# #     vaccinated_que_list = pickle.load(handle)
#
#
# # In[26]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=2)
#
#
# # #### Vaccinating at the 3rd month - running model until 4th month
#
# # In[27]:
#
#
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 3
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [[],[],[]], [[],[],[]]
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[28]:
#
#
# # Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_3.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_3.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_3.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_3.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # # Reading results:
# # with open('res_mdl_list_3.pickle', 'rb') as handle:
# #     res_mdl_list = pickle.load(handle)
# # with open('vaccination_days_t_3.pickle', 'rb') as handle:
# #     vaccination_days_t =pickle.load(handle)
# # with open('trans_prop_list_3.pickle', 'rb') as handle:
# #     trans_prop_list = pickle.load(handle)
# # with open('vaccinated_que_list_3.pickle', 'rb') as handle:
# #     vaccinated_que_list = pickle.load(handle)
#
#
# # In[29]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=3)
#
#
# # #### Vaccinating at the 4th month - running model until 5th month
#
# # In[30]:
#
#
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 4
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [[],[],[]], [[],[],[]]
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[31]:
#
#
# # Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_4.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_4.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_4.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_4.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # # Reading results:
# # with open('res_mdl_list_4.pickle', 'rb') as handle:
# #     res_mdl_list = pickle.load(handle)
# # with open('vaccination_days_t_4.pickle', 'rb') as handle:
# #     vaccination_days_t =pickle.load(handle)
# # with open('trans_prop_list_4.pickle', 'rb') as handle:
# #     trans_prop_list = pickle.load(handle)
# # with open('vaccinated_que_list_4.pickle', 'rb') as handle:
# #     vaccinated_que_list = pickle.load(handle)
#
#
# # In[32]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=4)
#
#
# # #### Vaccinating at the 5th month - running model until 6th month
#
# # In[33]:
#
#
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 5
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [[],[],[]], [[],[],[]]
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[34]:
#
#
# # Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_5.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_5.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_5.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_5.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # # Reading results:
# # with open('res_mdl_list_5.pickle', 'rb') as handle:
# #     res_mdl_list = pickle.load(handle)
# # with open('vaccination_days_t_5.pickle', 'rb') as handle:
# #     vaccination_days_t =pickle.load(handle)
# # with open('trans_prop_list_5.pickle', 'rb') as handle:
# #     trans_prop_list = pickle.load(handle)
# # with open('vaccinated_que_list_5.pickle', 'rb') as handle:
# #     vaccinated_que_list = pickle.load(handle)
#
#
# # In[35]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=5)
#
#
# # #### Vaccinating at the 6th month - running model until 7th month
#
# # In[36]:
#
#
# # fix the time period we vaccinate in: t is a number such that 1 <= t <= 6
# t = 6
# # Initializing lists to track the results, each element corresponds to an initial inventory level
# # Besides the res_mdl_list, each list will be in length of 20 - each value corresponds to each of the 6 months period
# res_mdl_list, trans_prop_list, vaccinated_que_list = [], [[],[],[]], [[],[],[]]
#
# # Iterating on the inventory level of time period t=1:
# for i in range(len(all_t_inv_dict[t])):
#     # Initializing the model's object (and its predictions) to t=0:
#     model_1_ml = defining_model('Scenario1')
#     # Running the model with that specific inventory level for 10 years, in 6 months resolution (a single run - 6 months X 20 == 10 years):
#     for j in range(20):
#         # Running until the vaccination month:
#         for k in range(t):
#             res_mdl_1_ml = model_1_ml.predict(
#                             days_in_season= 30,
#                             continuous_predict= True
#                             )
#         # Getting the current inventory level:
#         curr_inv = all_t_inv_dict[t][i]
#         # Getting the proportion of transition, and the combinations that we vaccinate:
#         trans_prop, vaccinated_que = calc_trans_prop(vaccination_pq, res_mdl_1_ml, curr_inv)
#         # Adding the results of a specific inventory level and "single-run" period to the result lists:
#         trans_prop_list.append(trans_prop)
#         vaccinated_que_list.append(vaccinated_que)
#
#         # Vaccinating (including the model update):
#         vaccinate(trans_prop, res_mdl_1_ml, vaccinated_que)
#
#         # Running the model until the end of the current "single" run (half a year)
#         for l in range(len(all_t_inv_dict.keys()) - t):
#             res_mdl_1_ml = model_1_ml.predict(
#                 days_in_season= 30,
#                 continuous_predict= True
#             )
#     # Adding the results of a specific inventory level over 10 years to the result list:
#     res_mdl_list.append(res_mdl_1_ml)
#
# ## Creating a list of days in which we vaccinate for the plot:
# # 529 == the start day of the simulation, (t * 30) == the time delta until we perform the vaccinations, 529+3650 == running the model for 10 years, 30*6 == we vaccinate exactly every half a year:
# vaccination_days_t = [t for t in range(529 + t * 30, 529 + 3650, 30*6)]
#
#
# # In[37]:
#
#
# # Saving the results of the current month, the days of the vaccinations, the transforming proportion and the vaccinated queue:
# with open('res_mdl_list_6.pickle', 'wb') as handle:
#     pickle.dump(res_mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccination_days_t_6.pickle', 'wb') as handle:
#     pickle.dump(vaccination_days_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('trans_prop_list_6.pickle', 'wb') as handle:
#     pickle.dump(trans_prop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('vaccinated_que_list_6.pickle', 'wb') as handle:
#     pickle.dump(vaccinated_que_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # # Reading results:
# # with open('res_mdl_list_6.pickle', 'rb') as handle:
# #     res_mdl_list = pickle.load(handle)
# # with open('vaccination_days_t_6.pickle', 'rb') as handle:
# #     vaccination_days_t =pickle.load(handle)
# # with open('trans_prop_list_6.pickle', 'rb') as handle:
# #     trans_prop_list = pickle.load(handle)
# # with open('vaccinated_que_list_6.pickle', 'rb') as handle:
# #     vaccinated_que_list = pickle.load(handle)
#
#
# # In[38]:
#
#
# # Display the compartments trend for six months, including the vaccination date
# plot_S_2_V_2_V_3_trend(res_mdl_list, vaccination_days_t, os.getcwd().split('/')[-1], t=6)

