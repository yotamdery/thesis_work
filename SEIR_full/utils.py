#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import poisson, binom
import pandas as pd
from SEIR_full.indices import *
import pickle, copy
import gc, os
import plotly.express as px
DATA_DIR = r'/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/Data'


def region_dict_age_aggregated():
    """This function produces a region mapper for the calibration data - when aggregating the data to 3 age groups"""
    regions_list = ['1100', '1101', '1103', '2100', '2200', '2202', '2300', '2302', '2400', '2402', '2900', '3100',
                    '3102', '3200', '3202', '4100', '4102', '4200', '4201', '4202', '4300', '4400', '5100', '5101',
                    '5103', '6100', '6200', '6202', '7100', '7101']
    region_dict = {}
    index_list = [i for i in range(0,90,3)]		# List of indices to start with, for each region
    for region in regions_list:
        i = index_list.pop(0)
        region_dict[region] = [i, i+1, i+2]
    return region_dict

def aggregate_to_3_age_groups(df):
    """This function gets a df, and aggregates it's columns to 3 age groups corresponding the 3 beta_js we have (<20, 20-59, 60+)"""
    df_copy = df.copy()
    # Getting the indices of each age group in the df:
    under_20 = sorted(list(range(0,270,9)) + list(range(1,270,9)) + list(range(2,270,9)))
    between_20_to_60 = sorted(list(range(3,270,9)) + list(range(4,270,9)) + list(range(5,270,9))+ list(range(6,270,9)))
    above_60 = sorted(list(range(7,270,9)) + list(range(8,270,9)))

    # Performing the aggregation:
    column_counter = 0  # Initialize column counter - we want to operate only when we arrived the last column within the range of current age group
    for i, tup in enumerate(df_copy.columns):
        if i in under_20:
            if column_counter < 2:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '0-19_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1] + df_copy.iloc[:, i-2]
                column_counter = 0

        if i in between_20_to_60:
            if column_counter < 3:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '20-59_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1] + df_copy.iloc[:, i-2] + df_copy.iloc[:, i-3]
                column_counter = 0

        if i in above_60:
            if column_counter < 1:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '60+_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1]
                column_counter = 0

    # Returning the relevant columns
    return df_copy.iloc[: , 270:]

def aggregate_to_4_age_groups(df):
    """This function gets a df, and aggregates it's columns to 3 age groups corresponding the 3 beta_js we have (<20, 20-59, 60+)"""
    df_copy = df.copy()
    # Getting the indices of each age group in the df:
    under_20 = sorted(list(range(0,270,9)) + list(range(1,270,9)) + list(range(2,270,9)))
    between_20_to_40 = sorted(list(range(3,270,9)) + list(range(4,270,9)))
    between_40_to_60 = sorted(list(range(5, 270, 9)) + list(range(6, 270, 9)))
    above_60 = sorted(list(range(7,270,9)) + list(range(8,270,9)))

    # Performing the aggregation:
    column_counter = 0  # Initialize column counter - we want to operate only when we arrived the last column within the range of current age group
    for i, tup in enumerate(df_copy.columns):
        if i in under_20:
            if column_counter < 2:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '0-19_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1] + df_copy.iloc[:, i-2]
                column_counter = 0
                continue

        if i in between_20_to_40:
            if column_counter < 1:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '20-39_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1]
                column_counter = 0
                continue

        if i in between_40_to_60:
            if column_counter < 1:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '40-59_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1]
                column_counter = 0
                continue

        if i in above_60:
            if column_counter < 1:
                column_counter+=1
                continue
            else:   # If it's the last column within the range of the age groups
                df_copy[(str(tup[0]), '60+_absolute_cases')] = df_copy.iloc[:, i] + df_copy.iloc[:, i-1]
                column_counter = 0
                continue

    # Returning the relevant columns
    return df_copy.iloc[: , 270:]

def vaccinating_from_V_2_S_2(V_3_last_day_: np.array, comp_last_day: np.array, trans_prop_copy_: float, curr_indexes):
    """ This function gets the last day of V_3, the last day of either V_2 or S_2, and the indexes to operate on,
    and transfers the right amount of population from either V_2 or S_2 to V_3"""
    # getting the absolute amounts that we need to transfer to V_3
    amounts_to_transfer_from_comp = comp_last_day[curr_indexes] * trans_prop_copy_
    # Updating the compartment to the new values after transferring:
    comp_last_day[curr_indexes] = comp_last_day[curr_indexes] * (1-trans_prop_copy_)
    comp_last_day_updated = comp_last_day

    ## TRANSFERRING:
    # getting the current amounts in the right places in V_3:
    amounts_to_receive_transfer_V_3 = V_3_last_day_[curr_indexes]
    # Calculating the transfer ratio for each cell (element-wise):
    proportions_to_transfer = amounts_to_transfer_from_comp / amounts_to_receive_transfer_V_3
    # Performing the actual transfer calculation for each cell (element-wise):
    V_3_last_day_[curr_indexes] = V_3_last_day_[curr_indexes] * (1+proportions_to_transfer)
    V_3_last_day_updated = V_3_last_day_

    return comp_last_day_updated, V_3_last_day_updated

def list_to_dict(lst: list, list_of_keys: list):
    """This function gets a list of lists and list of strings to be keys, and turns it to list of dictionaries"""
    dict = {}
    for i, element in enumerate(lst):
        dict[list_of_keys[i]] = element
    return dict

def get_opposite_dict(dic, keys):
    """The function gets a dict and new keys list and returns a dictionary in which keys as keys,
    and values are the keys from dic """
    res_dict = {}

    for new_key in keys:
        new_val_lst = []
        for old_k, old_v in dic.items():
            if all(i in old_v for i in new_key):
                new_val_lst.append(old_k)
        if len(new_key) == 1:
            res_dict[new_key[0]] = new_val_lst
        else:
            res_dict[new_key] = new_val_lst
    return res_dict


def expand_partial_array(mapping_dic, array_to_expand, size):
    """The function gets mapping_dic - indices to assign in the expanded array (with key granularity based on
    array_to_expand), and array_to_expand - the expanded array's values will be based on this array. Returns
    and expanded array shape (len(N),1) based on the array_to_expand"""

    # Assigning values to the full array
    full_array = np.zeros(size)
    for i, k in enumerate(mapping_dic.keys()):
        full_array[mapping_dic[k]] = array_to_expand[i]

    return full_array


def divide_population(
        ind,
        prop_dict,
        vector_to_switch,
    ):
    """
    The function move population from non-Intervention group to Intervention group
    :param prop_dict: key = (region,risk,age), value=prop to shift
    :param vector_to_switch: the current population distribution
    :param low_only: shift low-risk groups only (boolean)
    :return: vector with new population distribution.
    """
    new_distribution = vector_to_switch.copy()
    for region, risk, age in prop_dict.keys():
        # indices of intervention group and non-intervention group
        inter_idx = ind.inter_region_risk_age_dict['Intervention',region,risk,age]
        non_inter_idx = ind.inter_region_risk_age_dict['Non-intervention', region, risk, age]

        # Performing the population transition
        new_distribution[inter_idx] = vector_to_switch[non_inter_idx] * prop_dict[region,risk,age]
        new_distribution[non_inter_idx] = vector_to_switch[non_inter_idx] * (1 - prop_dict[region, risk, age])


    return new_distribution


def ML_Bin(
        data,
        model_pred,
        threshold = 5,
        approx=True,
        factor=1,
    ):
    """
    This function calculates the log-likelihood of the Bin approximation of the
    measurment of illnes in israel.
    It assumes the number of tests is n_{j,k,t}, the probability for getting a
    result is p_{j,k,t} - the model prediction, and the data point is q_{j,k,t}.
    in total the likelihood P(X=q)~Bin(n,p) per data point.
    For cells (specific t,j,k triplet) of not sufficient number of tests:
    with n_{j,k,t} < threshold the likelihood will be ignored.
    :param data: np.array of 4 dimensions :
                    axis 0: n, q - representing different values:
                            starting from  total tests, and
                            than positives rate.
                    axis 1: t - time of sample staring from the first day in
                            quastion calibrated to the model.
                    axis 2: k - area index
                    axis 3: j - age index
            data - should be smoothed.
            (filled with zeros where no test accured)
    :param model_pred: np.ndarray of 3 dimensions representing the probability:
                        axis 1: t - time of sample staring from the first day
                                in quastion calibrated to the model.
                        axis 2: k - area index
                        axis 3: j - age index
    :return: the -log-likelihood of the data given the prediction of the model.
    """
    n = data[0, :, :]
    q = factor*data[1, :, :]
    p = model_pred
    if approx:
        ##				###
        # poison approx. ##
        ##				###
        ll = -poisson.logpmf(
            k=n * q,
            mu=n * p,
        )
    else:
        ##				###
        # Binomial dist. ##
        ##				###
        ll = -binom.logpmf(
            k=n * q,
            n=n,
            p=p,
        )

    # cut below threshold values
    ll = np.nan_to_num(ll,
                  nan=0,
                  posinf=0,
                  neginf=0)
    ll = ll * (n > threshold)
    return ll.sum()


def ML_Pois_Naiv(
        data,
        model_pred,
    ):
    """
    This function calculates the log-likelihood of the POISSON of the
    measurment of illnes in israel.
    It assumes the number of tests is n_{j,k,t}, the probability for getting a
    result is p_{j,k,t} - the model prediction, and the data point is q_{j,k,t}.
    in total the likelihood P(X=q)~Bin(n,p) per data point.
    For cells (specific t,j,k triplet) of not sufficient number of tests:
    with n_{j,k,t} < threshold the likelihood will be ignored.
    :param data: np.array of 4 dimensions :
                    axis 0: n, q - representing different values:
                            starting from  total tests, and
                            than positives rate.
                    axis 1: t - time of sample staring from the first day in
                            quastion calibrated to the model.
                    axis 2: k - area index
                    axis 3: j - age index
            data - should be smoothed.
            (filled with zeros where no test accured)
    :param model_pred: np.ndarray of 3 dimensions representing the probability:
                        axis 1: t - time of sample staring from the first day
                                in quastion calibrated to the model.
                        axis 2: k - area index
                        axis 3: j - age index
    :return: the -log-likelihood of the data given the prediction of the model.
    """
    factor = 1
    q = data
    p = factor*model_pred

    ##				###
    # poison approx. ##
    ##				###
    ll = -poisson.logpmf(
        k=q,
        mu=p,
    )

    # cut below threshold values
    ll = np.nan_to_num(ll,
                  nan=0,
                  posinf=0,
                  neginf=0)
    return ll.sum()


def MSE(
        data,
        model_data_processed
    ):
    """
    :param data:
    :param model_data_processed:
    :return:
    """
    # data = np.array(data, dtype= np.float64)
    # model_data_processed = np.array(model_data_processed, dtype= np.float64)
    mse = (((data - model_data_processed) ** 2).mean(axis=0)).sum()
    if (mse is None) or (mse == np.nan):
        return 1e+4
    return mse

def shrink_array_sum(
        mapping_dic,
        array_to_shrink
    ):
    """
    The function gets mapping_dic - indices to assign in the shrunk array
    (with key granularity based on array_to_expand), and array_to_shrink -
    the shrunk array's values will be based on this array. Returns and shrunk
    array shape (len(mapping_dic),1) based on the array_to_shrink.
    """

    # Assigning values to the shrinked array
    shrunk_array = np.zeros(len(mapping_dic))
    for i, k in enumerate(mapping_dic.keys()):
        shrunk_array[i] = array_to_shrink[mapping_dic[k]].sum()

    return shrunk_array


def create_C_mtx_leisure_work(
        ind,
        od_mat,
        base_mat,
        age_dist_area,
        eye_mat=None,
        stay_home_idx=None
    ):
    """
    This function creates Contact matrix for a specific scenario with or w/o
    stay_home_idx
    :param od_mat:
    :param base_mat:
    :param age_dist_area:
    :param eye_mat:
    :param stay_home_idx:
    :return:
    """
    od_mat = od_mat.apply(lambda row: row / row[list(ind.G.values())].sum()).copy()
    full_C = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(list(ind.MI.values()),
                                        names=['age', 'area', 'age']),
        columns=od_mat.index
    )
    # fill the matrix:
    if stay_home_idx is None:
        factor = 1
    for index in list(full_C.index):
        if factor != 1:
            try:
                factor = stay_home_idx.loc[index[1]]['mean']
            except:
                factor = 1
        tmp1 = age_dist_area[index[2]]
        tmp1.loc[tmp1 == 0] = 1
        tmp1.index = list(ind.G.values())
        if (index[0] in ['0-4', '5-9']) and (eye_mat is not None):
            tmp2 = eye_mat.loc[index[1]] * \
                   base_mat.loc[index[0]][index[2]] * factor
        else:
            tmp2 = od_mat.loc[index[1]] * \
                   base_mat.loc[index[0]][index[2]] * factor
        full_C.loc[index] = tmp2.divide(tmp1)

    return csr_matrix(
        full_C.unstack().reorder_levels(['area', 'age']).sort_index().values.astype(float)
    )


def Beta2beta_j(Beta):
    return np.array([Beta[0], Beta[0], Beta[0],
                    Beta[1], Beta[1], Beta[1], Beta[1],
                    Beta[2], Beta[2]])


def isolate_areas(
        ind,
        base_mtx,
        isolation_mtx,
        area_lst,
    ):
    """
    The function gets base_mtx and switches its rows and columns representing the areas in area_lst with thw
    matching rows and columns from isolation_mtx.
    :param base_mtx: numpy array matrix
    :param isolation_matx: numpy array matrix with the same shape as base_mtx
    :param area_lst: areas to put in/out quarantine
    :return: base_mtx with the matching rows and columns from isolation_mtx
    """
    base_mtx = copy.deepcopy(base_mtx)

    for area in area_lst:
        # switch rows:
        base_mtx[ind.region_ga_dict[area], :] = isolation_mtx[ind.region_ga_dict[area], :]
        # switch columns:
        base_mtx[:, ind.region_ga_dict[area]] = isolation_mtx[:, ind.region_ga_dict[area]]
    return base_mtx


def isolate_areas_vect(ind,base_vect_in, isolation_vect, area_lst):
    """
    The function gets base_vect_in and replaces values representing the areas in area_lst from isolation_mtx.
    :param base_vect_in: numpy array matrix
    :param isolation_vect: numpy array matrix with the same shape as base_mtx
    :param area_lst: areas to put in/out quarantine
    :return: base_vect with the matching rows and columns from isolation_mtx
    """

    # sometimes the configuration files hold a scalar value instead of vector for initial indices.
    if (type(base_vect_in) == float) | (type(base_vect_in) == int):
        return base_vect_in

    base_vect = copy.deepcopy(base_vect_in)


    for area in area_lst:
        base_vect[ind.region_gra_dict[area]] = isolation_vect[ind.region_gra_dict[area]]
    return base_vect


def multi_inter_by_name(
        ind,
        model,
        is_pop,
        inter_names,
        inter_times=None,
        sim_length=300,
        fix_vents=True,
        deg_param=None,
        no_pop=False,
        min_deg_run=20,
        start=0,
    ):
    time_in_season = start
    if len(inter_names) == 1:
        inter_times = []
        inter_times.append(sim_length)
        sim_left = sim_length
    else:
        inter_times.append(sim_length - np.sum(inter_times))

    model_inter = copy.deepcopy(model)
    if deg_param is not None:
        with open('../Data/interventions/C_inter_' + deg_param['inter_max'] + '.pickle',
                  'rb') as pickle_in:
            C_max = pickle.load(pickle_in)

    for i, inter_name in enumerate(inter_names):
        # First intervention
        with open('../Data/interventions/C_inter_' + inter_name + '.pickle',
                  'rb') as pickle_in:
            C_inter = pickle.load(pickle_in)

        with open(
                '../Data/interventions/stay_home_idx_inter_' + inter_name + '.pickle',
                'rb') as pickle_in:
            stay_home_idx_inter = pickle.load(pickle_in)

        with open(
                '../Data/interventions/routine_t_inter_' + inter_name + '.pickle',
                'rb') as pickle_in:
            routine_t_inter = pickle.load(pickle_in)

        if i != 0 or no_pop:
            transfer_pop_inter = None
        else:
            with open(
                    '../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle',
                    'rb') as pickle_in:
                transfer_pop_inter = pickle.load(pickle_in)

        sim_left = inter_times[i]

        if deg_param is not None:
            if i==0:
                curr_time = 0
            else:
                curr_time = np.sum(inter_times[:i])
            # curr_t * deg_rate,
            # max_deg_rate
            if (sim_left < min_deg_run) or \
                (curr_time * deg_param['deg_rate'] > deg_param['max_deg_rate']) :
                C_inter_deg, stay_home_idx_inter_deg = policy_degredation(
                    ind,
                    C_inter,
                    C_max,
                    stay_home_idx_inter,
                    deg_param['deg_rate'],
                    range(curr_time, curr_time + sim_left, 1),
                    deg_param['max_deg_rate'],
                    True,
                )
                res_mdl = model_inter.intervention(
                    C=C_inter_deg,
                    days_in_season=sim_left,
                    stay_home_idx=stay_home_idx_inter_deg,
                    not_routine=routine_t_inter,
                    prop_dict=transfer_pop_inter,
                    start=time_in_season,
                )
                time_in_season += sim_left
            else:
                time_left = sim_left
                while time_left > 0:
                    C_inter_deg, stay_home_idx_inter_deg = policy_degredation(
                        ind,
                        C_inter,
                        C_max,
                        stay_home_idx_inter,
                        deg_param['deg_rate'],
                        range(curr_time,
                              curr_time + min(min_deg_run, time_left),
                              1),
                        deg_param['max_deg_rate'],
                        True,
                    )
                    res_mdl = model_inter.intervention(
                        C=C_inter_deg,
                        days_in_season=min(min_deg_run, time_left),
                        stay_home_idx=stay_home_idx_inter_deg,
                        not_routine=routine_t_inter,
                        prop_dict=transfer_pop_inter,
                        start=time_in_season,
                    )
                    time_in_season += sim_left
                    time_left -= min_deg_run
                    curr_time +=min_deg_run
                    gc.collect()
        else:
            res_mdl = model_inter.intervention(
                C=C_inter,
                days_in_season=sim_left,
                stay_home_idx=stay_home_idx_inter,
                not_routine=routine_t_inter,
                prop_dict=transfer_pop_inter,
                start=time_in_season,
            )
            time_in_season += sim_left
        del C_inter
        del stay_home_idx_inter
        del routine_t_inter
        del transfer_pop_inter
        gc.collect()

    if fix_vents:
        for i, vent in enumerate(res_mdl['Vents']):
            res_mdl['Vents'][i] = vent + (
                    (60.0 / is_pop) * vent) / vent.sum()

    return (res_mdl, model_inter)


def automatic_global_stop_inter(
        model,
        is_pop,
        inter_name,
        closed_inter,
        thresh=4960,
        thresh_var='Vents',
        sim_length=300,
        start=0,
    ):
    time_in_season=start
    # First intervention
    with open('../Data/interventions/C_inter_' + inter_name + '.pickle',
              'rb') as pickle_in:
        C_inter = pickle.load(pickle_in)

    with open(
            '../Data/interventions/stay_home_idx_inter_' + inter_name + '.pickle',
            'rb') as pickle_in:
        stay_home_idx_inter = pickle.load(pickle_in)

    with open(
            '../Data/interventions/routine_t_inter_' + inter_name + '.pickle',
            'rb') as pickle_in:
        routine_t_inter = pickle.load(pickle_in)

    with open(
            '../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle',
            'rb') as pickle_in:
        transfer_pop_inter = pickle.load(pickle_in)

    #Seconed intervention
    with open('../Data/interventions/C_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
        C_close = pickle.load(pickle_in)

    with open('../Data/interventions/stay_home_idx_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
        stay_home_idx_close = pickle.load(pickle_in)

    with open('../Data/interventions/routine_t_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
        routine_t_close = pickle.load(pickle_in)

    with open('../Data/interventions/transfer_pop_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
        transfer_pop_close = pickle.load(pickle_in)

    # run normal intervention of letting everybody out
    model_inter = copy.deepcopy(model)
    res_mdl = model_inter.intervention(
        C=C_inter,
        days_in_season=sim_length,
        #             days_in_season=(dates[-1]-start_inter).days,
        #             days_in_season=inter2_timing - (start_inter-beginning).days,
        stay_home_idx=stay_home_idx_inter,
        not_routine=routine_t_inter,
        prop_dict=transfer_pop_inter,
        start=time_in_season,
    )

    days_to_closing = (res_mdl[thresh_var].sum(
        axis=1) * is_pop < thresh).sum() - 14
    days_to_closing -= len(model.S)

    # run normal intervention of letting everybody out until sec inter.
    model_inter = copy.deepcopy(model)
    model_inter.intervention(
        C=C_inter,
        days_in_season=days_to_closing,
        #             days_in_season=(dates[-1]-start_inter).days,
        #             days_in_season=inter2_timing - (start_inter-beginning).days,
        stay_home_idx=stay_home_idx_inter,
        not_routine=routine_t_inter,
        prop_dict=transfer_pop_inter,
        start=time_in_season,
    )
    time_in_season += days_to_closing
    res_mdl_close = model_inter.intervention(
        C=C_close,
        days_in_season=sim_length-days_to_closing,
        stay_home_idx=stay_home_idx_close,
        not_routine=routine_t_close,
        prop_dict=None,
        start=time_in_season,
    )

    for i, vent in enumerate(res_mdl['Vents']):
        res_mdl['Vents'][i] = vent + (
                    (60.0 / is_pop) * vent) / vent.sum()
    for i, vent in enumerate(res_mdl_close['Vents']):
        res_mdl_close['Vents'][i] = vent + (
                    (60.0 / is_pop) * vent) / vent.sum()

    return (res_mdl, res_mdl_close, days_to_closing)


def print_stat_fit_behave(fit_results_object):

    """The function gets sci optimization results object and print additional info about the optimization
    :param fit_results_object:
    :return:
    """
    print('minimized value:', fit_results_object.fun)
    print('Fitted parameters:\n Beta={0}\n beta_lockdown= {1},\n, beta_isolation= {2}\n,'
                                'beta_school= {3}'.format(fit_results_object.x[:3],
                                                         fit_results_object.x[3],
                                                         fit_results_object.x[4],
                                                         fit_results_object.x[5]))
    print('num of sampling the target function:', fit_results_object.nfev)


def print_stat_fit_hosp(fit_results_object, tracking='hosp'):

    """The function gets sci optimization results object and print additional info about the optimization
    :param fit_results_object:
    :return:
    """
    print('minimized value:', fit_results_object.fun)
    if tracking == 'hosp':
        print('Fitted parameters:\n Eta={0}\n Nu={1},\n '.format(fit_results_object.x[0],
                                                                                  fit_results_object.x[1]))

    elif tracking == 'vents':
        print('Fitted parameters:\n Xi={0}\n Mu={1},\n '.format(fit_results_object.x[0],
                                                                                  fit_results_object.x[1]))
    print('num of sampling the target function:', fit_results_object.nfev)


def policy_degredation(
    ind,
    C,
    C_max,
    sh_idx,
    deg_rate,
    t,
    max_deg_rate,
    risk_deg,
    ):

    new_c = {}
    new_sh = {}

    t_factor = int(float(max_deg_rate)/deg_rate)
    if risk_deg:
        for key in C.keys():
            new_c[key] = []
            C_mat_step = (C_max[key][0].todense() - C[key][0].todense())
            for curr_t in t:
                if curr_t < t_factor:
                    C_mat = copy.deepcopy(C[key][0].todense())
                    factor = np.minimum(
                        curr_t * deg_rate,
                        max_deg_rate) / 100.0
                    if key in ['home_non', 'work_non', 'leisure_non']:
                        C_mat += C_mat_step * factor
                    else:
                        idx = list(ind.age_ga_dict['60-69']) + \
                            list(ind.age_ga_dict['70+'])
                        C_mat[idx, :] += C_mat_step[idx, :] * factor
                        C_mat[:, idx] += C_mat_step[:, idx] * factor
                    C_mat = csr_matrix(C_mat)
                    new_c[key].append(C_mat)
                    del C_mat
                    gc.collect()
                else:
                    break
            if t_factor <= t[-1]:
                C_max_key = C_max[key][0].todense()
                if key in ['home_non', 'work_non', 'leisure_non']:
                    C_mat = C_max_key
                else:
                    C_mat = copy.deepcopy(C[key][0].todense())
                    idx = list(ind.age_ga_dict['60-69']) + \
                          list(ind.age_ga_dict['70+'])
                    C_mat[idx, :] = C_max_key[idx, :]
                    C_mat[:, idx] = C_max_key[:, idx]
                C_mat = csr_matrix(C_mat)
                for curr_t in range(t_factor, t[-1]+1, 1):
                    new_c[key].append(C_mat)
            del C_mat_step
            gc.collect()
        for key in sh_idx.keys():
            new_sh[key] = {}
            for key2 in sh_idx[key].keys():
                new_sh[key][key2] = []
                sh_idx_step = (np.ones_like(sh_idx[key][key2][0]) -
                               sh_idx[key][key2][0])
                for curr_t in t:
                    if curr_t < t_factor:
                        sh_idx_vec = sh_idx[key][key2][0].copy()
                        factor = np.minimum(
                            curr_t * deg_rate,
                            max_deg_rate) / 100.0
                        if key == 'non_inter':
                            sh_idx_vec += sh_idx_step * factor
                        else:
                            idx = list(ind.age_ga_dict['60-69']) + \
                                  list(ind.age_ga_dict['70+'])
                            sh_idx_vec[idx] += sh_idx_step[idx] * factor
                        new_sh[key][key2].append(sh_idx_vec)
                        del sh_idx_vec
                        gc.collect()
                    else:
                        break
                if t_factor <= t[-1]:
                    if key == 'non_inter':
                        sh_idx_vec = np.ones_like(sh_idx[key][key2][0])
                    else:
                        sh_idx_vec = sh_idx[key][key2][0].copy()
                        idx = list(ind.age_ga_dict['60-69']) + \
                              list(ind.age_ga_dict['70+'])
                        sh_idx_vec[idx] = 1
                    for curr_t in range(t_factor, t[-1]+1, 1):
                        new_sh[key][key2].append(sh_idx_vec)
                del sh_idx_step
                gc.collect()
    return (new_c, new_sh)


def inter2name(
        ind,
        pct,
        no_risk=False,
        no_kid=False,
        kid_019=True,
        kid_09=False,
        kid_04=False,
    ):
    inter_name = ind.cell_name + '@' + str(pct)
    if pct != 10:
        if no_risk:
            inter_name += '_no_risk65'
        if no_kid:
            inter_name += '_no_kid'
        else:
            if kid_019:
                inter_name += '_kid019'
            elif kid_09:
                inter_name += '_kid09'
            elif kid_04:
                inter_name += '_kid04'
    return inter_name

def save_cal(scen, res_fit):
    """Saving the results of the calibration, for each scenario"""
    cal_parameters = pd.read_pickle(DATA_DIR + '/calibration/calibration_dict.pickle')
    cal_parameters[scen] = {
        'beta_j': Beta2beta_j([res_fit.x[0], res_fit.x[1], res_fit.x[2]]),
        'beta_activity': res_fit.x[3],
        'beta_school': res_fit.x[4],
    }
    with open(DATA_DIR + '/calibration/calibration_dict.pickle', 'wb') as handle:
        pickle.dump(cal_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_cal_track(res_fit, ind, scen, phase, track, no_mobility, no_haredim):
    cal_parameters = pd.read_pickle('../Data/calibration/calibration_dict.pickle')
    if no_mobility:
        cal_tpl = (int(scen[-1]), phase, 'no_mobility')
    elif no_haredim:
        cal_tpl = (int(scen[-1]), phase, 'no_haredim')
    else:
        cal_tpl = (int(scen[-1]), phase)

    if track == 'H':
        cal_parameters[ind.cell_name][cal_tpl]['eta'] = \
            res_fit.x[0]
        cal_parameters[ind.cell_name][cal_tpl]['nu'] = \
            res_fit.x[1]

    elif track == 'Vents':

        cal_parameters[ind.cell_name][cal_tpl]['xi'] = \
            res_fit.x[0]
        cal_parameters[ind.cell_name][cal_tpl]['mu'] = \
            res_fit.x[1]
    with open('../Data/calibration/calibration_dict.pickle', 'wb') as handle:
        pickle.dump(cal_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)


def num2scen(num):
    return 'Scenario'+str(num)

def reading_aggregated_new_cases_file():
    """
    This function returns a smoothed data of absolute new cases ( 7 days smoothing)
    """
    cols_for_main_morbidity = ['30_county', 'date', 'accumulated_cases', 'population']
    main_morbidity = pd.read_excel(
        r'/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/Data/sick/sick_data_new_cases_only_30.xlsx',
        usecols=cols_for_main_morbidity, engine= 'openpyxl')
    main_morbidity['date'] = pd.to_datetime(main_morbidity['date'], dayfirst=True)

    # Aggtegating
    main_morbidity = main_morbidity.groupby(by='date', as_index=True).sum().drop('population', axis=1)
    # Getting the absolut numbers
    main_morbidity['absolute_cases'] = main_morbidity['accumulated_cases'].diff(periods=1)
    # Taking as pd Series
    absolute_main_morbidity = main_morbidity['absolute_cases']
    # Smoothing
    smoothed_absolute_morbidity = absolute_main_morbidity.rolling(window=7).mean().dropna()
    # Taking the relevant dates
    smoothed_absolute_morbidity = smoothed_absolute_morbidity[smoothed_absolute_morbidity.index >= '2020-05-15']
    return smoothed_absolute_morbidity

def moving_events_dates_backward(events_dict, days_to_move):
    """
    This function gets the static events dict created in parameters.py, and move all of its dates X days backwards
    to match the smoothing actions that are made.
    """
    events_dict_copy = events_dict.copy()
    updated_events_dict = {}		# Initializing the updated dict
    for event in events_dict_copy.keys():	# Iterating over the keys (each key is an event)
        updated_dates_pair = []				# List to hold the event's updated dates
        for element in events_dict_copy[event]:		# Each tuple consist of 2 elements (2 dates)
            updated_dates_pair.append(pd.to_datetime(element) -
                                      pd.Timedelta(days= days_to_move))		# Performing the shifting in dates
        updated_events_dict[event] = tuple(updated_dates_pair)
    return updated_events_dict


