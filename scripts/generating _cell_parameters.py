## Imports
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
import scipy
from scipy import optimize
from scipy.sparse import csr_matrix
import sys, os
from SEIR_full.utils import *
from SEIR_full.indices import *

#############################################
# Generating parameters files based on tazs #
#############################################

cell_name = '30'
israel_pop = 9345000
start_model = pd.to_datetime('2020-05-15')
end_model = pd.to_datetime('2021-10-25')
simulation_end_model = pd.to_datetime('2022-10-25')

# add functions
def make_pop(df):
    df = df.iloc[:, 0:-2]
    return df.sum(axis=0)


def make_pop_religion(df):
    df = df.iloc[:, 1:8].multiply(df['tot_pop'], axis='index')
    return df.sum(axis=0)


def robust_max(srs, n=3):
    sort = sorted(srs)
    return np.mean(sort[-n:])


def robust_min(srs, n=3):
    sort = sorted(srs)
    return np.mean(sort[:n])


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)


def avg_by_dates(df, from_date, to_date, weights=None):
    filtered = df[(df.index >= from_date) & (df.index <= to_date)]
    if weights is None:
        return filtered.describe().T[['mean', 'std', 'min', 'max']]

    weights = pd.Series(weights)
    stats = filtered.describe().T[['min', 'max']]
    stats['mean'] = filtered.apply(
        lambda col: np.average(col, weights=weights))
    stats['std'] = filtered.apply(
        lambda col: weighted_std(col, weights=weights))
    return stats


def wheighted_average(df):
    tot = df['tot_pop'].sum()
    return (df['cases_prop'].sum() / tot)


def normelize(row, global_min, span):
    new_row = (row - global_min) / span
    new_row = np.minimum(new_row, 1.0)
    new_row = np.maximum(new_row, 1e-6)
    return new_row


def create_demograph_age_dist_empty_cells(ind):
    ### Creating demograph/age_dist
    pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2, engine='openpyxl')
    ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]
    pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
    pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
    pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
    pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
    pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
        pop_dist['id']).join(pop_dist['tot_pop'])
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
    pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
    pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'], axis=0)

    taz2cell = pd.read_excel(
        '../Data/division_choice/' + ind.cell_name + '/taz2county_id.xlsx', engine='openpyxl')
    taz2cell = taz2cell[['taz_id', 'cell_id']]
    taz2cell.columns = ['id', 'new_id']

    pop_cell = pop_dist.merge(taz2cell, left_on='id', right_on='id')
    pop_cell['new_id'] = pop_cell['new_id'].astype(str)
    pop_cell.sort_values(by='new_id')

    pop_cell = pop_cell.groupby(by='new_id').apply(lambda df: make_pop(df))
    pop_cell['10-19'] = pop_cell['10-14'] + pop_cell['15-19']
    pop_cell['20-29'] = pop_cell['20-24'] + pop_cell['25-29']
    pop_cell['30-39'] = pop_cell['30-34'] + pop_cell['35-39']
    pop_cell['40-49'] = pop_cell['40-44'] + pop_cell['45-49']
    pop_cell['50-59'] = pop_cell['50-54'] + pop_cell['55-59']
    pop_cell['60-69'] = pop_cell['60-64'] + pop_cell['65-69']
    pop_cell['70+'] = pop_cell['70-74'] + pop_cell['75+']
    pop_cell = pop_cell[list(ind.A.values())]
    pop_cell = pop_cell / pop_cell.sum().sum()
    pop_cell.reset_index(inplace=True)
    pop_cell.columns = ['cell_id'] + list(ind.A.values())

    ## empty cells file to save
    try:
        os.mkdir('../Data/demograph')
    except:
        pass
    empty_cells = pop_cell[pop_cell.sum(axis=1) == 0]['cell_id']
    empty_cells.to_csv('../Data/demograph/empty_cells.csv')

    empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
        'cell_id'].astype(str)
    pop_cell = pop_cell[
        pop_cell['cell_id'].apply(lambda x: x not in empty_cells.values)]
    pop_cell.to_csv('../Data/demograph/age_dist_area.csv')


def create_paramaters_ind(ind):
    ind.update_empty()
    ## empty cells file to save
    try:
        os.mkdir('../Data/parameters')
    except:
        pass
    with open('../Data/parameters/indices.pickle', 'wb') as handle:
        pickle.dump(ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ind


def create_demograph_religion(ind):
    ### creating demograph/religion
    religion2taz = pd.read_csv('../Data/raw/religion2taz.csv')
    religion2taz.sort_values(by='taz_id', inplace=True)
    religion2taz.columns = ['id', 'Orthodox', 'Druze', 'Other', 'Sacular',
                            'Muslim', 'Christian']
    religion2taz['Jewish'] = religion2taz['Orthodox'] + religion2taz['Sacular']
    taz2cell = pd.read_excel(
        '../Data/division_choice/' + ind.cell_name + '/taz2county_id.xlsx', engine='openpyxl')
    taz2cell = taz2cell[['taz_id', 'cell_id']]
    taz2cell.columns = ['id', 'new_id']
    religion2taz = religion2taz.merge(taz2cell, on='id')
    religion2taz['new_id'] = religion2taz['new_id'].astype(str)
    religion2taz.sort_values(by='new_id', inplace=True)
    pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2, engine='openpyxl')
    ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]

    pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
    pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
    pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
    pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
    pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
        pop_dist['id']).join(pop_dist['tot_pop'])
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
    pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
    pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'], axis=0)
    pop_dist = pop_dist[['id', 'tot_pop']]

    religion2taz = religion2taz.merge(pop_dist, on='id')
    religion2taz.sort_values(by='id', inplace=True)

    # fixing religion city factor
    if ind.cell_name == '20':
        cell_num = len(list(set(religion2taz['new_id'])))
        factor = pd.DataFrame({'new_id': list(set(religion2taz['new_id'])),
                               'orth_factor': [1] * cell_num,
                               'arab_factor': [1] * cell_num, }).sort_values(by='new_id')
        factor = factor.reset_index().drop(['index'], axis=1)
        factor.iloc[:, 1] = pd.Series(
            [1,
             0.48 / 0.41,
             0.13 / 0.04,
             0.05 / 0.02,
             1,
             1,
             1,
             1,
             1,
             1,
             0.1 / 0.05,
             1,
             1,
             1,
             0.82 / 0.6,
             1,
             1,
             1,
             1,
             0.24 / 0.36])
        factor.iloc[:, 2] = pd.Series(
            [0.38 / 0.01,
             1,
             0.106 / 0.01,
             0.36 / 0.14,
             0.65 / 0.4,
             1.1 / 0.38,
             1.1 / 0.1,
             0.15 / 0.07,
             0.6 / 0.3,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             1,
             1.3 / 0.8,
             1])

        religion2taz = religion2taz.merge(factor, on='new_id')
        religion2taz['Orthodox'] = religion2taz['Orthodox'] * religion2taz[
            'orth_factor']
        religion2taz['Sacular'] = religion2taz['Sacular'] - religion2taz[
            'Orthodox'] * (religion2taz['orth_factor'] - 1)
        religion2taz['Muslim'] = religion2taz['Muslim'] * religion2taz[
            'arab_factor']

    religion2taz = religion2taz.groupby(by='new_id').apply(make_pop_religion)
    tmp = religion2taz[
        ['Druze', 'Other', 'Muslim', 'Christian', 'Jewish']].sum(axis=1)
    tmp.loc[tmp == 0] = 1
    religion2taz = religion2taz.divide(tmp, axis=0)
    religion2taz.reset_index(inplace=True)
    religion2taz.columns = ['cell_id', 'Orthodox', 'Druze', 'Other', 'Sacular',
                            'Muslim', 'Christian', 'Jewish']
    religion2taz['cell_id'] = religion2taz['cell_id'].astype(str)
    empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
        'cell_id'].astype(str)
    religion2taz = religion2taz[
        religion2taz['cell_id'].apply(lambda x: x not in empty_cells.values)]
    religion2taz.to_csv('../Data/demograph/religion_dis.csv')


def create_stay_home(ind):
    ## Creating stay_home/ALL
    home = pd.read_excel('../Data/raw/Summary_Home_0_TAZ.xlsx', engine='openpyxl')
    home = home.iloc[:, 1:]
    home.columns = ['date', 'taz_id', 'stay', 'out']
    home['date'] = pd.to_datetime(home['date'], dayfirst=True)
    #home['stay'] = home['stay'].apply(lambda x: x.replace(',', '')).astype(int)
    #home['out'] = home['out'].apply(lambda x: x.replace(',', '')).astype(int)
    home['total'] = home['stay'] + home['out']
    home['out_pct'] = home['out'] / home['total']

    taz2cell = pd.read_excel(
        '../Data/division_choice/' + ind.cell_name + '/taz2county_id.xlsx', engine='openpyxl')
    home = home.merge(taz2cell, on='taz_id')
    home['cell_id'] = home['cell_id'].astype(str)
    empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
        'cell_id'].astype(str)
    home = home[
        home['cell_id'].apply(lambda x: x not in empty_cells.values)]

    home_cell = home.groupby(['date', 'cell_id'])[
        ['stay', 'out', 'total']].sum().reset_index()
    home_cell['out_pct'] = home_cell['out'] / home_cell['total']
    home_cell = home_cell.set_index('date')
    home_cell = home_cell.groupby(by='cell_id')['out_pct'].rolling(7, center=True).mean()
    home_cell = home_cell.unstack(level=0).dropna()

    global_max = home_cell.apply(robust_max)
    global_min = home_cell.apply(robust_min)
    span = global_max - global_min
    relative_rate = home_cell.apply(
        lambda row: normelize(row, global_min, span), axis=1)

    result = dict()
    result['routine'] = avg_by_dates(relative_rate, '2020-02-02', '2020-02-29')
    result['no_school'] = avg_by_dates(relative_rate, '2020-03-14',
                                       '2020-03-16',
                                       weights={'2020-03-14': 2 / 7,
                                                '2020-03-15': 2.5 / 7,
                                                '2020-03-16': 2.5 / 7})
    result['no_work'] = avg_by_dates(relative_rate, '2020-03-17', '2020-03-25',
                                     weights={i: 1 / 14 if i.day in [17, 18, 24, 25] else 1 / 7
                                              for i in pd.date_range('2020-03-17', '2020-03-25')})

    result['no_100_meters'] = avg_by_dates(relative_rate, '2020-03-26', '2020-04-02',
                                           weights={i: 1 / 14 if i.day in [26, 2] else 1 / 7
                                                    for i in pd.date_range('2020-03-26', '2020-04-02')})

    result['no_bb'] = avg_by_dates(relative_rate, '2020-04-03', '2020-04-06',
                                   weights={i: 5 / 14 if i.day in [5, 6] else 1 / 7
                                            for i in pd.date_range('2020-04-03', '2020-04-06')
                                            }
                                   )

    result['full_lockdown'] = avg_by_dates(
        relative_rate,
        '2020-04-07',
        '2020-04-16',
        weights={i: 5 / 28 if i.day in [7, 12, 13, 16] else 1 / 21
                 for i in pd.date_range('2020-04-07', '2020-04-16')},
    )

    weights_release = dict()
    for i in pd.date_range('2020-04-17', '2020-05-02'):
        if i.day in [17, 18, 24, 25, 1, 2]:
            weights_release[i] = 1 / 21
        elif i.day in [28, 29]:
            weights_release[i] = 0
        else:
            weights_release[i] = 5 / 56
    result['release'] = avg_by_dates(
        relative_rate,
        '2020-04-17',
        '2020-05-02',
        weights=weights_release,
    )
    # save
    try:
        os.mkdir('../Data/stay_home')
    except:
        pass
    result['routine'].to_csv('../Data/stay_home/routine.csv')
    result['no_school'].to_csv('../Data/stay_home/no_school.csv')
    result['no_work'].to_csv('../Data/stay_home/no_work.csv')
    result['no_100_meters'].to_csv('../Data/stay_home/no_100_meters.csv')
    result['no_bb'].to_csv('../Data/stay_home/no_bb.csv')
    result['full_lockdown'].to_csv('../Data/stay_home/full_lockdown.csv')
    result['release'].to_csv('../Data/stay_home/release.csv')
    relative_rate.to_csv('../Data/stay_home/per_date.csv')


def create_demograph_sick_pop(ind):
    ### Creating demograph/sick_pop.csv
    taz2sick = pd.read_csv('../Data/sick/taz2sick.csv')

    taz2cell = pd.read_excel(
        '../Data/division_choice/' + ind.cell_name + '/taz2county_id.xlsx', engine='openpyxl')
    pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2, engine='openpyxl')
    ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]
    pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
    pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
    pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
    pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
    pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
        pop_dist['id']).join(pop_dist['tot_pop'])
    pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
    pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
    pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'], axis=0)
    pop_dist = pop_dist[['id', 'tot_pop']]

    taz2sick = taz2sick.merge(taz2cell, on='taz_id')
    taz2sick = taz2sick.merge(pop_dist, left_on='taz_id', right_on='id')
    taz2sick['cell_id'] = taz2sick['cell_id'].astype(str)
    empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')['cell_id'].astype(str)
    taz2sick = taz2sick[taz2sick['cell_id'].apply(lambda x: x not in empty_cells.values)]
    # taz2sick['cases_prop'] = taz2sick['cases_prop'] * taz2sick['tot_pop']

    taz2sick = taz2sick.groupby(by='cell_id')[['cases_prop']].apply(sum)
    taz2sick.name = 'cases_prop'
    taz2sick.to_csv('../Data/demograph/sick_prop.csv')


def create_stay_idx_routine(ind, start, end, date_beta_behave):
    ### Data loading
    stay_home_idx = pd.read_excel('../Data/stay_home/per_date.xlsx', engine= 'openpyxl',
                                index_col=0)
    stay_home_idx.columns = stay_home_idx.columns.astype(str)
    stay_home_idx.index = pd.to_datetime(stay_home_idx.index)
    stay_home_idx = stay_home_idx[pd.Timestamp(start):]

    # preparing model objects:
    stay_idx_t = []
    routine_vector = []
    d_tot = 500

    for i in pd.date_range(pd.Timestamp(start), pd.Timestamp(end)):
        stay_home_idx_daily = stay_home_idx.loc[i].values
        stay_home_idx_daily = expand_partial_array(
            mapping_dic=ind.region_gra_dict,
            array_to_expand=stay_home_idx_daily,
            size=len(ind.GRA),
        )
        stay_idx_t.append(stay_home_idx_daily)

        if i < pd.Timestamp(date_beta_behave):
            routine_vector.append(0)
        else:
            routine_vector.append(1)

    stay_home_idx_daily = stay_home_idx.iloc[-1].values
    stay_home_idx_daily = expand_partial_array(
        mapping_dic=ind.region_gra_dict,
        array_to_expand=stay_home_idx_daily,
        size=len(ind.GRA),
    )
    for i in range(d_tot - len(pd.date_range(pd.Timestamp(start), pd.Timestamp(end)))):
        stay_idx_t.append(stay_home_idx_daily)
        routine_vector.append(1)

    stay_idx_calibration = {
        'Non-intervention': stay_idx_t,
        'Intervention': [0] * 500,
    }

    routine_vector_calibration = {
        'Non-intervention': {
            'work': routine_vector,
            'not_work': routine_vector
        },
        'Intervention': {
            'work': [1] * 500,
            'not_work': [1] * 500,
        }
    }

    # save objects
    with open('../Data/parameters/stay_home_idx.pickle', 'wb') as handle:
        pickle.dump(stay_idx_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../Data/parameters/routine_t.pickle', 'wb') as handle:
        pickle.dump(routine_vector_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_full_matices(ind):
    ### Full Matrixes
    with (open('../Data/division_choice/' + ind.cell_name + '/mat_macro_model_df_30.pickle', 'rb')) as openfile:
        OD_dict = pickle.load(openfile)

    base_leisure = pd.read_csv('../Data/raw/leisure_mtx.csv', index_col=0)
    base_work = pd.read_csv('../Data/raw/work_mtx.csv', index_col=0)
    base_school = pd.read_csv('../Data/raw/school_mtx.csv', index_col=0)
    religion_dist = pd.read_csv('../Data/demograph/religion_dis.csv', index_col=0)
    age_dist_area = pd.read_csv('../Data/demograph/age_dist_area.csv', index_col=0)
    home_secularism = pd.read_excel('../Data/raw/secularism_base_home.xlsx', index_col=0, engine='openpyxl')
    home_haredi = pd.read_excel('../Data/raw/haredi_base_home.xlsx', index_col=0, engine='openpyxl')
    home_arabs = pd.read_excel('../Data/raw/arabs_base_home.xlsx', index_col=0, engine='openpyxl')

    # fix_shahaf_bug
    if ind.cell_name == '250':
        if len(str(OD_dict[list(OD_dict.keys())[0]].columns[0])) == 6:
            print('shahaf bug returned!!!!')
            for k in OD_dict.keys():
                OD_dict[k].columns = pd.Index(ind.G.values())
        if len(str(OD_dict[list(OD_dict.keys())[0]].index[0])) == 6:
            for k in OD_dict.keys():
                OD_dict[k].index = pd.Index(ind.G.values())

    # make sure index of area is string
    for k in OD_dict.keys():
        OD_dict[k].columns = OD_dict[k].columns.astype(str)
        OD_dict[k].index = OD_dict[k].index.astype(str)
        OD_dict[k] = OD_dict[k].filter(list(ind.G.values()), axis=1)
        OD_dict[k] = OD_dict[k].filter(list(ind.G.values()), axis=0)

    OD_const = OD_dict['routine', 1]
    OD_const.loc[:, :] = 1

    ############ no_mobility #############
    full_leisure_const = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_const,
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 21.2-14.3 #############
    full_leisure_routine = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['routine', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 14.3-16.3 #############
    full_leisure_no_school = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_school', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 17.3-25.3 #############
    full_leisure_no_work = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_work', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 26.3-2.4 #############
    full_leisure_no_100_meters = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_100_meters', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 3.4-6.4 #############
    full_leisure_no_bb = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_bb', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 7.4-16.4 #############
    full_leisure_full_lockdown = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['full_lockdown', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 17.4 - 4.5 #############
    full_leisure_release = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['release', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    ############ 5.5 - 11.5 #############
    full_leisure_back2routine = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['back2routine', 2],
        base_mat=base_leisure,
        age_dist_area=age_dist_area
    )

    # save matrix
    try:
        os.mkdir('../Data/base_contact_mtx')
    except:
        pass
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_routine.npz', full_leisure_routine)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_school.npz', full_leisure_no_school)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_work.npz', full_leisure_no_work)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_100_meters.npz', full_leisure_no_100_meters)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_bb.npz', full_leisure_no_bb)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_full_lockdown.npz', full_leisure_full_lockdown)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_release.npz', full_leisure_release)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_back2routine.npz', full_leisure_back2routine)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_const.npz', full_leisure_const)

    # creating school- work matrix;
    base_work_school = base_work.copy()
    base_work_school.loc['0-4'] = base_school.loc['0-4']
    base_work_school.loc['5-9'] = base_school.loc['5-9']
    base_work_school['0-4'] = base_school['0-4']
    base_work_school['5-9'] = base_school['5-9']
    # creating eye matrix
    eye_OD = OD_dict['routine', 1].copy()

    for col in eye_OD.columns:
        eye_OD[col].values[:] = 0
    eye_OD.values[tuple([np.arange(eye_OD.shape[0])] * 2)] = 1

    ############ no_mobility #############
    full_work_const = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_const,
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 21.2-14.3 #############
    full_work_routine = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['routine', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 14.3-16.3 #############
    full_work_no_school = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_school', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 17.3-25.3 #############
    full_work_no_work = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_work', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 26.3-2.4 #############
    full_work_no_100_meters = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_100_meters', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 3.4-6.4 #############
    full_work_no_bb = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['no_bb', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 7.4-16.4 #############
    full_work_full_lockdown = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['full_lockdown', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 17.4 - 4.5 #############
    full_work_release = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['release', 1],
        base_mat=base_work_school,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    ############ 5.5 - 11.5 #############
    full_work_back2routine = create_C_mtx_leisure_work(
        ind=ind,
        od_mat=OD_dict['back2routine', 1],
        base_mat=base_leisure,
        age_dist_area=age_dist_area,
        eye_mat=eye_OD,
    )

    # save matrix
    try:
        os.mkdir('../Data/base_contact_mtx')
    except:
        pass
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_routine.npz', full_work_routine)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_school.npz', full_work_no_school)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_work.npz', full_work_no_work)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_100_meters.npz', full_work_no_100_meters)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_bb.npz', full_work_no_bb)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_full_lockdown.npz', full_work_full_lockdown)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_release.npz', full_work_release)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_back2routine.npz', full_work_back2routine)
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_const.npz', full_work_const)

    ## Home Matrices
    full_home = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(list(ind.MI.values()), names=['age', 'area', 'age']),
        columns=OD_dict['routine', 0].index)
    religion_dist.set_index('cell_id', inplace=True)
    religion_dist.index = religion_dist.index.astype(str)

    # fill the matrix:
    for index in list(full_home.index):
        religion_area = religion_dist.loc[index[1]].copy()
        cell_val = religion_area['Orthodox'] * home_haredi.loc[index[0]][
            index[2]] + \
                   religion_area['Sacular'] * home_secularism.loc[index[0]][
                       index[2]] + \
                   religion_area['Christian'] * home_arabs.loc[index[0]][
                       index[2]] + \
                   religion_area['Other'] * home_secularism.loc[index[0]][
                       index[2]] + \
                   religion_area['Druze'] * home_arabs.loc[index[0]][
                       index[2]] + \
                   religion_area['Muslim'] * home_arabs.loc[index[0]][index[2]]
        full_home.loc[index] = (eye_OD.loc[index[1]] * cell_val) / \
                               age_dist_area[index[2]]

    full_home = csr_matrix(full_home.unstack().reorder_levels(
        ['area', 'age']).sort_index().values.astype(float))
    # save matrix
    try:
        os.mkdir('../Data/base_contact_mtx')
    except:
        pass
    scipy.sparse.save_npz('../Data/base_contact_mtx/full_home.npz', full_home)


def create_parameters_indices(ind):
    with open('../Data/parameters/indices.pickle', 'wb') as handle:
        pickle.dump(ind, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_f0(ind: Indices):
    """
This function builds the dictionary of scens - proportions for each scen. each scen holds an array of 540X1 (for each age - region - risk)
    """
    ### Asymptomatic
    asymp = pd.read_csv('../Data/raw/asymptomatic_proportions_yotams_update.csv', index_col=0, usecols=[0,1,2,3])
    asymp = asymp.iloc[:10, :4]
    f0_full = {}  # dict that contains the possible scenarios
    # asymptomatic with risk group, high risk with 0
    f_init = np.zeros(len(ind.R.values()) * len(ind.A.values()))
    for i in [1, 2, 3]:
        f_tmp = f_init.copy()
        # Puts the same values in the first half and the second half of the array:
        f_tmp[:9] = asymp['Scenario ' + str(i)].values[:-1]
        f_tmp[9:] = asymp['Scenario ' + str(i)].values[:-1]
        # Updating the main dictionary, stretching the proportions from 18X1 to 540X1
        f0_full['Scenario' + str(i)] = expand_partial_array(ind.risk_age_dict, f_tmp, len(ind.N))
    # Save
    try:
        os.mkdir('../Data/parameters')
    except:
        pass
    with open('../Data/parameters/f0_full.pickle', 'wb') as handle:
        pickle.dump(f0_full, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_eps_by_region_prop(ind, age_dist):
    asymp = pd.read_csv('../Data/raw/asymptomatic_proportions.csv', index_col=0)

    ### eps by region proportion
    risk_dist = pd.read_csv('../Data/raw/population_size.csv')
    init_I_dis_italy = pd.read_csv('../Data/raw/init_i_italy.csv')['proportion'].values[:-1]
    f_init = pd.read_pickle('../Data/parameters/f0_full.pickle')
    init_I_IL = {}
    init_I_dis = {}
    for i in [1, 2, 3, 4]:
        scen = 'Scenario' + str(i)
        f_init_i = f_init[scen][:(len(ind.R) * len(ind.A))]
        init_I_IL[scen] = (491. / (1 - asymp['Scenario ' + str(i)].values[-1])) / israel_pop
        init_I_dis[scen] = init_I_dis_italy * init_I_IL[scen]

    # Loading data
    region_prop = pd.read_csv('../Data/demograph/sick_prop.csv', index_col=0)[
        'cases_prop'].copy()  # proportion of people in each county
    region_prop.index = region_prop.index.astype(str)
    risk_prop = pd.read_csv('../Data/raw/risk_dist.csv', index_col=0)[
        'risk'].copy()  # Proportion of high risk people for each age
    eps_t_region = {}
    for sc, init_I in zip(init_I_dis.keys(), init_I_dis.values()):
        eps_temp = []
        for t in range(1000):
            if t < len(init_I):
                # empty array for day t
                day_vec = np.zeros(len(ind.N))
                # fill in the array, zero for intervention groups
                for key in ind.N.keys():
                    day_vec[key] = init_I[t] * region_prop[ind.N[key][0]] * \
                                   age_dist[ind.N[key][2]] * \
                                   (risk_prop[ind.N[key][2]] ** (1 - (ind.N[key][1] == 'Low'))) * \
                                   ((1 - risk_prop[ind.N[key][2]]) ** (ind.N[key][1] == 'Low'))
                eps_temp.append(day_vec)
            else:
                eps_temp.append(0.0)

            eps_t_region[sc] = eps_temp
    # save eps:
    with open('../Data/parameters/eps_by_region.pickle', 'wb') as handle:
        pickle.dump(eps_t_region, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_hosptialization(ind):
    ### hospitalization
    hosp_init = pd.read_csv('../Data/raw/hospitalizations.csv')
    hosp = expand_partial_array(ind.risk_age_dict, hosp_init['pr_hosp'].values, len(ind.N))
    # Save
    with open('../Data/parameters/hospitalizations.pickle', 'wb') as handle:
        pickle.dump(hosp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_vents_proba(ind):
    ### Ventilation
    vents_init = pd.read_csv('../Data/raw/vent_proba.csv')
    vent = expand_partial_array(ind.risk_age_dict, vents_init['pr_vents'].values, len(ind.N))
    # Save
    with open('../Data/parameters/vents_proba.pickle', 'wb') as handle:
        pickle.dump(vent, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_C_calibration(ind):
    ### Calibration contact matrix
    full_mtx_home = scipy.sparse.load_npz(
        '../Data/base_contact_mtx/full_home.npz')

    full_mtx_work = {
        'routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_routine.npz'),
        'no_school': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_school.npz'),
        'no_work': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_work.npz'),
        'no_100_meters': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_100_meters.npz'),
        'no_bb': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_bb.npz'),
        'full_lockdown': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_full_lockdown.npz'),
        'release': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_release.npz'),
        'back2routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_back2routine.npz'),
    }

    full_mtx_leisure = {
        'routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_routine.npz'),
        'no_school': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_school.npz'),
        'no_work': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_work.npz'),
        'no_100_meters': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_100_meters.npz'),
        'no_bb': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_bb.npz'),
        'full_lockdown': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_full_lockdown.npz'),
        'release': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_release.npz'),
        'back2routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_back2routine.npz'),
    }

    full_mtx_const = {
        'work': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_const.npz'),
        'leisure': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_const.npz'),
    }

    C_calibration = {}
    C_const = {}
    d_tot = 500
    # no intervation are null groups
    home_inter = []
    work_inter = []
    leis_inter = []

    work_const = []
    leis_const = []

    for i in range(d_tot):
        home_inter.append(csr_matrix((full_mtx_home.shape[0], full_mtx_home.shape[1])))
        work_inter.append(csr_matrix((full_mtx_work['routine'].shape[0], full_mtx_work['routine'].shape[1])))
        leis_inter.append(csr_matrix((full_mtx_leisure['routine'].shape[0], full_mtx_leisure['routine'].shape[1])))
        work_const.append(full_mtx_const['work'])
        leis_const.append(full_mtx_const['leisure'])

    # Intervantion
    home_no_inter = []
    work_no_inter = []
    leis_no_inter = []

    # first days of routine from Feb 21st - March 13th
    d_rout = 9 + 13
    for i in range(d_rout):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['routine'])
        leis_no_inter.append(full_mtx_leisure['routine'])

    # first days of no school from March 14th - March 16th
    d_no_school = 3
    for i in range(d_no_school):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['no_school'])
        leis_no_inter.append(full_mtx_leisure['no_school'])

    # without school and work from March 17th - March 25th
    d_no_work = 9
    for i in range(d_no_work):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['no_work'])
        leis_no_inter.append(full_mtx_leisure['no_work'])

    # 100 meters constrain from March 26th - April 2nd
    d_no_100_meters = 8
    for i in range(d_no_100_meters):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['no_100_meters'])
        leis_no_inter.append(full_mtx_leisure['no_100_meters'])

    # Bnei Brak quaranrine from April 3rd - April 18th
    d_bb = 16
    for i in range(d_bb):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['no_bb'])
        leis_no_inter.append(full_mtx_leisure['no_bb'])

    # full lockdown from April 7th - April 16th
    d_full_lockdown = 10
    for i in range(d_full_lockdown):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['full_lockdown'])
        leis_no_inter.append(full_mtx_leisure['full_lockdown'])

    # full release from April 17th - May 4th
    d_release = 18
    for i in range(d_release):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['release'])
        leis_no_inter.append(full_mtx_leisure['release'])

    # full release from April 5th - May 11th
    for i in range(d_tot - (d_rout + d_no_school + d_no_work + d_no_100_meters + d_bb + d_full_lockdown + d_release)):
        home_no_inter.append(full_mtx_home)
        work_no_inter.append(full_mtx_work['back2routine'])
        leis_no_inter.append(full_mtx_leisure['back2routine'])

    C_calibration['home_inter'] = home_inter
    C_calibration['work_inter'] = work_inter
    C_calibration['leisure_inter'] = leis_inter
    C_calibration['home_non'] = home_no_inter
    C_calibration['work_non'] = work_no_inter
    C_calibration['leisure_non'] = leis_no_inter

    C_const['home_inter'] = home_no_inter
    C_const['work_inter'] = work_const
    C_const['leisure_inter'] = leis_const
    C_const['home_non'] = home_no_inter
    C_const['work_non'] = work_const
    C_const['leisure_non'] = leis_const

    # Save
    with open('../Data/parameters/C_calibration.pickle', 'wb') as handle:
        pickle.dump(C_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../Data/parameters/C_const.pickle', 'wb') as handle:
        pickle.dump(C_const, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_is_haredim(ind):
    ### Haredim vector - size of 270X1. each value repeats itself on every 9 sequential indexes (due to 9 age-groups)
    hared_dis = pd.read_csv('../Data/demograph/religion_dis.csv', index_col=0)[
        ['cell_id', 'Orthodox']].copy()
    hared_dis.set_index('cell_id', inplace=True)
    hared_dis.index = hared_dis.index.astype(str)
    # Creating model orthodox dist. and save it as pickle
    model_orthodox_dis = np.zeros(len(ind.GA))
    for i in ind.GA.keys():
        model_orthodox_dis[i] = hared_dis.loc[str(ind.GA[i][0])]
    with open('../Data/parameters/orthodox_dist.pickle', 'wb') as handle:
        pickle.dump(model_orthodox_dis, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_is_arab(ind):
    ### Arabs vector - size of 270X1. each value repeats itself on every 9 sequential indexes (due to 9 age-groups)
    arab_dis = pd.read_csv('../Data/demograph/religion_dis.csv', index_col=0)[
        ['cell_id', 'Druze', 'Muslim', 'Christian']].copy()
    arab_dis.set_index('cell_id', inplace=True)
    arab_dis.index = arab_dis.index.astype(str)
    arab_dis = arab_dis.sum(axis=1)

    # Creating model arab dist. and save it as pickle
    model_arab_dis = np.zeros(len(ind.GA))
    for i in ind.GA.keys():
        model_arab_dis[i] = arab_dis.loc[str(ind.GA[i][0])]

    with open('../Data/parameters/arab_dist.pickle', 'wb') as handle:
        pickle.dump(model_arab_dis, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_init_pop(ind):
    """
This function creates the proportion vector of Israel's population for each cell in ind.N (for each region - risk - age  intersection)
    """
    age_dist_area = pd.read_csv('../Data/demograph/age_dist_area.csv')
    age_dist_area.drop(['Unnamed: 0'], axis=1, inplace=True)
    age_dist_area.set_index('cell_id', inplace=True)
    age_dist_area = age_dist_area.stack()
    init_pop = expand_partial_array(ind.region_age_dict, age_dist_area.values, len(ind.N))
    risk_pop = pd.read_csv('../Data/raw/risk_dist.csv')
    risk_pop.set_index('Age', inplace=True)
    risk_pop['High'] = risk_pop['risk']
    risk_pop['Low'] = 1 - risk_pop['risk']
    risk_pop.drop(columns='risk', axis=1, inplace=True)
    risk_pop = risk_pop.stack()
    risk_pop.index = risk_pop.index.swaplevel(0, 1)
    risk_pop = risk_pop.unstack().stack()

    # r == risk (High/Low),		a == age_group (e.g. '0-4', '5-9', '10-19'),		g_idx == array of corresponding indexes in N
    # Multiplication - explained: for each combination of county and age_group, we multiply by the probability to be in High/Low risk
    for (r, a), g_idx in zip(ind.risk_age_dict.keys(), ind.risk_age_dict.values()):
        init_pop[g_idx] = init_pop[g_idx] * risk_pop[r, a]

    # Save
    # init_pop == proportion of initial population for each intersection in N
    with open('../Data/parameters/init_pop.pickle', 'wb') as handle:
        pickle.dump(init_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_model_initialization(ind: Indices, date):
    """
This function creates a dictionary that initializes the model - for each scenario, it holds the proportion of Israel pop for each compartment
E.g. { scen_1 : {S_1 : [540X1], E_1 : [540X1],  Ia_1 : [540X1], Is : [540X1], R : [540X1]}, scen_2 : {S_1 : [540X1], E_1 : [540X1], ...} }
Input: ind == Indices.ind,		date == start date of the model
    """
    # Reading
    with(open('../Data/parameters/init_pop.pickle', 'rb')) as openfile:
        init_pop = pickle.load(openfile)
    with(open('../Data/parameters/f0_full.pickle', 'rb')) as openfile:
        f0_full = pickle.load(openfile)
    sick_df = pd.read_excel(r'../Data/sick/sick_data_raw_stat_areas_updated.xlsx', engine= 'openpyxl')
    sick_df['date'] = pd.to_datetime(sick_df['date'])
    # Initializing empty dictionary
    init_model = {}

    ## Initializing the S_1 compartment - proportion of people who haven't been sick
    # Getting the start date of the df
    exact_date_df = sick_df[sick_df['date'] == date]
    # Eliminating the categorical values
    filtered_df = exact_date_df[
        (exact_date_df['accumulated_cases'] != '0') & (exact_date_df['accumulated_cases'] != '<15')]
    accumulated_cases_series = filtered_df['accumulated_cases'].astype(float)
    # Getting the amount of cases up until 15th May. Factoring by 1.5 because of the '<15' values - hidden amounts
    accumulated_cases = np.sum(accumulated_cases_series) * 1.5
    total_susceptible_prop = (israel_pop - accumulated_cases) / israel_pop
    # Spreading the total proportion of susceptible individuals over the age- region- risk- intersections
    # This results in a vector of size ( len(ind.N) , 1 )
    init_S_1 = total_susceptible_prop * init_pop

    ## Initializing the E_1 compartment - new cases (diff of acc cases between 'date' param and 1 day before)
    yesterday_date = date - datetime.timedelta(days=1)
    yesterday_date_df = sick_df[sick_df['date'] == yesterday_date]
    # Eliminating the categorical values
    filtered_df = yesterday_date_df[
        (yesterday_date_df['accumulated_cases'] != '0') & (yesterday_date_df['accumulated_cases'] != '<15')]
    yesterday_accumulated_cases_series = filtered_df['accumulated_cases'].astype(float)
    # Getting the amount of cases up until 15th May. Factoring by 1.5 because of the '<15' values - those are hidden amounts
    accumulated_cases_yesterday = np.sum(yesterday_accumulated_cases_series) * 1.5
    total_new_cases_prop = (accumulated_cases - accumulated_cases_yesterday) / israel_pop
    init_E_1 = total_new_cases_prop * init_pop

    ## Initializing the R_1 compartment - proportion of recovered people
    # Eliminating the categorical values
    filtered_df = exact_date_df[
        (exact_date_df['accumulated_recoveries'] != '0') & (exact_date_df['accumulated_recoveries'] != '<15')]
    accumulated_recoveries_series = filtered_df['accumulated_recoveries'].astype(float)
    # Getting the amount of recoveries up until 15th May.
    accumulated_recoveries = np.sum(accumulated_recoveries_series) * 1.5
    total_recoveries_prop = accumulated_recoveries / israel_pop
    init_R_1 = total_recoveries_prop * init_pop
    # Initializing Is_1 and Ia_1 compartments
    init_Is_1 = 1e-4 * init_pop
    init_Ia_1 = 2 * 1e-4 * init_pop
    # Finalized starting condition model:
    init_model = {'init_S_1': init_S_1, 'init_E_1': init_E_1, 'init_R_1': init_R_1, \
                  'init_Is_1': init_Is_1, 'init_Ia_1': init_Ia_1}

    # Saving
    with open('../Data/parameters/init_model.pickle', 'wb') as handle:
        pickle.dump(init_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_s_1_to_v1_transition(ind, start, end):
    """
    This function takes a DF which is produced by the First_dose_vaccination_by_30area_age.ipynb script,
    and generates a list of lists as described above
    """
    proportion_df = pd.read_excel('../Data/Vaccinations/First_dose_vaccination_by_30area_age.xlsx', engine= 'openpyxl')
    proportion_df['Date'] = pd.to_datetime(proportion_df['Date'])
    # Shifting the data 3 weeks ahead - as per Dan's request (changing every date to 3 week forward's date)
    proportion_df.set_index('Date', drop= True, inplace= True)
    #proportion_df_shifted = proportion_df.shift(periods= 14, freq= "D")
    list_of_lists = []
    for date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end)):  # Iterates over the whole model's period of time
        # If the date is before vaccines were given - add zeros
        if date not in pd.to_datetime(proportion_df.index.unique()):
            list_of_lists.append(np.zeros(shape=len(ind.N), dtype=float))

        else:
            # Slicing for a specific date and relevant columns
            temp_df = proportion_df[proportion_df.index == date].iloc[ : , 2:11]
            proportion_arr = temp_df.to_numpy().flatten() * 1.1 # Creating array from the matrix

            s_to_v1_daily = expand_partial_array(  # Expanding the array
                mapping_dic=ind.region_age_dict,
                array_to_expand=proportion_arr,
                size=len(ind.N)
            )
            list_of_lists.append(s_to_v1_daily)  # Creating an array from the df, then appending the array
    s_to_v1 = np.array(list_of_lists, dtype=float)

    # Saving:
    with open('../Data/parameters/s_to_v1_transition.pickle', 'wb') as handle:
        pickle.dump(s_to_v1, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_v_2_to_v_3_transition(ind, start, end):
    """
    This function takes a DF which is produced by the First_dose_vaccination_by_30area_age.ipynb script,
    and generates a list of lists as described above
    """
    proportion_df = pd.read_excel('../Data/Vaccinations/Third_dose_vaccination_by_30area_age.xlsx', engine= 'openpyxl')
    proportion_df['Date'] = pd.to_datetime(proportion_df['Date'])
    list_of_lists = []
    for date in pd.date_range(pd.Timestamp(start), pd.Timestamp(end)):  # Iterates over the whole model's period of time
        # Slicing for a specific date and relevant columns
        temp_df = proportion_df[proportion_df['Date'] == date].iloc[:, 3:12]
        # If the date is before vaccines were given - add zeros
        if date not in pd.to_datetime(proportion_df['Date'].unique()):
            list_of_lists.append(np.zeros(shape=len(ind.N), dtype=float))

        # If all values in the matrix on specific date are zeroes - add all zeroes to the final outcome
        elif (temp_df.sum(axis=0) == 0).all(axis=0):
            list_of_lists.append(np.zeros(shape=len(ind.N), dtype=float))

        else:
            # Creating array from the matrix
            temp_df = temp_df.to_numpy().flatten()

            v2_to_v3_daily = expand_partial_array(  # Expanding the array
                mapping_dic=ind.region_age_dict,
                array_to_expand=temp_df,
                size=len(ind.N)
            )
            list_of_lists.append(v2_to_v3_daily)  # Creating an array from the df, then appending the array
    v2_to_v3 = np.array(list_of_lists, dtype=float)

    # Saving:
    with open('../Data/parameters/v2_to_v3_transition.pickle', 'wb') as handle:
        pickle.dump(v2_to_v3, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_alpha_variant_vec(start, end):
    """
    Parameters: start - start date of the whole model,	end - end date of the whole model
    This function creates the vector of the weighted average of the morbidity that the variant accounts for
    """
    alpha_var_df = pd.read_excel(r'../Data/raw/alpha_variant_wt_morbidity.xlsx', engine= 'openpyxl', skiprows=2, usecols=[0, 3])
    alpha_var_df.columns = ['date', 'p_variant']
    #alpha_var_df['date'] = pd.to_datetime(alpha_var_df['date']) - pd.to_timedelta(9, unit= 'd')
    alpha_var_vec = []
    # Creating a date range and converting its type to timestamp:
    whole_model_dates = pd.date_range(start=start, end=end)

    for date in whole_model_dates:
        if date in list(alpha_var_df['date']):  # If the alpha variant is relevant
            p_variant = alpha_var_df[alpha_var_df['date'] == date].iloc[0][
                1]  # Retrieve the proportion of the variant's morbidity
            weighted_avg = p_variant * 1.4 + (1 - p_variant) * 1  # Calc the weighted avg of the variant and the WT
            alpha_var_vec.append(weighted_avg)
        else:  # If there was no alpha variant, multiply the force of infection by 1 (DO NOTHING)
            alpha_var_vec.append(1)
    df_alpha_var = pd.DataFrame(alpha_var_vec, columns= ['alpha_var'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns= ['date'])
    df_with_dates = df_alpha_var.join(dates_dict)#), left_on= df_alpha_var.index, right_on= dates_dict.index)
    #$df_with_dates.drop('key_0', axis= 1, inplace= True)
    alpha_var_vec = np.array(df_with_dates['alpha_var'])

    # Saving:
    with open('../Data/parameters/alpha_variant_vec.pickle', 'wb') as handle:
        pickle.dump(alpha_var_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_delta_variant_vec(start, end):
    """
    Parameters: start - start date of the whole model,	end - end date of the whole model
    This function creates the vector of the weighted average of the morbidity that the variant accounts for
    """
    delta_var_df = pd.read_excel(r'../Data/raw/delta_variant_wt_morbidity.xlsx', engine= 'openpyxl', skiprows=2, usecols=[0, 3])
    delta_var_df.columns = ['date', 'p_variant']
    #delta_var_df['date'] = pd.to_datetime(delta_var_df['date']) + pd.to_timedelta(30, unit= 'd')
    # delta_var_df.set_index('date', inplace= True)
    # delta_var_df = delta_var_df.shift(periods= -12).dropna()
    # delta_var_df.reset_index(inplace= True)
    delta_var_vec = []
    whole_model_dates = pd.date_range(start=start, end=end)

    for date in whole_model_dates:
        if date in list(delta_var_df['date']):  # If the delta variant is relevant
            p_variant = delta_var_df[delta_var_df['date'] == date].iloc[0][
                1]  # Retrieve the proportion of the variant's morbidity
            weighted_avg = p_variant * 1.4 * 1.6 + (
                    1 - p_variant) * 1.4  # Calc the weighted avg of the delta variant and the alpha variant
            delta_var_vec.append(weighted_avg)
        else:  # If there was no delta variant, multiply the force of infection by 1 (DO NOTHING)
            delta_var_vec.append((1))

    df_delta_var = pd.DataFrame(delta_var_vec, columns=['delta_var'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns=['date'])
    df_with_dates = df_delta_var.join(dates_dict)#, left_on= df_delta_var.index, right_on= dates_dict.index)
    #df_with_dates.drop('key_0', axis=1, inplace=True)
    delta_var_vec = np.array(df_with_dates['delta_var'])

    # Saving:
    with open('../Data/parameters/delta_variant_vec.pickle', 'wb') as handle:
        pickle.dump(delta_var_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_omicron_variant_vec(end, start= start_model):
    """
   Parameters: start - start date of the whole model,	end - end date of the whole model
   This function creates the vector of the omicron influence on the force of infection
   """
    omicron_var_vec = []
    omicron_dates = pd.date_range('2022-01-01', '2022-04-01')
    whole_model_dates = pd.date_range(start=start, end=end)
    for date in whole_model_dates:
        if date in omicron_dates:  # If the delta variant is relevant
            variant_force = 1 * 1.4 * 1.5 * 2.4 # Omicron infectious compared to delta variant
            omicron_var_vec.append(variant_force)
        else:  # If there was no delta variant, multiply the force of infection by 1 (DO NOTHING)
            omicron_var_vec.append((1))

    df_omicron_var = pd.DataFrame(omicron_var_vec, columns=['omicron_var'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns=['date'])
    df_with_dates = df_omicron_var.join(dates_dict)

    # Saving:
    with open('../Data/parameters/omicron_variant_vec.pickle', 'wb') as handle:
        pickle.dump(omicron_var_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_neutralized_alpha_delta_variant_vecs():
    """
    Creating 2 vectors for the alpha and the delta variants, all ones, to neutralize the effect of these parameters
    when running the model over it's end date
    """
    variant_vec = np.ones(shape=(3650,1))
    # Saving:
    with open('../Data/parameters/neutralized_alpha_variant_vec.pickle', 'wb') as handle:
        pickle.dump(variant_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../Data/parameters/neutralized_delta_variant_vec.pickle', 'wb') as handle:
        pickle.dump(variant_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_isolation_morbidity_vector():
    isolated_cases_ratio = pd.read_csv(r'../Data/sick/isolation_morbidity_ratio_vector.csv')
    # Taking only the relevant column
    isolated_cases_ratio_array = np.array(isolated_cases_ratio['isolated_cases_ratio'])
    # Saving to pickle file
    with open('../Data/parameters/isolation_morbidity_ratio_vector.pickle', 'wb') as handle:
        pickle.dump(isolated_cases_ratio_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_neutralized_isolation_morbidity_vector():
    """
    Creating a vector to neutralize the effect of the isolation parameter when running the model over it's end date.
    """
    isolation_vec = np.zeros(shape=(3650,1))
    # Saving to pickle file
    with open('../Data/parameters/neutralized_isolation_morbidity_ratio_vector.pickle', 'wb') as handle:
        pickle.dump(isolation_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_extended_isolation_morbidity_vector():
    """
    Creating an extended vector of isolations to for the calibration_validation_omicron script
    """
    isolation_vec = pd.read_csv(r'../Data/sick/isolation_morbidity_ratio_vector_extended.csv')
    # Taking only the relevant column
    isolated_cases_ratio_array = np.array(isolation_vec['isolated_cases_ratio'])
    # Saving to pickle file
    with open('../Data/parameters/isolation_morbidity_ratio_vector_extended.pickle', 'wb') as handle:
        pickle.dump(isolated_cases_ratio_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_lockdown_vec(start: pd.datetime, end: pd.datetime):
    """This function gets the start and end dates of the *whole* model,
And returns two binary vectors that represent when there was lockdown/isolation.
1 - for relevant dates (e.g. lockdown was lifted), else 0.
The isolation vector is opposite to the lockdown vector - if lockdown was lifted, there were no isolations, and vice versa
"""
    whole_model_dates_list = pd.date_range(start=start, end=end)
    second_lockdown = pd.date_range(start='2020-09-15', end='2020-10-20')
    third_lockdown = pd.date_range(start='2020-12-27', end='2021-03-20')
    forth_lockdown = pd.date_range(start='2021-08-18', end='2021-10-22')
    # Shifting the dates (to delay the impact of the lockdown - lockdowns starting to reduce
    # second_lockdown_shifted = second_lockdown.shift(periods= 14, freq= "D")
    # third_lockdown_shifted = third_lockdown.shift(periods= 14, freq="D")
    # Creating list of dates indicating active lockdown
    lockdown_dates = second_lockdown.append(third_lockdown).append(forth_lockdown)
    # Initializing the lockdown vector
    lockdown_vec = np.zeros(len(whole_model_dates_list))
    # Iterating over the dates of the model
    for i, date in enumerate(whole_model_dates_list):
        # if lockdown was lifted:
        if date in lockdown_dates:
            lockdown_vec[i] = 1
        # if there was no lockdown
        else:
            lockdown_vec[i] = 0

    # For debugging - checking the correctness of the vector with actual dates:
    df_lockdown = pd.DataFrame(lockdown_vec, columns=['lockdown'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns=['date'])
    df_with_dates = df_lockdown.join(dates_dict)
    ## End of debug
    # Saving:
    with open('../Data/parameters/lockdown_vec.pickle', 'wb') as handle:
        pickle.dump(lockdown_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_activity_vec(start: pd.datetime, end: pd.datetime):
    """This function gets the start and end dates of the *whole* model, And returns a vector of values from the workplace analysis. It contains:
    1 - if there was no lockdown lifted), else, if there was lockdown - it contains the value of the raw workplace vector"""
    # Reading the raw workplace data
    with open('../Data/parameters/visitors_covid.pickle', 'rb') as pickle_in:
        df_raw_workplace = pickle.load(pickle_in)
    # Getting the model's dates and the lockdown dates
    whole_model_dates_list = pd.date_range(start=start, end=end)
    second_lockdown = pd.date_range(start='2020-09-15', end='2020-10-20')
    third_lockdown = pd.date_range(start='2020-12-27', end='2021-03-20')
    forth_lockdown = pd.date_range(start='2021-08-18', end='2021-10-22')
    # Creating list of dates indicating active lockdown
    lockdown_dates = second_lockdown.append(third_lockdown).append(forth_lockdown)
    # Initializing the lockdown vector
    activity_vec = np.zeros(len(whole_model_dates_list))
    # Iterating over the dates of the model
    for i, date in enumerate(whole_model_dates_list):
        # if lockdown was lifted:
        if date in lockdown_dates:
            activity_vec[i] = df_raw_workplace.loc[date]   # Getting the actual number our of the raw workplace df
        # if there was no lockdown
        else:
            activity_vec[i] = 1
    # For debugging - checking the correctness of the vector with actual dates:
    df_activity = pd.DataFrame(activity_vec, columns=['activity'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns=['date'])
    df_with_dates = df_activity.join(dates_dict)
    ## End of debug
    # Saving:
    with open('../Data/parameters/activity_vec.pickle', 'wb') as handle:
        pickle.dump(activity_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_neutralized_lockdown_vec():
    """
   Creating a vector to neutralize the effect of the lockdown parameter when running the model over it's end date.
   """
    lockdown_vec = np.zeros(shape=(3650,1))
    # Saving:
    with open('../Data/parameters/neutralized_lockdown_vec.pickle', 'wb') as handle:
        pickle.dump(lockdown_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_schoool_vec(start: pd.datetime, end: pd.datetime):
    """This function gets the start and end dates of the *whole* model,
	And returns a binary vectors that represent when there was school.
	0 - for relevant dates (e.g. there was school), else (no school) -> 1. (no school -> lower force of infection -> power of 1)
	"""
    whole_model_dates_list = pd.date_range(start=start, end=end)
    first_school_period = pd.date_range(start='2020-05-15', end='2020-07-01')
    second_school_period = pd.date_range(start='2020-08-20', end='2020-09-06')
    third_school_period = pd.date_range(start='2020-11-02', end='2021-02-15')

    # Creating list of dates indicating active schools
    school_dates = first_school_period.append(second_school_period).append(third_school_period)
    # Initializing the school vector
    school_vec = np.zeros(len(whole_model_dates_list))
    # Iterating over the dates of the model
    for i, date in enumerate(whole_model_dates_list):
        # if there was school -> we assign value of 0 (higher force of infection):
        if date in school_dates:
            school_vec[i] = 0
        else:
            school_vec[i] = 1
    # Adding dates for monitoring and debug purposes
    df_school = pd.DataFrame(school_vec, columns=['school'])
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'rb') as pickle_in:
        index_t_to_date_mapper = pickle.load(pickle_in)
    dates_dict = pd.DataFrame(index_t_to_date_mapper.values(), columns=['date'])
    df_with_dates = df_school.join(dates_dict)
    #df_with_dates.drop('key_0', axis=1, inplace=True)
    # Saving:
    with open('../Data/parameters/school_vec.pickle', 'wb') as handle:
        pickle.dump(school_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_neutralized_schoool_vec():
    """
     Creating a vector to neutralize the effect of the school parameter when running the model over it's end date.
     """
    school_vec = np.zeros(shape=(3650,1))
    # Saving:
    with open('../Data/parameters/neutralized_school_vec.pickle', 'wb') as handle:
        pickle.dump(school_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_neutralized_transition_rate_to_V_3():
    """ Creating a vector to remove the transition between V_2 and S_2 to V_3 when running the model's
     simulations over it's end date."""
    zero_vec = np.zeros(shape=(3650,1))
    # Saving:
    with open('../Data/parameters/neutralized_transition_rate_to_V_3.pickle', 'wb') as handle:
        pickle.dump(zero_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)


def creating_index_t_to_date_mapper():#(start, end):
    """
    start - start date of the model
    end - end date of the model
    Returns a mapper - each date to it's corresponding day t
    """
    # Creating a mapper dict - such that each key is index (t in the model), each value is the corresponding date
    mapper_dict = {}
    whole_model_dates = pd.date_range(start= '2020-05-15', end= '2031-05-15')
    for i, date in enumerate(whole_model_dates):
        mapper_dict[i] = date
    # Saving
    with open('../Data/parameters/index_t_to_date_mapper.pickle', 'wb') as handle:
        pickle.dump(mapper_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def creating_calib_index_t_to_date_mapper(start, end):
    """
    start - start date of the model
    end - end date of the model
    Returns a mapper - each date to it's corresponding day t
    """
    # Creating a mapper dict - such that each key is index (t in the model), each value is the corresponding date
    mapper_dict = {}
    whole_model_dates = pd.date_range(start= start, end= end)
    for i, date in enumerate(whole_model_dates):
        mapper_dict[i] = date
    # Saving
    with open('../Data/parameters/calib_index_t_to_date_mapper.pickle', 'wb') as handle:
        pickle.dump(mapper_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#############################
# def create_init_pop

### define indices
ind = Indices(cell_name)

age_dist = {'0-4': 0.02, '5-9': 0.02, '10-19': 0.11, '20-29': 0.23, '30-39': 0.15,
            '40-49': 0.14, '50-59': 0.14, '60-69': 0.11, '70+': 0.08}

#create_neutralized_transition_rate_to_V_3()
#
#creating_index_t_to_date_mapper()#(start= start_model, end= simulation_end_model)
#
#creating_calib_index_t_to_date_mapper(start= '2020-05-15', end= '2021-10-25')
#
#create_alpha_variant_vec(start= start_model, end= end_model)
#
#create_delta_variant_vec(start= start_model, end= end_model)
#
# create_omicron_variant_vec(end= '2022-05-01', start= start_model)
#
# create_parameters_s_1_to_v1_transition(ind, start= start_model, end= end_model)
#
#create_parameters_v_2_to_v_3_transition(ind, start= start_model, end= 'end')
#
# create_demograph_age_dist_empty_cells(ind)
#
#ind = create_paramaters_ind(ind)
#
#create_demograph_religion(ind)
#
#create_stay_home(ind)
#
#create_demograph_sick_pop(ind)
#
#create_stay_idx_routine(ind, '2020-02-20', '2020-05-08', '2020-03-14')
#
#create_full_matices(ind)
#
#create_parameters_indices(ind)
#
#create_parameters_f0(ind)
#
#create_parameters_eps_by_region_prop(ind, age_dist)
#
#create_parameters_hosptialization(ind)
#
#create_parameters_vents_proba(ind)
#
#create_parameters_C_calibration(ind)
#
# create_parameters_is_haredim(ind)
#
# create_parameters_is_arab(ind)
#
# create_init_pop(ind)
#
#create_model_initialization(ind, date= start_model)
#
# create_lockdown_vec(start= start_model,end= end_model)
#
# create_activity_vec(start= start_model, end= end_model)
#
#create_schoool_vec(start= start_model ,end= end_model)
#
# create_isolation_morbidity_vector()
#
# create_neutralized_alpha_delta_variant_vecs()
#
# create_neutralized_isolation_morbidity_vector()
#
# create_neutralized_lockdown_vec()
#
# create_neutralized_schoool_vec()
#
#create_extended_isolation_morbidity_vector()