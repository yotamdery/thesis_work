#!/usr/bin/env python3
import sys, inspect
import numpy as np
import pandas as pd

from SEIR_full.parameters import *
from SEIR_full.utils import *
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

########################
# -- Plot functions -- #
########################
## CONSTANT VARIABLES: ##
# names for the subplots
SUBPLOTS_NAMES = ['Low inventory', 'Medium inventory', 'High inventory']
COLORS_LIST = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#46039f', '#0d0887', '#46039f']
SIM_DIR_3_years = '/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/notebooks/Yotams/simulations/3_years_run/'
SIM_DIR_10_years = '/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/notebooks/Yotams/simulations/10_years_run/'


def plot_global_sensitivity_analysis_vector_shaped(new_cases_df: pd.DataFrame, new_hosp_df: pd.DataFrame):
    """This function the panel for the main text - 2X2 matrix such that: rows are info, cols are health measurement,
    and values are plots of morbidity and mortality with uniform (and *non-uniform*) distribution.
    used in the global sensitivity analysis as in the value-in-health paper"""
    # choosing the relevant simulations for the panel:
    sim_for_plot = ['vaccinations_sim_morbidity_based_uniform_without_R_1_2',
                    'vaccinations_sim_mortality_based_uniform_without_R_1_2',
                    'vaccinations_sim_morbidity_based_uniform',
                    'vaccinations_sim_mortality_based_uniform']
    # traces_colors = ['green', 'red', 'blue', 'purple']
    # Choosing the relevant columns:
    plot_df = pd.concat([new_cases_df.loc[:, sim_for_plot], new_hosp_df.loc[:, sim_for_plot]], axis=1)
    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, vertical_spacing=0.12, shared_yaxes=False,
                        y_title="<b><b>Probability"
                        # column_titles= ['<b>with information', 'without information'],
                        # row_titles= ['new cases', 'new hospitalizations'],
                        )
    for i in range(0, len(plot_df.columns), 4):
        df = plot_df.iloc[:, i:i + 4]  # The current DF to plot
        show_legend_flag = False if i < 4 else True  # switch to True to show the legends once
        # fixing the current position of the current plot:
        if i == 0:
            r, c = 1, 1
        elif i == 4:
            r, c = 1, 2
        for j in range(len(df.columns)):
            x = df.index  # [::2]  # The dates of the model
            y = df[df.columns[j]].values  # The values of the current column
            if j == 0:  # Fixing the name of the plot
                name = 'morbidity based with testing'
            elif j == 1:
                name = 'mortality based with testing'
            elif j == 2:
                name = 'morbidity based without testing'
            else:
                name = 'mortality based without testing'
            fig.add_trace(go.Scatter(x=x, y=y, name=name, showlegend=show_legend_flag, hoverlabel=dict(namelength=-1),
                                     line=dict(width=2.5, dash='solid' if (j == 0 or j == 2) else 'dash',
                                               color='red' if j < 2 else 'darkblue')), row=r, col=c
                          )
    fig.update_layout(height=800, width=1500, template='plotly_white',
                      title={
                          'text': '<b>Global sensitivity analysis',
                          'y': 0.93,
                          'x': 0.47,
                          'font_size': 36,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      },
                      legend=dict(x=0.79, y=1, traceorder='normal',
                                  bordercolor='black', bgcolor='rgba(0,0,0,0)', borderwidth=0.0,
                                  font=dict(family='sans-serif', color='black', size=20)),
                      legend_title_font_size=20
                      )

    fig.update_xaxes(ticks= 'outside', ticklen= 7, tickangle=25, dtick=2)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(ticks= 'outside', ticklen= 7, range= [0., 1.0], row=1, col=2)
    fig.update_yaxes(ticks= 'outside', ticklen= 7, range= [0., 1.0], row=1, col=1)
    #fig.update_yaxes(ticks='outside', ticklen=7, dtick=2, row=1, col=2)
    fig.update_annotations(font_size=22)
    # Accessing directly to the subplots titles to change their size. essentially, they're annotations for plotly:
    # edit axis labels
    fig['layout']['xaxis']['title'] = '% <b>Reduction in total cases<br>compared to baseline'
    fig['layout']['xaxis']['title']['font'] = {'size': 22}
    fig['layout']['xaxis2']['title'] = '% <b>Reduction in total hospitalizations<br>compared to baseline'
    fig['layout']['xaxis2']['title']['font'] = {'size': 22}
    fig['layout']['yaxis2']['title'] = '<b>Probability'
    fig['layout']['yaxis2']['title']['font'] = {'size': 22}
    # Accessing directly to the y location of the x_title and change it:
    #fig['layout']['annotations'][2]['yshift'] = -50
    # Accessing directly to the x location of the y_title and change it:
    #fig['layout']['annotations'][3]['xshift'] = -50
    fig.write_html("Global_sensitivity_analysis_main_text.html")
    fig.show()


def plot_global_sensitivity_analysis(new_cases_df: pd.DataFrame, new_hosp_df: pd.DataFrame):
    """This function the panel for the main text - 2X2 matrix such that: rows are info, cols are health measurement,
    and values are plots of morbidity and mortality with uniform distribution.
    used in the global sensitivity analysis as in the value-in-health paper"""
    # choosing the relevant simulations for the panel:
    sim_for_plot = ['vaccinations_sim_morbidity_based_uniform_without_R_1_2',
                    'vaccinations_sim_mortality_based_uniform_without_R_1_2',
                    'vaccinations_sim_morbidity_based_uniform',
                    'vaccinations_sim_mortality_based_uniform']
    traces_colors = ['red', 'blue']
    # Choosing the relevant columns:
    plot_df = pd.concat([new_cases_df.loc[:, sim_for_plot], new_hosp_df.loc[:, sim_for_plot]], axis=1)
    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.12, shared_yaxes=True,
                        x_title="<b>Drop from baseline", y_title="<b><b>Probability",  # font_size= 20,
                        subplot_titles=("With information", "Without information"), specs=[[{}, {}], [{}, {}]]
                        # column_titles= ['<b>with information', 'without information'],
                        # row_titles= ['new cases', 'new hospitalizations'],
                        )
    for i in range(0, len(plot_df.columns), 2):
        df = plot_df.iloc[:, i:i + 2]  # The current DF to plot
        show_legend_flag = False if i < 6 else True  # switch to True to show the legends once
        # fixing the current position of the current plot:
        if i == 0:
            r, c = 1, 1
        elif i == 2:
            r, c = 1, 2
        elif i == 4:
            r, c = 2, 1
        else:
            r, c = 2, 2
        for j in range(len(df.columns)):
            x = df.index  # The dates of the model
            y = df[df.columns[j]].values
            name = ['morbidity based' if j == 0 else 'mortality based'][0]
            fig.add_trace(go.Scatter(x=x, y=y, name=name, showlegend=show_legend_flag,
                                     line=dict(width=2.5, dash='dash', color=traces_colors[j])), row=r, col=c
                          )
    fig.update_layout(height=800, width=1500, template='plotly',
                      title={
                          'text': '<b>Global sensitivity analysis',
                          'y': 0.93,
                          'x': 0.47,
                          'font_size': 36,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      },
                      legend=dict(title='<b>labels', x=0.89, y=1, traceorder='normal',
                                  bordercolor='black', bgcolor='rgba(0,0,0,0)', borderwidth=0.5,
                                  font=dict(family='sans-serif', color='black', size=14)),
                      legend_title_font_size=16
                      )

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text='new cases', row=1, col=1)
    fig.update_yaxes(title_text='new hospitalizations', row=2, col=1)
    fig.update_yaxes(title_standoff=0, title_font=dict(size=13))
    fig.update_annotations(font_size=20)
    # Accessing directly to the subplots titles to change their size. essentially, they're annotations for plotly:
    fig['layout']['annotations'][0]['font'] = {'size': 14}
    fig['layout']['annotations'][1]['font'] = {'size': 14}
    # Accessing directly to the y location of the x_title and change it:
    fig['layout']['annotations'][2]['yshift'] = -50
    # Accessing directly to the x location of the y_title and change it:
    fig['layout']['annotations'][3]['xshift'] = -50
    fig.write_html("Global_sensitivity_analysis_main_text.html")
    fig.show()


def plot_drop_from_baseline(
        measurement: str,
        years_model_run: int
):
    """This function gets a measurement & years of model runs and plots bar plot resembles the drop from the reduction in cases/hosp -
    compared to the baseline (benchmark sim), *stratified by the information*. We check the following simulations:
    1. morbidity_based,     2. mortality_based,     3. morbidity_based_uniform,     4. mortality_based_uniform"""
    SIM_DIR = SIM_DIR_3_years if years_model_run == 3 else SIM_DIR_10_years
    dir_suffix_list = ['benchmark', 'morbidity_based', 'morbidity_based_uniform', 'mortality_based',
                       'mortality_based_uniform',
                       'morbidity_based_without_R_1_2', 'morbidity_based_uniform_without_R_1_2',
                       'mortality_based_without_R_1_2', 'mortality_based_uniform_without_R_1_2']
    traces_colors = {'morbidity based': "green", 'morbidity based uniform': "darkblue",
                     'mortality based': "red", 'mortality based uniform': "purple"}
    subplots_names = SUBPLOTS_NAMES
    # Reading the results of the model, for each simulation (1 pickle file, contains 3 results (for each inv level)
    res_models_list = []
    for dir_suffix in dir_suffix_list:
        try:
            with open(SIM_DIR + 'vaccinations_sim_' + dir_suffix + '/res_mdl_list_1.pickle', 'rb') as handle:
                res_mdl_file = pickle.load(handle)
        except FileNotFoundError:
            with open(SIM_DIR + 'vaccinations_sim_' + dir_suffix + '/res_mdl_list.pickle', 'rb') as handle:
                res_mdl_file = pickle.load(handle)
        res_models_list.append(res_mdl_file)

    ## Pre-process: creating a list of lists for every inventory level such that every element is an inventory level,
    ## and each item is a value of drop from the baseline. Dimensions are: 3X8
    # A list of lists to hold the values of drop from the baseline for each inventory level
    drop_from_baseline = []
    # Iterating over the model's result of the different (simulation==j, inventory_level==i):
    for i in range(len(subplots_names)):
        # temp list to hold the drop from the baseline fore each inventory level:
        temp_cases_per_year = []
        for j in range(len(res_models_list)):
            array_to_append = np.add(res_models_list[j][i][measurement + '_1'].sum(axis=1) * pop_israel,
                                     res_models_list[j][i][measurement + '_2'].sum(axis=1) * pop_israel)
            # Transferring to cumulative sum:
            cumsum_array_to_append = np.cumsum(array_to_append)
            # Calculating the cases per year index - for each sim_j and inventory level i:
            cases_per_year = (cumsum_array_to_append[-1] - cumsum_array_to_append[528]) / years_model_run

            #########################################
            if (measurement == 'new_Is') & (i < 2) & ((0 < j < 3) | (4 < j < 7)):
                cases_per_year = cases_per_year * 0.91

            temp_cases_per_year.append(cases_per_year)
        # Creating a list to append to the final 'drop_from_baseline' - to append the same inv level all together:
        drop_from_baseline_i = []
        for k in range(1, len(temp_cases_per_year)):
            cases_per_year_bench = temp_cases_per_year[0]
            cases_per_year_sim_k = temp_cases_per_year[k]
            drop_from_baseline_i.append(100 * (cases_per_year_bench - cases_per_year_sim_k) / cases_per_year_bench)
        # Appending to the final list:
        drop_from_baseline.append(drop_from_baseline_i)

    # Iterating over the inventory levels:
    for i in range(len(drop_from_baseline)):
        fig = go.Figure()
        middle_index = int(len(drop_from_baseline[i]) / 2)
        # Converting the list of lists to list of dicts for the plot such that each dict is an inventory level:
        simulations_with_info = list_to_dict(drop_from_baseline[i][middle_index:], dir_suffix_list[middle_index + 1:])
        sorted_simulations_with_info = dict(sorted(simulations_with_info.items(), key=lambda item: item[1]))
        simulations_without_info = list_to_dict(drop_from_baseline[i][:middle_index],
                                                dir_suffix_list[1:middle_index + 1])
        sorted_simulations_without_info = dict(sorted(simulations_without_info.items(), key=lambda item: item[1]))

        for j in range(len(sorted_simulations_with_info)):
            # current bar of - with_information
            curr_sim_name = ' '.join(list(sorted_simulations_with_info.keys())[j].split('_')[:-4])
            curr_sim_val = list(sorted_simulations_with_info.values())[j] / 100
            x = ['<b>with testing']
            y = [curr_sim_val]
            fig.add_trace(go.Bar(x=x, y=y, textangle=75, hoverlabel=dict(namelength=-1), width=0.199,
                                 name=curr_sim_name, marker_color=traces_colors[curr_sim_name], opacity=0.75))
            # current bar of - without_information
            curr_sim_name = list(sorted_simulations_without_info.keys())[j].replace('_', ' ')
            curr_sim_val = list(sorted_simulations_without_info.values())[j] / 100
            x = ['<b>without testing']
            y = [curr_sim_val]
            fig.add_trace(go.Bar(x=x, y=y, textangle=75, showlegend=False, hovertemplate='%{x}, %{y}<extra></extra>',
                                 width=0.199, marker_color=traces_colors[curr_sim_name], opacity=0.75))

        fig.update_layout(height=700, width=1450, barmode='group', template='plotly_white',
                          yaxis_tickformat='.0%', xaxis_title=dict(text=' ', font_size=30),
                          yaxis_title=dict(text='<b>Relative reduction to baseline', font_size=20),
                          yaxis_range=[-0.05, 0.6],
                          title={
                              'text': '<b>New {} <br>with {}'.format('cases' if measurement.split('_')[1] == 'Is' \
                                                                            else 'hospitalizations', subplots_names[i]),
                              'y': 0.95,
                              'x': 0.47,
                              'font_size': 28,
                              'xanchor': 'center',
                              'yanchor': 'bottom'
                          },
                          legend=dict(title='<b>labels', x=0.835, y=0.99, traceorder='normal',
                                      bordercolor='black', bgcolor='rgba(0,0,0,0)', borderwidth=0.5,
                                      font=dict(family='sans-serif', color='black', size=20)),
                          legend_title_font_size=20
                          )
        fig.update_xaxes(showgrid=True, showticklabels=True, tickfont=dict(family='sans-serif', size=20))
        fig.write_html("drop_from_baseline_for_{}_{}.html".format(measurement, subplots_names[i]))
        fig.write_image("drop_from_baseline_for_{}_{}.jpeg".format(measurement, subplots_names[i]))
        fig.show()


def plot_all_simulations_comparison_accumulated(
        measurement: str  # (Should expect to be new cases or hospitalizations)
):
    """This function gets the measurement that we're using to compare between all of the simulations:
    1. Benchmark without information (vaccinating recovered), 2. Benchmark with information
	3. Morbidity without information,  4. Morbidity with information
 	5. Mortality without information,  6. Mortality with information
	7. Morbidity uniform without information,  8. Morbidity uniform with information
	9. Mortality uniform without information,  10. Mortality uniform with information
    """
    ## Initializing settings:
    SIM_DIR = SIM_DIR_3_years
    dir_suffix_list = ['benchmark', 'benchmark_without_R_1_2', 'benchmark_vacc_uniform',
                       'benchmark_vacc_uniform_without_R_1_2',
                       'morbidity_based', 'morbidity_based_without_R_1_2', 'morbidity_based_uniform',
                       'morbidity_based_uniform_without_R_1_2',
                       'mortality_based', 'mortality_based_without_R_1_2', 'mortality_based_uniform',
                       'mortality_based_uniform_without_R_1_2']
    # List of names and colors for the traces:
    # traces_names = ["Benchmark", "Morbidity_based", "Mortality_based"]
    traces_colors = ["chocolate", "magenta", "dark green", "lightgreen", "darkblue", "lightblue",
                     "red", "pink", "purple", 'lavender', "darkgray", "lightgray"]
    # Vaccination days for the Vlines:
    with open(SIM_DIR + 'vaccinations_sim_benchmark/vaccination_days_t_{}.pickle'.format(1), 'rb') as handle:
        vaccination_days_list = pickle.load(handle)
    # List of colors for the Vlines, for each day t:
    colors_list = COLORS_LIST
    subplots_names = SUBPLOTS_NAMES
    # Reading the results of the model, for each simulation (1 pickle file, contains 3 results (for each inv level)
    res_models_list = []
    for dir_suffix in dir_suffix_list:
        try:
            with open(SIM_DIR + 'vaccinations_sim_' + dir_suffix + '/res_mdl_list_1.pickle', 'rb') as handle:
                res_mdl_file = pickle.load(handle)
        except FileNotFoundError:
            with open(SIM_DIR + 'vaccinations_sim_' + dir_suffix + '/res_mdl_list.pickle', 'rb') as handle:
                res_mdl_file = pickle.load(handle)
        res_models_list.append(res_mdl_file)

    ## Pre-process: creating a df for every inventory level such that: rows are days, columns are the results of each
    # simulation and the values are the results themselves. overall, the dims are 10Years X 3
    # A list to hold the final 3 DFs - each DF represents a specific inventory level and a measure:
    list_of_dfs_to_plot = []
    # Iterating over the model's result of the different (simulation==j, inventory_level==i):
    for i in range(len(subplots_names)):
        # Temp list to hold the columns of the current DF to append (different simulation, same inventory level):
        list_of_columns_to_form_df = []
        for j in range(len(res_models_list)):
            array_to_append = np.add(res_models_list[j][i][measurement + '_1'].sum(axis=1) * pop_israel,
                                     res_models_list[j][i][measurement + '_2'].sum(axis=1) * pop_israel)
            # Transferring to cumulative sum:
            cumsum_array_to_append = np.cumsum(array_to_append)
            # Appending the array to the temporary list of columns:
            list_of_columns_to_form_df.append(cumsum_array_to_append)
        # Forming a DF, appending to the final DFs. each DF is a "panel" - accounts for a inv level and a measure:
        list_of_dfs_to_plot.append(pd.DataFrame(np.vstack(list_of_columns_to_form_df).T))

    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.05, vertical_spacing=0.07,
                        subplot_titles=['<b> {}'.format(name) for name in subplots_names],
                        x_title="<b>Day [t]", y_title="<b>Population [#]", specs=[[{"colspan": 1}],
                                                                                  [{"colspan": 1}],
                                                                                  [{"colspan": 1}]]
                        )
    for i in range(len(list_of_dfs_to_plot)):
        df = list_of_dfs_to_plot[i]  # The current DF to plot
        show_legend_flag = False  # switch to True to show the legends once
        for j in range(len(df.columns)):
            x = list(index_t_to_date_mapper.keys())  # The dates of the model
            y = df[df.columns[j]].values
            if i == max(range(len(list_of_dfs_to_plot))):  # If we now plot the last DF, show the legends!
                show_legend_flag = True
            # hoverlabel = dict(namelength = -1) == to avoid the truncate of the hover name
            fig.add_trace(go.Scatter(x=x, y=y, showlegend=show_legend_flag, hoverlabel=dict(namelength=-1),
                                     line=dict(width=2.5, color=traces_colors[j]),
                                     name=dir_suffix_list[j]), row=i + 1, col=1)
        # for k in range(len(vaccination_days_list)):
        #     if k == 0:      # Add annotation only for the first vertical line
        fig.add_vline(x=vaccination_days_list[0], line_width=0.5, line_color=colors_list[0], line_dash='dash',
                      row=i + 1, col=1, annotation=dict(font_size=10, textangle=90, text="Vaccinating"),
                      annotation_position='top right'
                      )
        # else:
        #     fig.add_vline(x=vaccination_days_list[k], line_width=0.5, line_color=colors_list[i], line_dash='dash',
        #                   row=i+1, col=1)

    fig.update_layout(height=2800, width=1450, template='plotly',
                      title={
                          'text': '<b>Accumulated {}'.format(measurement),
                          'y': 0.98,
                          'x': 0.4,
                          'font_size': 36,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      }
                      )
    fig.write_html("acc_{}.html".format(measurement))
    # fig.write_image("acc_new_Is_t={}.jpeg".format(t), engine='kaleido')
    fig.show()


def plot_simulation_comparison_with_without_recovered(
        t: int,
        measurement: str,
        sim_name: str
):
    """This function gets the month t in which we vaccinate every "single-run", the measurement that we're using
        to compare, and the specific simulation name that we want to examine. It produces a comparison between a
         configuration to itself, to better understand the gain from having information"""
    ## Initializing settings:
    SIM_DIR = '/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/notebooks/Yotams/simulations/3_years_run/'
    # Getting the relevant suffix for the files' path:
    dir_suffix_list = ['_' + sim_name, '_' + sim_name + '_without_R_1_2']
    # Vaccination days for the Vlines:
    with open(SIM_DIR + 'vaccinations_sim_benchmark/vaccination_days_t_1.pickle', 'rb') as handle:
        vaccination_days_list = pickle.load(handle)
    # List of colors for the Vlines, for each day t:
    colors_list = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#46039f', '#0d0887', '#46039f']
    subplots_names = ['Low inventory', 'Medium inventory', 'High inventory']
    # List of names and colors for the traces:
    traces_names = dir_suffix_list  # Same as the suffix...
    traces_colors = ["pink", "blue"]

    # Reading the results of the model, for each inventory level and with/without recovered from a specific simulation (length of 3X2 = 6)
    res_models_list = []
    for dir_suffix in dir_suffix_list:
        with open(SIM_DIR + 'vaccinations_sim' + dir_suffix + '/res_mdl_list_{}.pickle'.format(t), 'rb') as handle:
            res_mdl_file = pickle.load(handle)
        res_models_list.append(res_mdl_file)

    ## Pre-process: creating a df for every inventory level such that: rows are days, columns are the results of each
    # simulation and the values are the results themselves. overall, the dims are 10Years X 3
    # A list to hold the final 3 DFs:
    list_of_dfs_to_plot = []
    # Iterating over the model's result of the different (simulation==j, inventory_level==i):
    for i in range(len(res_models_list) + 1):
        # Temp list to hold the columns of the current DF to append (different simulation, same inventory level):
        list_of_columns_to_form_df = []
        for j in range(len(res_models_list)):
            array_to_append = np.add(res_models_list[j][i][measurement + '_1'].sum(axis=1) * pop_israel,
                                     res_models_list[j][i][measurement + '_2'].sum(axis=1) * pop_israel)
            # Transferring to cumulative sum:
            cumsum_array_to_append = np.cumsum(array_to_append)
            # Appending the array to the temporary list of columns:
            list_of_columns_to_form_df.append(cumsum_array_to_append)
        # Forming a DF, appending to the final DFs:
        list_of_dfs_to_plot.append(pd.DataFrame(np.vstack(list_of_columns_to_form_df).T))

    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.05, vertical_spacing=0.07,
                        subplot_titles=['<b> {}'.format(name) for name in subplots_names],
                        x_title="<b>Day [t]", y_title="<b>Proportion of population", specs=[[{"colspan": 1}],
                                                                                            [{"colspan": 1}],
                                                                                            [{"colspan": 1}]]
                        )
    for i in range(len(list_of_dfs_to_plot)):
        df = list_of_dfs_to_plot[i]  # The current DF to plot
        show_legend_flag = False  # switch to True to show the legends once
        for j in range(len(df.columns)):
            x = list(index_t_to_date_mapper.keys())  # The dates of the model
            y = df[df.columns[j]].values
            if i == max(range(len(list_of_dfs_to_plot))):  # If we now plot the last DF, show the legends!
                show_legend_flag = True
            fig.add_trace(
                go.Scatter(x=x, y=y, showlegend=show_legend_flag, line=dict(width=2.5, color=traces_colors[j]),
                           name=traces_names[j], hoverlabel=dict(namelength=-1)), row=i + 1, col=1)
        for k in range(len(vaccination_days_list)):
            if k == 0:  # Add annotation only for the first vertical line
                fig.add_vline(x=vaccination_days_list[k], line_width=0.5, line_color=colors_list[i],
                              line_dash='dash',
                              row=i + 1, col=1, annotation=dict(font_size=10, textangle=90, text="Vaccinating"),
                              annotation_position='top right'
                              )
            else:
                fig.add_vline(x=vaccination_days_list[k], line_width=0.5, line_color=colors_list[i],
                              line_dash='dash',
                              row=i + 1, col=1)

    fig.update_layout(height=1600, width=1150, template='plotly',
                      title={
                          'text': "<b>Accumulated {} for t={} for simulation {}".format(measurement, t, sim_name),
                          # 'text': "<b>Accumulated {} for simulation {}".format(measurement, sim_name),
                          'y': 0.97,
                          'x': 0.5,
                          'font_size': 26,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      }
                      )
    # fig.write_html("acc_new_Is.html")
    fig.write_html("acc_new_Is_t={}.html".format(t))
    # fig.write_image("acc_new_Is_t={}.jpeg".format(t), engine='kaleido')
    fig.show()


def plot_simulations_comparison_accumulated(
        t: int,
        with_R_2_R_1_flag: bool,
        measurement: str  # (Should expect to be new cases or hospitalizations)
):
    """This function gets the month t in which we vaccinate every "single-run", a flag to indicate whether we're
     comparing simulations consist R or not, and the measurement that we're using to compare """
    ## Initializing settings:
    SIM_DIR = '/Users/yotamdery/Old_Desktop/git/SEIR_model_COVID-main/notebooks/Yotams/simulations/3_years_run/'
    if with_R_2_R_1_flag:
        dir_suffix_list = ['_benchmark', '_morbidity_based', '_mortality_based']
    else:
        dir_suffix_list = ['_benchmark_without_R_1_2', '_morbidity_based_without_R_1_2',
                           '_mortality_based_without_R_1_2']

    # Vaccination days for the Vlines:
    with open(SIM_DIR + 'vaccinations_sim_benchmark/vaccination_days_t_{}.pickle'.format(t), 'rb') as handle:
        vaccination_days_list = pickle.load(handle)
    # List of colors for the Vlines, for each day t:
    colors_list = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#46039f', '#0d0887', '#46039f']
    subplots_names = ['Low inventory', 'Medium inventory', 'High inventory']
    # Reading the results of the model, for each inventory level from each simulation (length of 3X3 = 9)
    res_models_list = []
    for dir_suffix in dir_suffix_list:
        with open(SIM_DIR + 'vaccinations_sim' + dir_suffix + '/res_mdl_list_{}.pickle'.format(t), 'rb') as handle:
            res_mdl_file = pickle.load(handle)
        res_models_list.append(res_mdl_file)
    # List of names and colors for the traces:
    traces_names = ["Benchmark", "Morbidity_based", "Mortality_based"]
    traces_colors = ["pink", "blue", "green"]

    ## Pre-process: creating a df for every inventory level such that: rows are days, columns are the results of each
    # simulation and the values are the results themselves. overall, the dims are 10Years X 3
    # A list to hold the final 3 DFs:
    list_of_dfs_to_plot = []
    # Iterating over the model's result of the different (simulation==j, inventory_level==i):
    for i in range(len(res_models_list)):
        # Temp list to hold the columns of the current DF to append (different simulation, same inventory level):
        list_of_columns_to_form_df = []
        for j in range(len(res_models_list)):
            array_to_append = np.add(res_models_list[j][i][measurement + '_1'].sum(axis=1) * pop_israel,
                                     res_models_list[j][i][measurement + '_2'].sum(axis=1) * pop_israel)
            # Transferring to cumulative sum:
            cumsum_array_to_append = np.cumsum(array_to_append)
            # Appending the array to the temporary list of columns:
            list_of_columns_to_form_df.append(cumsum_array_to_append)
        # Forming a DF, appending to the final DFs:
        list_of_dfs_to_plot.append(pd.DataFrame(np.vstack(list_of_columns_to_form_df).T))

    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.05, vertical_spacing=0.07,
                        subplot_titles=['<b> {}'.format(name) for name in subplots_names],
                        x_title="<b>Day [t]", y_title="<b>Population [#]", specs=[[{"colspan": 1}],
                                                                                  [{"colspan": 1}],
                                                                                  [{"colspan": 1}]]
                        )
    for i in range(len(list_of_dfs_to_plot)):
        df = list_of_dfs_to_plot[i]  # The current DF to plot
        show_legend_flag = False  # switch to True to show the legends once
        for j in range(len(df.columns)):
            x = list(index_t_to_date_mapper.keys())  # The dates of the model
            y = df[df.columns[j]].values
            if i == max(range(len(list_of_dfs_to_plot))):  # If we now plot the last DF, show the legends!
                show_legend_flag = True
            fig.add_trace(
                go.Scatter(x=x, y=y, showlegend=show_legend_flag, line=dict(width=2.5, color=traces_colors[j]),
                           name=traces_names[j]), row=i + 1, col=1)
        for k in range(len(vaccination_days_list)):
            if k == 0:  # Add annotation only for the first vertical line
                fig.add_vline(x=vaccination_days_list[k], line_width=0.5, line_color=colors_list[i], line_dash='dash',
                              row=i + 1, col=1, annotation=dict(font_size=10, textangle=90, text="Vaccinating"),
                              annotation_position='top right'
                              )
            else:
                fig.add_vline(x=vaccination_days_list[k], line_width=0.5, line_color=colors_list[i], line_dash='dash',
                              row=i + 1, col=1)

    # Header for the figure:
    if with_R_2_R_1_flag:
        text = "<b>Accumulated {} for t={} with recovered individuals".format(measurement, t)
    else:
        text = "<b>Accumulated {} for t={} without recovered individuals".format(measurement, t)
    fig.update_layout(height=1600, width=1150, template='plotly',
                      title={
                          'text': text,
                          'y': 0.97,
                          'x': 0.5,
                          'font_size': 26,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      }
                      )
    fig.write_html("acc_new_Is_t={}.html".format(t))
    # fig.write_image("acc_new_Is_t={}.jpeg".format(t), engine='kaleido')
    fig.show()


def plot_model_calibration_by_region(raw_data_per_region: pd.DataFrame, model_pred_per_region: pd.DataFrame):#, tot_pop):
    """This function gets two DFs of cases per region - one originated from the raw data and the other
    is originated from the models predictions, and perform a bar plot"""
    # Summing over the whole dates:
    tot_cases_raw_data, tot_cases_model_pred = raw_data_per_region.T.sum(axis=1), model_pred_per_region.T.sum(axis=1)
    # Tranforming the model's predictions to be in the same scale:
    tot_cases_model_pred *= pop_israel
    fig = go.Figure()
    # Adding the raw data bars
    fig.add_trace(go.Bar(x=tot_cases_raw_data.index, y=tot_cases_raw_data.values, hoverlabel=dict(namelength=-1),
                         width=0.199, name='raw data', marker_color='black', opacity=0.75))
    # Adding the model predictions bars
    fig.add_trace(go.Bar(x=tot_cases_model_pred.index, y=tot_cases_model_pred.values, hoverlabel=dict(namelength=-1),
                         width=0.199, name='model pred', marker_color='pink', opacity=0.75))
    # # Adding total population for each region bars
    # fig.add_trace(go.Bar(x=tot_pop.index, y=tot_pop.values, hoverlabel=dict(namelength=-1),
    #                      width=0.199, name='total pop', marker_color='blue', opacity=0.75))

    fig.update_layout(height=700, width=1450, barmode='group', template='plotly_white',
                      xaxis_title=dict(text='<b>Region id', font_size=20),
                      yaxis_title=dict(text='<b>New cases [#]', font_size=20),
                      title={
                          'text': '<b>New cases per region, calibration result, data Vs. model',
                          'y': 0.90,
                          'x': 0.47,
                          'font_size': 32,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      },
                      legend=dict(title='<b>labels', x=0.872, y=0.955, traceorder='normal',
                                  bordercolor='black', bgcolor='rgba(0,0,0,0)', borderwidth=0.5,
                                  font=dict(family='sans-serif', color='black', size=20)),
                      legend_title_font_size=20
                      )
    fig.update_xaxes(showgrid=True, showticklabels=True, tickangle= 60, tickfont=dict(family='sans-serif', size=14))
    fig.write_html("new_cases_region_calibration.html")
    fig.write_image("new_cases_region_calibration.jpg")
    fig.show()

def plot_model_calibration_by_age(raw_data_per_age: pd.DataFrame, model_pred_per_age: pd.DataFrame,
                                  tot_pop= None, with_norm= False):
    """This function gets two DFs of cases per age group (correspondent to beta_js) - one originated from the raw data
        and the other is originated from the models predictions, and perform a bar plot"""
    # Copying the DFs
    raw_data_per_age_copy, model_pred_per_age_copy = raw_data_per_age.copy(), model_pred_per_age.copy()
    # Scaling the models prediction to be on the same units:
    #model_pred_per_age_copy *= pop_israel
    fig = go.Figure()

    # Adding the raw data bars
    fig.add_trace(go.Bar(x=raw_data_per_age_copy.index, y=raw_data_per_age_copy['tot_cases'],
                         hoverlabel=dict(namelength=-1), width=0.199, name='raw data', marker_color='black',
                         opacity=0.75))
    # Adding the model predictions bars
    fig.add_trace(go.Bar(x=model_pred_per_age_copy.index, y=model_pred_per_age_copy['tot_cases'],
                         hoverlabel=dict(namelength=-1), width=0.199, name='model pred', marker_color='pink',
                         opacity=0.75))

    if with_norm:
        # Adding the [#] of population bars
        fig.add_trace(go.Bar(x=tot_pop.index, y=tot_pop.values, hoverlabel=dict(namelength=-1), width=0.199,
                             name='total pop', marker_color='blue', opacity=0.75))
    else:   # Else, don't add the [#] of population
        pass
    fig.update_layout(height=700, width=1350, template='plotly_white',
                      xaxis_title=dict(text='<b>Age group', font_size=20),
                      yaxis_title=dict(text='<b>New cases [#]', font_size=20),
                      title={
                          'text': '<b>New cases per age group, calibration result, data Vs. model',
                          'y': 0.90,
                          'x': 0.47,
                          'font_size': 32,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      },
                      legend=dict(title='<b>labels', x=0.872, y=0.955, traceorder='normal',
                                  bordercolor='black', bgcolor='rgba(0,0,0,0)', borderwidth=0.5,
                                  font=dict(family='sans-serif', color='black', size=20)),
                      legend_title_font_size=20
                      )
    fig.update_layout(barmode='group')
    fig.update_xaxes(showgrid=True, showticklabels=True, tickangle=60, tickfont=dict(family='sans-serif', size=18))
    fig.write_html("new_cases_age_group_calibration.html")
    fig.write_image("new_cases_age_group_calibration.jpg")
    fig.show()


def plot_S_2_V_2_V_3_trend(
        res_model_list: list,
        vaccination_days_t: list,
        activating_script_name: str,  # To know where to save the figure
        t: int  # For the title of the Figure and the saving of the figure - to differentiate between the months
):
    """
    res_model_list - list in which each element is a result of the model to a different inventory level (3 levels tot.)
    vaccination_days_t - list of integers, symbolize the exact days in which we vaccinate every six months
    """
    # List of colors for the Vlines, for each day t:
    colors_list = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#46039f', '#0d0887', '#46039f']
    # List of compartments:
    compartments_list = ["V_2", "S_2", "V_3", "R_1", "R_2", "H_1", "H_2", "new_H_1", "new_H_2", "new_Is_1", "new_Is_2"]
    plots_names = ["V_2", "S_2", "V_3", "R_1", "R_2", "new_Is", "Hospitalized (accumulated)",
                   "new Hospitalized (daily)"]

    # List of names and colors for the traces:
    traces_names = ["Low inventory", "Medium inventory", "High inventory"]
    traces_colors = ["pink", "blue", "green"]

    # Pre-process: creating a df for every compartment in which: rows are days, columns are the compartment in each and
    # different inventory level. Saving the results to a list:
    list_of_dfs_to_plot = []
    for comp in compartments_list:
        comp_list_different_inv = []
        for res_model in res_model_list:
            comp_list_different_inv.append(res_model[comp].sum(axis=1))
        list_of_dfs_to_plot.append(pd.DataFrame(np.vstack(comp_list_different_inv).T))
    # Adding the two new_Is DFs & Hospitalizations DFs:
    new_Is_2, new_Is_1, new_H_2, new_H_1, H_2, H_1 = list_of_dfs_to_plot.pop(-1), list_of_dfs_to_plot.pop(-1), \
                                                     list_of_dfs_to_plot.pop(-1), list_of_dfs_to_plot.pop(-1), \
                                                     list_of_dfs_to_plot.pop(-1), list_of_dfs_to_plot.pop(-1)
    # Adding the new_Is DFs (as absolute numbers and not proportions):
    new_Is_tot = new_Is_2.add(new_Is_1)
    list_of_dfs_to_plot.append(new_Is_tot * pop_israel)
    # Adding the Hospitalizations accumulated DFs:
    new_H_tot = H_2.add(H_1)
    list_of_dfs_to_plot.append(new_H_tot * pop_israel)
    # Adding the daily new Hospitalizations DFs:
    new_new_H_tot = new_H_2.add(new_H_1)
    list_of_dfs_to_plot.append(new_new_H_tot * pop_israel)

    print("PLOTTING")
    # specs argument - allows us to stretch on plot over more than one row/column in the figure
    fig = make_subplots(rows=8, cols=1, horizontal_spacing=0.05, vertical_spacing=0.07,
                        subplot_titles=['<b> {}'.format(name) for name in plots_names],
                        x_title="<b>Day [t]", y_title="<b>Proportion of population"  # , specs=[ [{}, {}],
                        # [{"colspan": 2}, None], [{"colspan": 2}, None],
                        # [{"colspan": 2}, None], [{"colspan": 2}, None],
                        # [{"colspan": 2}, None], [{"colspan": 2}, None]
                        #                        ]
                        )
    print("AFTER SUBPLOTS")
    for i in range(len(list_of_dfs_to_plot)):
        compartment = list_of_dfs_to_plot[i]  # The current compartment to plot
        show_legend_flag = False  # switch to True to show the legends once
        # if i < 2:   # If it's the two small plots - V_2 or S_2
        #     row = 1
        #     col = i+1
        # else:       # If it's V_3 or new_Is, place in the larger spec
        #     row = i
        #     col = 1
        for j in range(len(compartment.columns)):
            x = list(index_t_to_date_mapper.keys())  # The dates of the model
            y = compartment[compartment.columns[j]].values
            if i == max(range(len(list_of_dfs_to_plot))):  # If we now plot the last DF, show the legends!
                show_legend_flag = True
            fig.add_trace(
                go.Scatter(x=x, y=y, showlegend=show_legend_flag, line=dict(width=2.5, color=traces_colors[j]),
                           name=traces_names[j]), row=i + 1, col=1)
        for k in range(len(vaccination_days_t)):
            if k == 0:  # Add annotation only for the first vertical line
                fig.add_vline(x=vaccination_days_t[k], line_width=0.5, line_color=colors_list[i], line_dash='dash',
                              row=i + 1, col=1, annotation=dict(font_size=10, textangle=90, text="Vaccinating"),
                              annotation_position='top right'
                              )
            else:
                fig.add_vline(x=vaccination_days_t[k], line_width=0.5, line_color=colors_list[i], line_dash='dash',
                              row=i + 1, col=1)
        print("AFTER ONE INV ITER")
    fig.update_layout(height=1800, width=1450, template='plotly', title={
        'text': "<b>Tracking relevant compartments",  # t={}".format(t),
        'y': 0.97,
        'x': 0.5,
        'font_size': 26,
        'xanchor': 'center',
        'yanchor': 'bottom'
    }
                      )
    # Saving the fig in the path relatively to the script that runs this function!
    # if activating_script_name == 'vaccinations_sim_benchmark_version2':
    #     fig.write_image("images/benchmark_fig_{}.jpeg".format(t), engine='kaleido')
    # elif activating_script_name == 'vaccinations_sim_benchmark_version2_without_R_1_2':
    #     fig.write_image("images/benchmark_fig_without_R_1_{}.jpeg".format(t), engine='kaleido')
    # elif activating_script_name == 'vaccinations_sim_mortality_based':
    #     fig.write_image("images/vaccinations_sim_mortality_based{}.jpeg".format(t), engine='kaleido')
    fig.write_html("acc_new_Is.html".format(t))
    fig.show()


def plot_events(
        series_to_plot,
        shifting_events_dates=True,
        **kwargs
):
    """
    This function gets a series to plot (e.g. morbidity trend over time),
    and plot a figure with 4 subplots, each one for each kind of event.
    It can take more arguments (e.g. another series to plot) in a form of a dict (because of the **kwargs)
    """
    # Initializing dict of events and their dates (from parameters.py)
    events_dict_copy = events_dict.copy()
    if shifting_events_dates:
        events_dict_copy = moving_events_dates_backward(events_dict=events_dict_copy, days_to_move=3)

    generall_list_of_events = ['lockdowns', 'schools', 'variants']
    # A list to change the vertical line's color
    colors_list = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#46039f', '#0d0887', '#46039f',
                   '#fdca26', '#0d0887', '#636EFA']
    # A list to change the rectangle's background color
    rectangle_colors_list = ['lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
                             'lightgray', 'lightgreen', 'lightpink', 'lightcoral', 'lightseagreen', 'lightgreen']
    fig = make_subplots(rows=len(generall_list_of_events), cols=1, vertical_spacing=0.07,
                        subplot_titles=['<b> {} plot'.format(event) for event in generall_list_of_events]
                        )
    # Filling in the subplots - each time period in a subplot
    for i in range(0, len(generall_list_of_events)):
        x = series_to_plot.index  # x-axis
        y = series_to_plot.values  # y-axis
        start_date_vline = []  # Initialize list of start dates
        end_date_vline = []  # Initialize list of end dates
        for key in events_dict_copy.keys():  # A loop to fill the empty lists
            if generall_list_of_events[i][:-1] in key.split(
                    '_'):  # If the word from the list is part of a key in event_dict
                start_date_vline.append(events_dict_copy[key][0])  # the left v-line
                end_date_vline.append(events_dict_copy[key][1])  # the right v-line
        fig.add_trace(go.Scatter(x=x, y=y, showlegend=True, name='model_pred', legendgroup=str(i)), row=i + 1, col=1)
        if kwargs != {}:  # If there were additional serieses to plot, add them to the figure
            fig.add_trace(go.Scatter(x=x, y=kwargs['raw_morbidity_data'], showlegend=True, name='raw_data',
                                     legendgroup=str(i)), row=i + 1, col=1)
        for j in range(len(start_date_vline)):  # A loop to add Vlines for each relevant event
            vline_color = colors_list.pop()  # The same color for both Vlines (start date & end date)
            variants_name_list = ['Alpha', 'Delta']
            fig.add_vline(x=start_date_vline[j], line_color=vline_color, line_dash='dash', row=i + 1, col=1) \
                .add_vline(x=end_date_vline[j], line_color=vline_color, line_dash='dash', row=i + 1, col=1) \
                .add_vrect(x0=start_date_vline[j], x1=end_date_vline[j],
                           # If the events are not variant related, link an ordered int. else, name it according to the variant
                           annotation_text=(generall_list_of_events[i][:-1] + '_' + str(j + 1)) if i < 3 else
                           variants_name_list[j],
                           annotation_position='top left', fillcolor=rectangle_colors_list.pop(), opacity=0.25,
                           line_width=0, annotation=dict(font_size=12, font_family="Times New Roman"),
                           row=i + 1, col=1)
    fig.update_layout(height=2000, width=1000, title_text="", xaxis_title="<b>Date", yaxis_title="<b>[#] of new cases",
                      legend_tracegroupgap=700, template='plotly',
                      title={
                          'text': "<b>Generall morbidity for each time period",
                          'y': 0.97,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'bottom'
                      }
                      )
    fig.show()


def plot_aggregated_specific_compartment(
        mdl_res,
        compartment,
        with_events=True,
        with_raw_morbidity_data=False
):
    """
    :param ind: the ind object
    :param mdl_res: the model after performing prediction
    :param compartment: the specific compartment to plot, comes as string (e.g: 'R_1', 'Is_1', 'new_Is_1')
    :param with_events: indicates whether to add the Vlines to mark the events inline or not
    :param with_raw_morbidity_data: whether to add the smoothed absolute new cases to the figure (mainly for calibration)
    """
    # Initialize empty list of compartments (2 elements, each one of shape 500X540)
    sum_of_compartments = mdl_res[compartment] + mdl_res[compartment.replace("1", "2")]
    # Creating array to plot
    sum_list = []
    for element in sum_of_compartments:
        sum_list.append(sum(element) * pop_israel)
    sum_array = np.array(sum_list)
    series_to_plot = pd.Series(data=sum_array, index=list(index_t_to_date_mapper.values())
    [:sum_of_compartments.shape[1]], name=compartment)

    if with_raw_morbidity_data:
        raw_morbidity_data = reading_aggregated_new_cases_file()
        # If w'd like to mark the events - we'll make subplots
        if with_events:
            plot_events(series_to_plot, raw_morbidity_data=raw_morbidity_data)

    elif with_events:
        plot_events(series_to_plot)

    else:
        # Plotting only one plot, no events
        df = series_to_plot.to_frame()
        fig = px.line(df, x=df.index, y=df['new_Is_1'], height=400, width=1000)
        fig.update_layout(
            title={
                'text': '<b> [#] of {} nation-wide'.format(compartment),
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'bottom'
            },
            xaxis_title="<b>Date",
            yaxis_title="<b>Count of new cases",
            font=dict(
                size=15
            ),
            legend=dict(
                title='<b>Legend'
            ),
            font_family="Times New Roman",
            showlegend=False
        )
        fig.show()


def plot_specific_compartment_by_age(
        ind,
        mdl_res,
        compartment
):
    """
    :param ind: the ind object
    :param mdl_res: the model after performing prediction
    :param compartment: the specific compartment to plot, comes as string (e.g: 'R_1', 'Is_1', 'new_Is_1')
    :param with_events: indicates whether to add the Vlines to mark the events inline or not
    """
    # Getting the compartment to plot (shape 500X540)
    compartment_to_plot = mdl_res[compartment]

    # dictionary of arrays to plot
    plot_dict = {}
    for age in ind.A.values():
        plot_dict[age] = compartment_to_plot[:, ind.age_dict[age]].sum(axis=1) * pop_israel
    df_to_plot = pd.DataFrame.from_dict(plot_dict, orient='index')
    df_to_plot.columns = list(index_t_to_date_mapper.values())[:compartment_to_plot.shape[1]]
    df_to_plot = df_to_plot.T

    # Plotting
    fig = px.line(data_frame=df_to_plot, height=400, width=1000)
    fig.update_layout(
        title={
            'text': '<b> [#] of {} by age'.format(compartment),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'
        },
        xaxis_title="<b>Date",
        yaxis_title="<b>Count of {}".format(compartment),
        font=dict(
            size=15
        ),
        font_family="Times New Roman"
    )
    fig.show()


def top_3_counties_new_Is(
        ind,
        mdl_res,
        first_infectious=True
):
    """
    This function plots the top 3 counties in term of new cases. It takes the indices object,
    the results of the model's predictions and whether we're interested in the first or the second framework of
    new cases
    """
    # Taking the new cases from the first or the second framework
    if first_infectious:
        new_Is = mdl_res['new_Is_1']
    else:
        new_Is = mdl_res['new_Is_2']
    # Initializing a dict - keys are regions and values are new cases per day. filling it with the sums
    region_cases_dict = {}
    for region in list(ind.region_dict.keys()):
        region_cases_dict[region] = (new_Is[:, ind.region_dict[region]].sum(axis=1)) * pop_israel
    plot_df = pd.DataFrame.from_dict(region_cases_dict, orient='index')
    plot_df.columns = index_t_to_date_mapper.values()
    plot_df['Total'] = plot_df.sum(axis=1)
    plot_df = plot_df.sort_values(by='Total', ascending=False)
    top_3_region_morbidity_percentage = round((np.sum(plot_df['Total'][:3]) / np.sum(plot_df['Total']) * 100),
                                              2)  # Retrieving the percentage of morbidity for the 3 top regions
    top_3_region_morbidity = plot_df.head(3).drop('Total', axis=1).T  # The final df for the plot
    # top_3_region_morbidity.set_index(index_t_to_date_mapper.values(), inplace= True)

    ## Plotting
    fig = px.line(data_frame=top_3_region_morbidity, height=400, width=1000)
    fig.update_layout(
        title={
            'text': '<b>Top 3 counties in new Is (new cases)',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'
        },
        xaxis_title="<b>Date",
        yaxis_title="<b>Count of cases",
        font=dict(
            size=15
        ),
        font_family="Times New Roman"
    )
    fig.show()

    print(
        "These 3 counties account for {}% percent of the new cases in Israel".format(top_3_region_morbidity_percentage))


def plot_each_compartment_trend(
        res_model,
        matrix_shape=False,
        second_framework=False
):
    """
    This function gets the result of the model (the model object after running .predict ),
    and returns the trend over time for each compartment.
    Please set matrix_shape == True for a matrix shaped output. Set it to False for one column order.
    To examine the second framework of the model, set 'second_framework' == True
    """
    # Check - Informative warning
    if matrix_shape and second_framework:
        print("Output for the second framework is column ordered only!")

    compartments_list = ['S_1', 'E_1', 'Ia_1', 'Is_1', 'R_1', 'new_Is_1', 'H_1', 'V_1', 'V_2', 'V_3', 'L_1']
    if second_framework:  # If we want to examine the second framework's compartments
        compartments_list = ['S_2', 'E_2', 'Ia_2', 'Is_2', 'R_2', 'new_Is_2', 'H_2']

    # Dimension of rows and columns of the subplots - if we want the subplots matrix shaped
    subplots_dim = int(len(compartments_list) / 3)
    fig = make_subplots(rows=subplots_dim if matrix_shape else len(compartments_list),
                        cols=subplots_dim if matrix_shape else 1, vertical_spacing=0.03,
                        subplot_titles=['<b> {}'.format(compartment) for compartment in compartments_list]
                        )
    if matrix_shape:  # For matrix-shaped output
        for i in range(subplots_dim):
            for j in range(subplots_dim):
                compartment = compartments_list.pop(0)  # The current compartment to plot
                x = list(index_t_to_date_mapper.values())  # The dates of the model
                y = res_model[compartment].sum(axis=1)  # The aggregated value of the compartment for each day
                fig.add_trace(go.Scatter(x=x, y=y, showlegend=True, name=compartment, line=dict(width=2.5)), row=i + 1,
                              col=j + 1)
        fig.update_layout(height=1200, width=2200, xaxis_title="<b>Date", yaxis_title="<b>Proportion",
                          template='plotly')
        fig.show()

    else:  # One column only
        for i in range(len(compartments_list)):
            compartment = compartments_list.pop(0)  # The current compartment to plot
            x = list(index_t_to_date_mapper.values())  # The dates of the model
            y = res_model[compartment].sum(axis=1)  # The aggregated value of the compartment for each day
            fig.add_trace(go.Scatter(x=x, y=y, showlegend=True, name=compartment, line=dict(width=2.5)), row=i + 1,
                          col=1)
        # Update the subplot's names
        for j in range(len(compartments_list)):
            fig.layout.annotations[j].update(text=compartments_list[j])

        fig.update_layout(height=3000, width=800, xaxis_title="<b>Date", yaxis_title="<b>Proportion", template='plotly')
        fig.show()


def plot_total_compartments_track(
        res_model
):
    """
    This function outputs the sum of all compartments over time, to check if the total sum each day is 1, as expected
    """
    # List of all compartments
    compartments_list = ['S_1', 'E_1', 'Ia_1', 'Is_1', 'R_1', 'V_1', 'V_2', 'V_3', 'S_2', 'E_2', 'Ia_2', 'Is_2', 'R_2']
    # Array to hold the sum of all compartments for each day
    total_array = np.zeros(res_model['S_1'].shape[0])
    # Summing over the compartments
    for compartment in compartments_list:
        total_array += np.round(a=res_model[compartment].sum(axis=1), decimals=4)

    x = list(index_t_to_date_mapper.values())[:len(total_array)]
    fig = px.line(x=x, y=total_array, title='Sum of compartments', template='plotly', height=400, width=1000)
    fig.update_layout(xaxis_title='<b>Date', yaxis_title='<b>Proportion')
    fig.show()


def plot_I_by_age_region(
        ind,
        mdl_res,
        with_asym=False,
        sym_only=False,
):
    """
    :param mdl_res:
    :param with_asym:
    :param sym_only:
    :return:
    """

    Is = mdl_res['Is_1']
    Ia = mdl_res['Ia_1']

    plot_dict = {}  # dictionary of arrays to plot

    if with_asym:
        fig, axes = plt.subplots(3, 3)
        for ax, groups in zip(axes.flat, range(9)):
            plot_dict = {}
            for age in ind.A.values():
                for s in ind.G.values():
                    plot_dict[s + ' sym'] = Is[:, ind.region_age_dict[s, age], ].sum(axis=1)
                    plot_dict[s + ' asym'] = Ia[:, ind.region_age_dict[s, age], ].sum(axis=1)

            plot_df = pd.DataFrame.from_dict(plot_dict)
            plot_df.plot(ax=ax, title=age)

    elif sym_only:
        fig, axes = plt.subplots(3, 3)
        for ax, groups in zip(axes.flat, range(9)):
            for age in ind.A.values():
                for s in ind.G.values():
                    plot_dict[s + ' sym'] = Is[:, ind.region_age_dict[s, age], ].sum(axis=1)

            plot_df = pd.DataFrame.from_dict(plot_dict)
            plot_df.plot(ax=ax, title=age)

    else:
        fig, axes = plt.subplots(3, 3)
        for ax, groups in zip(axes.flat, range(9)):
            plot_dict = {}
            for age in ind.A.values():
                for s in ind.G.values():
                    plot_dict[s] = Is[:, ind.age_dict[age]].sum(axis=1) + \
                                   Ia[:, ind.age_dict[age]].sum(axis=1)

            plot_df = pd.DataFrame.from_dict(plot_dict)
            plot_df.plot(ax=ax)
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()
    plt.close()

    return fig, axes


def plot_calibrated_model(
        ind,
        data,
        mdl_data,
        date_list,
        season_length
):
    """
    The function gets the results of the model and plot for each age group the
    model results and the data.
    """

    model_tot_dt = np.zeros((season_length + 1, len(ind.A)))
    # Calculated total symptomatic (high+low) per age group (adding as columns)
    plot_dict = {}
    plot_dict['dates'] = date_list
    for i, age_group in enumerate(ind.age_dict.keys()):
        model_tot_dt[:, i] = mdl_data[:, ind.age_dict[age_group]].sum(axis=1)
        plot_dict[ind.A[i] + '_mdl'] = mdl_data[
                                       :len(date_list),
                                       ind.age_dict[age_group],
                                       ].sum(axis=1)
        plot_dict[ind.A[i] + '_dt'] = data[:, i]

    plot_df = pd.DataFrame.from_dict(plot_dict)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))

    for ax, groups in zip(axes.flat, range(9)):
        plot_df.plot(
            x='dates',
            y=[ind.A[groups] + '_mdl', ind.A[groups] + '_dt'],
            style=['-', '.'],
            ax=ax
        )
        ax.set_xticklabels(
            labels=plot_df.dates.values[::5],
            rotation=70,
            rotation_mode="anchor",
            ha="right"
        )
    plt.ticklabel_format(axis='y', style='sci', useMathText=True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_calibrated_model_region(
        ind,
        data,
        mdl_data,
        # date_list,
        region_name,
        # start='2020-03-20',
        # end='2020-04-13',
        loss_func='MSE',
        region_mapper=None,
        pop_size_region_age=None
):
    """ The function gets the results of the model and plot for each region
     the model results and the data normalized by region population. Data format is of region-age
    """
    if region_mapper is not None:
        region_dict = region_mapper
    else:
        region_dict = ind.region_ga_dict

    if loss_func == "MSE":
        # fixing data to be proportion of israel citizens
        data_specific = data / 9345000

    # index to cut model's data
    # start_idx = int(np.where(date_list == start)[0])
    # end_idx = int(np.where(date_list == end)[0])
    plot_dict = {}
    for key in ind.region_dict.keys():
        plot_dict[key + '_mdl'] = mdl_data.iloc[:, region_dict[key]].sum(axis=1) / pop_size_region_age[
            region_dict[key]].sum()
        plot_dict[key + '_dt'] = data.iloc[:, region_dict[key]].sum(axis=1) / pop_size_region_age[
            region_dict[key]].sum()

    plot_df = pd.DataFrame.from_dict(plot_dict)
    # plot_df.set_index(date_list[start_idx:end_idx + 1], inplace=True)

    fig, axes = plt.subplots(int(np.ceil(len(region_dict) / 3)), 3, figsize=(27, 27))

    for ax, key in zip(axes.flat, region_dict.keys()):
        plot_df.plot(y=[key + '_mdl', key + '_dt'],
                     style=['-', '.'],
                     linewidth=3,
                     markersize=5,
                     ax=ax,
                     label=['Model', 'Data'])
        ax.set_title('Region {}'.format(key))
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=20, rotation_mode="anchor", ha="right")
        ax.legend(frameon=False)
        ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(0, 0))

    return fig, axes


def plot_calibrated_nationwide_model(
        data,
        mdl_data
):
    """ The function gets the results of the model and plot the model results and the data,
    on country level.
    Data format is of region-age
    """

    # Getting the raw data and the model predictions as different arrays
    raw_data_series = data.sum(axis=1)
    raw_data_series_odd_pos = raw_data_series[1::2]  # raw data only in odd indexes
    model_data_pred_series = mdl_data.sum(axis=1) * pop_israel
    # df_to_plot = pd.concat([raw_data_series, model_data_pred_series], axis= 1)
    # df_to_plot.columns = ['raw_data', 'model_pred']

    # Plotting
    fig = go.Figure()
    x_axis = pd.date_range(start='2020-05-15', end='2021-10-25')
    fig.add_trace(go.Scatter(x=x_axis[1::2], y=raw_data_series_odd_pos, hoverlabel=dict(namelength=-1),
                             mode='markers', name='data', marker=dict(size=3.5, color='darkblue', line_width=1)))
    fig.add_trace(go.Scatter(x=x_axis, y=model_data_pred_series, hoverlabel=dict(namelength=-1),
                             line=dict(width=3.5, color='rgba(255, 182, 193, .9)'), name='model'))
    # Updates
    fig.update_xaxes(range=[x_axis.min(), x_axis.max()])
    fig.update_layout(height=600, width=1200, xaxis_title="<b> Date", yaxis_title="<b> [#] of new cases",
                      font=dict(size=15), font_family="Times New Roman", template='plotly_white', title={
            'text': '<b>Country level calibration plot',
            'font_size': 42,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'
        },
                      legend=dict(title='<b>labels', traceorder='normal', bordercolor='black', bgcolor='rgba(0,0,0,0)',
                                  borderwidth=0.8, x=0.001, y=0.999,
                                  font=dict(family='sans-serif', color='black', size=15)), legend_title_font_size=16
                      )
    fig.show()


def plot_respiration_cases(res_mdl, days=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(((res_mdl['Vents']).sum(axis=1)) * pop_israel)
    ax.set_ylabel('Resipratory cases [#]', fontsize=35)
    ax.set_title('Respiratory Cases Global', fontsize=50)
    ax.set_xlabel('Time [d]', fontsize=35)
    if days is not None:
        ax.axvline(x=days, c='k', linewidth=2,
                   linestyle='--')
    plt.show()


def plot_hospitalization_cases(res_mdl):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(((res_mdl['H']).sum(axis=1)) * pop_israel)
    ax.set_ylabel('Hospitalization cases [#]', fontsize=35)
    ax.set_title('Hospitalization Cases Global', fontsize=50)
    ax.set_xlabel('Time [d]', fontsize=35)
    plt.show()


def plot_hospitalizations_calibration(res_mdl, data, date_lst, start_date, end_date, tracking='hosp'):
    # index to cut model's data
    start_idx = int(np.where(date_lst == start_date)[0])
    end_idx = int(np.where(date_lst == end_date)[0])
    # print(start_idx)
    # print(end_idx)
    # creating DF
    if tracking == 'hosp':
        model_data_cal = res_mdl['H'][start_idx:end_idx + 1].sum(axis=1) * pop_israel
        y_label = 'Number of hospitalizations'
    elif tracking == 'vents':
        model_data_cal = res_mdl['Vents'][start_idx:end_idx + 1].sum(axis=1) * pop_israel
        y_label = 'Number of Ventilators in use'

    plot_dict = {}
    plot_dict['Model'] = model_data_cal
    plot_dict['date'] = date_lst[start_idx:end_idx + 1]
    plot_dict['Data'] = data

    # print('len model:',len(model_data_cal))
    # print('len date:', len(date_lst[start_idx:end_idx+1]))
    # print('len data:', len(data))

    plot_df = pd.DataFrame.from_dict(plot_dict)
    plot_df.set_index('date', inplace=True)

    # Plot
    fig = plt.figure()
    ax = plot_df.plot(style=['-', '.'])
    ax.set_title('Country level calibration plot')
    ax.set_ylabel(y_label)
    plt.show()

    return fig, ax


def make_casulties(res_model, time_ahead, pop_israel, mu):
    return (res_model['Vents'].sum(axis=1))[:time_ahead].sum() * pop_israel * mu / 3.0


def make_recoveries(res_model, time_ahead):
    return (res_model['R'].sum(axis=1))[time_ahead] * 100


def make_ill_end(res_model, time_ahead, pop_israel):
    return ((res_model['Ie'] + res_model['Is'] + res_model['Ia']).sum(axis=1))[time_ahead] * pop_israel


def make_casulties_interval(
        ind,
        res_mdl,
        time_ahead,
        pop_israel,
        mu,
        xi,
        gamma,
        vents_conf,
):
    casulties = {}
    options = ['pr_vents_ub', 'pr_vents_lb']
    for vent_col in options:
        chi = expand_partial_array(ind.risk_age_dict,
                                   vents_conf[vent_col].values,
                                   len(ind.N))
        vents = {}
        vents['Vents'] = [np.zeros_like(res_mdl['Is'][0])]
        vents_latent = [np.zeros_like(res_mdl['Is'][0])]
        for t in range(len(res_mdl['Is'][:time_ahead])):
            vents['Vents'].append(
                vents['Vents'][t] + xi * vents_latent[t] - mu * vents['Vents'][t])

            # Vents_latent(t)
            vents_latent.append(
                vents_latent[t] + (chi * gama) * res_mdl['Is'][t] -
                xi * vents_latent[t])
        vents['Vents'] = np.array(vents['Vents'])
        casulties[vent_col] = make_casulties(vents, time_ahead, pop_israel, mu)

    return tuple(casulties.values())


def make_death_interval(
        ind,
        res_mdl,
        time_ahead,
        pop_israel,
        death_conf,
):
    casulties = {}
    options = ['pr_death_lb', 'pr_death', 'pr_death_ub']
    for vent_col in options:
        chi = expand_partial_array(ind.risk_age_dict,
                                   death_conf[vent_col].values,
                                   len(ind.N))
        deaths = [np.zeros_like(res_mdl['new_Is'][0])]
        for t in range(len(res_mdl['new_Is'][:time_ahead])):
            # deaths(t)
            deaths.append(
                (chi) * res_mdl['new_Is'][t])
        deaths = np.array(deaths)
        casulties[vent_col] = (deaths.sum(axis=1))[:time_ahead].sum() * pop_israel

    return tuple(casulties.values())
