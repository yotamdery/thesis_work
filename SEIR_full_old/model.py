#!/usr/bin/env python3
from numpy import ndarray

from SEIR_full.utils import *
from SEIR_full.parameters import *
from SEIR_full.plot_utils import *
#from notebooks.Yotams.Calibration.calibration_scripts.calibration_SEIR_model import aggregate_to_3_age_groups
import pandas as pd
import numpy as np
import scipy.sparse
from scipy import optimize


#######################
# ---- Model Class---- #
#######################

class Model_behave:
    def __init__(
            self,
            ind,
            beta_j= np.array([0.055, 0.055, 0.055, 0.05, 0.05, 0.05, 0.05, 0.001, 0.001]) * (0.83),
            beta_lockdown= 0.8,
            beta_isolation = 0.26,  # Permanent over the whole model's dates
            beta_school= 0.515,
            # beta_activity = 0.8,
            # activity_vec = visitors_covid,
            is_lockdown = lockdown_vec,  # 1 if lockdown was lifted, else 0
            is_school = school_vec,  # 0 if there was school, else 0
            # theta= 2.826729434860104,
            # theta_arab= 1.0,
            # beta_behave= 0.5552998605894367,
            is_haredi=is_haredi,
            is_arab=is_arab,
            alpha=alpha,
            # sigma= sigma,
            delta=delta,
            gama=gama,
            omega=omega,  # waining rate from recovered to susceptible
            tau=tau,  # parameter to reduce the second force of infection
            # population_size= population_size,
            init_model=init_model,  # Dictionary - initiation of the model
            rho= hospitalizations,
            chi=vents_proba,
            nu=nu,
            mu=mu,
            eta=eta,
            xi=xi,
            scen= 'Scenario2',
            # eps= eps_sector,
            f=f0_full,
            s_1_to_v1_transition_t=s_1_to_v1_transition_t,  # From the vaccination analysis
            v1_to_v2_transition=v1_to_v2_transition,       # Permanent - 1/21
            v2_to_v3_transition_t= v2_to_v3_transition_t,  # From the vaccination analysis
            booster_efficiency = 0.95,
            alpha_variant_vec= alpha_variant_vec,
            delta_variant_vec= delta_variant_vec,
            omicron_variant_vec= omicron_variant_vec,       # for omicron validation
            isolation_morbidity_ratio_vector= isolation_morbidity_ratio_vector,
            index_t_to_date_mapper = index_t_to_date_mapper
    ):
        """
                        Receives all model's hyper-parameters and creates model object
                        the results.
                        """
        # define indices
        self.ind = ind

        # defining parameters:
        self.init_region_pop = shrink_array_sum(
            mapping_dic=ind.region_dict,
            array_to_shrink=population_size
        )

        self.beta_j = beta_j
        self.beta_lockdown = beta_lockdown
        self.is_lockdown = is_lockdown
        #self.beta_activity = beta_activity
        #self.activity_vec = activity_vec
        self.beta_isolation = beta_isolation
        self.is_school = is_school
        self.beta_school = beta_school
        self.is_haredi = is_haredi
        self.is_arab = is_arab
        self.alpha = alpha          # factor for the probability for hospitalizations
        self.delta = delta
        self.gama = gama
        self.omega = omega
        self.tau = tau
        self.rho = rho
        self.nu = nu
        # self.sigma = sigma
        self.scen = scen
        # self.eps = eps
        self.f = f
        self.chi = chi
        self.mu = mu
        self.eta = eta
        self.xi = xi
        # self.population_size = population_size.copy()
        self.init_model = init_model.copy()
        self.s_1_to_v1_transition_t = s_1_to_v1_transition_t
        self.v1_to_v2_transition = v1_to_v2_transition
        self.v2_to_v3_transition_t = v2_to_v3_transition_t
        self.s_2_to_v3_transition_t = v2_to_v3_transition_t     # Note that this transition rate is identical in purpose
        self.booster_efficiency = booster_efficiency           # Efficiency of the booster dose
        self.alpha_variant_vec = alpha_variant_vec
        self.delta_variant_vec = delta_variant_vec
        self.omicron_variant_vec = omicron_variant_vec
        self.isolation_morbidity_ratio_vector = isolation_morbidity_ratio_vector
        self.index_t_to_date_mapper = index_t_to_date_mapper

        # defining model compartments:
        self.reset(self.init_model, self.scen)  # , self.eps)

        # counter for training
        self.fit_iter_count = 0

    def fit_tracking(
            self,
            p0,
            bnds,
            data,
            days_in_season=70,
            # C= C_calibration,
            # stay_home_idx=stay_home_idx,
            # not_routine=not_routine,
            method='TNC',
            maxiter=200,
            date_lst=pd.date_range('2020-02-20', periods=100, freq='d'),
            start='2020-03-20',
            end='2020-04-13',
            loss_func='MSE',
            factor=1,
            tracking='hosp'
    ):

        self.reset()
        self.fit_iter_count = 0

        res_fit = optimize.minimize(
            self.calc_loss_tracking,
            p0,
            bounds=bnds,
            method=method,
            args=(
                data,
                days_in_season,
                # C,
                # stay_home_idx,
                # not_routine,
                date_lst,
                start,
                end,
                loss_func,
                factor,
                tracking
            ),
            options={'maxiter': maxiter},
        )
        fitted_params = res_fit.x

        # run the fitted model:
        if tracking == 'hosp':
            eta = fitted_params[0]
            nu = fitted_params[1]

            self.update({
                'nu': nu,
                'eta': eta
            })

        elif tracking == 'vents':
            xi = fitted_params[0]
            mu = fitted_params[1]

            self.update({
                'mu': mu,
                'xi': xi
            })
        return res_fit

    def fit(
            self,
            p0,
            bnds,
            data,
            cols_names_original_data,   # Original names of the 270 columns before aggregation to 3 age groups
            # C= C_calibration,
            # stay_home_idx=stay_home_idx,
            # not_routine=not_routine,
            method='TNC',
            maxiter= 200,
            # date_lst=pd.date_range('2020-02-20', periods=100, freq='d'),
            # start='2020-03-20',
            # end='2020-04-13',
            loss_func='MSE',
            factor=1,
            # mapper= None,
    ):

        self.reset()
        self.fit_iter_count = 0

        res_fit = optimize.minimize(
            self.calc_loss,
            p0,
            bounds= bnds,
            method= method,
            args=(
                data,
                cols_names_original_data,
                # C,
                # stay_home_idx,
                # not_routine,
                #date_lst,
                #start,
                #end,
                loss_func,
                factor
                #mapper,
            ),
            options={'maxiter': maxiter},
        )
        fitted_params = res_fit.x

        # run the fitted model:
        fitted_beta = np.array(
            [fitted_params[0], fitted_params[0], fitted_params[0],
             fitted_params[1], fitted_params[1], \
             fitted_params[1], fitted_params[1], fitted_params[2],
             fitted_params[2]])
        beta_lockdown = fitted_params[3]
        #beta_activity = fitted_params[3]
        beta_isolation = fitted_params[4]
        beta_school = fitted_params[5]

        self.update({
            'beta_j': fitted_beta,
            #'beta_activity': beta_activity,
            'beta_lockdown': beta_lockdown,
            'beta_isolation': beta_isolation,
            'beta_school': beta_school,
        })
        return res_fit

    def reset(
            self,
            init_model=None,
            scen=None,
            # eps= None
    ):
        """
		Reset object's SEIR compartments.
		:param population_size:
		:param eps:
		"""
        if init_model is None:
            init_model = self.init_model
        # if eps is None:
        # 	eps = self.eps

        # defining model compartments:
        # self.L == Tracking on the force of infection
        self.S_1, self.E_1, self.Is_1, self.new_Is_1, self.Ia_1, self.R_1, self.V_1, self.V_2, self.V_3, \
        self.Vents_latent_1, self.new_H_1, self.H_1, self.L_1, self.L_home_1, self.Vents_1, self.Hosp_latent_1, \
        self.S_2, self.E_2, self.Is_2, self.new_Is_2, self.Ia_2, self.R_2, \
        self.Vents_latent_2, self.new_H_2, self.H_2, self.L_2, self.Vents_2, self.Hosp_latent_2 = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(12):
            ## Initializing compartments - first infection:
            self.L_1.append(np.zeros(len(self.ind.N)))
            # Initializing the first framework of the model (S_1, E_1, Is_1, Ia_1, R_1), according to the scenario
            self.S_1.append(init_model['init_S_1'].copy())

            # Initialize S_1 to population size of each age-group
            # self.S_1.append(population_size.copy())

            # Initialize R_1 - with only the naturally immune individuals
            # self.R_1.append(np.zeros(len(self.ind.N)))
            self.R_1.append(init_model['init_R_1'].copy())

            # Initialize E_1
            # self.E_1.append(np.zeros(len(self.ind.N)))
            self.E_1.append(init_model['init_E_1'].copy())

            # Initialize Ia_1 (asymptomatic)
            # self.Ia_1.append(np.zeros(len(self.ind.N)))
            self.Ia_1.append(init_model['init_Ia_1'].copy())

            # Initialize Is_1 (symptomatic)
            # self.Is_1.append(np.zeros(len(self.ind.N)))
            self.Is_1.append(init_model['init_Is_1'].copy())

            # Zero newly infected on the first day of the season
            self.new_Is_1.append(np.zeros(len(self.ind.N)))

            # Subtract E(0) from S(0)
            self.S_1[-1] -= (self.E_1[-1])

            # Initialize V1, V2, V3 Vaccination compartments
            self.V_1.append(np.zeros(len(self.ind.N)))
            self.V_2.append(np.zeros(len(self.ind.N)))
            self.V_3.append(np.zeros(len(self.ind.N)))

            # Initialize H_1(0), new_H_1(0) Vents_1(0), H_latent_1(0), Vents_latent_1(0) tracking compartment
            self.H_1.append(np.zeros(len(self.ind.N)))
            self.new_H_1.append(np.zeros(len(self.ind.N)))
            self.Vents_1.append(np.zeros(len(self.ind.N)))
            self.Hosp_latent_1.append(np.zeros(len(self.ind.N)))
            self.Vents_latent_1.append(np.zeros(len(self.ind.N)))

            ## Initializing compartments - second infection:
            self.L_2.append(np.zeros(len(self.ind.N)))
            # Initialize R_2
            self.R_2.append(np.zeros(len(self.ind.N)))
            # Initialize S_2
            self.S_2.append(np.zeros(len(self.ind.N)))
            # Initialize E_2
            self.E_2.append(np.zeros(len(self.ind.N)))
            # Initialize Ia_2 (asymptomatic)
            self.Ia_2.append(np.zeros(len(self.ind.N)))
            # Initialize Is_2 (symptomatic)
            self.Is_2.append(np.zeros(len(self.ind.N)))
            # Initialize new_Is_2
            self.new_Is_2.append(np.zeros(len(self.ind.N)))

            # Initialize H_2(0), new_H_2(0), Vents_2(0), H_latent_2(0), Vents_latent_2(0) tracking compartment
            self.H_2.append(np.zeros(len(self.ind.N)))
            self.new_H_2.append(np.zeros(len(self.ind.N)))
            self.Vents_2.append(np.zeros(len(self.ind.N)))
            self.Hosp_latent_2.append(np.zeros(len(self.ind.N)))
            self.Vents_latent_2.append(np.zeros(len(self.ind.N)))

    def calc_loss_tracking(
            self,
            tpl,
            data,
            C,
            days_in_season,
            stay_home_idx,
            not_routine,
            date_lst=pd.date_range('2020-02-20', periods=70, freq='d'),
            start='2020-03-20',
            end='2020-04-13',
            loss_func='MSE',
            factor=1,
            tracking='hosp'
    ):
        """
		Calibrates the model to data
		:param self:
		:param tpl:
		:param days_in_season:
		:param loss_func: loss function to minimize 'MSE' or 'BIN' or 'POIS'
		:param tracking: tracking compartment to calibrate 'hosp' or 'vents'
		:returns: loss functions' value
		"""
        # update counter of runs since last fit action
        self.fit_iter_count = self.fit_iter_count + 1

        if tracking == 'hosp':
            # Run model with given parameters
            model_res = self.predict(
                # C=C,
                days_in_season=days_in_season,
                # stay_home_idx=stay_home_idx,
                # not_routine=not_routine,
                eta=tpl[0],
                nu=tpl[1]
            )
            new_cases_model = model_res['H'].sum(axis=1)

        elif tracking == 'vents':
            # Run model with given parameters
            model_res = self.predict(
                # C=C,
                days_in_season=days_in_season,
                # stay_home_idx=stay_home_idx,
                # not_routine=not_routine,
                xi=tpl[0],
                mu=tpl[1]
            )
            new_cases_model = model_res['Vents'].sum(axis=1)

        # Taking relevant time frame from model
        start_idx = int(np.where(date_lst == start)[0])
        end_idx = int(np.where(date_lst == end)[0])
        model_for_loss = new_cases_model[start_idx:end_idx + 1].copy()

        if loss_func == "MSE":
            # fixing data to be proportion of israel citizens
            data_specific = data / pop_israel
            loss = np.log(MSE(data_specific, model_for_loss))

        self.reset()
        if self.fit_iter_count % 50 == 0:
            print('iter: ', self.fit_iter_count, ' loss: ', loss)
        return loss

    def calc_loss(
            self,
            p0,
            data,
            cols_names_original_data,
            # C,
            # stay_home_idx,
            # not_routine,
            #date_lst=pd.date_range('2020-02-20', periods=70, freq='d'),
            #start='2020-03-20',
            #end='2020-04-13',
            loss_func= 'MSE',
            factor=1,
    ):
        """
		Calibrates the model to data
		:param self:
		:param tpl:
		:param loss_func: loss function to minimize 'MSE'
		:returns: loss functions' value
		"""
        # update counter of runs since last fit action
        self.fit_iter_count = self.fit_iter_count + 1

        # setting parameters
        beta_j = np.array([p0[0], p0[0], p0[0], p0[1], p0[1], p0[1], p0[1], p0[2], p0[2]])

        # Run model with given parameters
        model_res = self.predict(
            # C=C,
            days_in_season= 526,
            shifting_12_days=True,
            # stay_home_idx=stay_home_idx,
            # not_routine=not_routine,
            beta_j= beta_j,
            #beta_activity= p0[3],
            beta_lockdown= p0[3],
            beta_isolation= p0[4],
            beta_school= p0[5]
        )
        new_cases_model = model_res['new_Is_1'] + model_res['new_Is_2']
        model_results_cal = np.zeros((len(new_cases_model), len(self.ind.region_age_dict)))

        # Calculated total symptomatic (high+low) per age group (adding as columns)
        for i, key in enumerate(self.ind.region_age_dict.keys()):
            model_results_cal[:, i] = new_cases_model[:, self.ind.region_age_dict[key]].sum(axis=1)

        final_df_pred_model = aggregate_to_3_age_groups(pd.DataFrame(model_results_cal, columns= cols_names_original_data))
        if loss_func == "MSE":
            # fixing data to be proportion of israel citizens
            data_specific = data.to_numpy() / pop_israel
            # Calculating the loss
            loss = np.log(MSE(data_specific, final_df_pred_model))

        self.reset()
        if self.fit_iter_count % 50 == 1:
            print('iter: ', self.fit_iter_count, ' loss: ', loss)
            print('curr param: BETAj:', p0[0], ' ', p0[1], ' ', p0[2], ' ',
                  'beta_lockdown:', p0[3],
                  'beta_isolation:', p0[4],
                  'beta_school:', p0[5])
        return loss

    def update(self, new_param):
        """
		update object's attributes.
		:param new_param: dictionary object, key=attribute name, value= attribute value to assign
		"""
        for key, value in new_param.items():
            setattr(self, key, value)

    def predict(
            self,
            # C,
            # stay_home_idx,
            # not_routine,
            days_in_season,
            start=0,
            beta_j=None,
            beta_lockdown= None,
            #beta_activity= None,
            beta_isolation= None,
            beta_school= None,
            xi=None,
            mu=None,
            eta=None,
            nu=None,
            continuous_predict = False,     # If we predict upon the model's continuation (without doing a reset)
            shifting_12_days= False         # If we choose to shift the predictions 12 days forward
    ):
        """
		Receives  model's parameters and run model for days_in_season days
		the results.
		:param C:
		:param days_in_season:
		:param stay_home_idx:
		:param not_routine:
		:param beta_j:
		:param beta_behave:
		:param theta:
		:param theta_arab:
		:param xi:
		:param mu:
		:param eta:
		:param nu
		:return:
		"""

        if beta_j is None:
            beta_j = self.beta_j
        # if beta_activity is None:
        #     beta_activity = self.beta_activity
        if beta_lockdown is None:
            beta_lockdown = self.beta_lockdown
        if beta_isolation is None:
            beta_isolation = self.beta_isolation
        if beta_school is None:
            beta_school = self.beta_school
        if xi is None:
            xi = self.xi
        if mu is None:
            mu = self.mu
        if eta is None:
            eta = self.eta
        if nu is None:
            nu = self.nu

        # Expanding the beta_j vector to match the size of the contact force (1X270)
        beta_j = expand_partial_array(
            mapping_dic=self.ind.age_ga_dict,
            array_to_expand=beta_j,
            size=len(self.ind.GA)
        )

        # Running the SEIR model:
        if shifting_12_days:    # If it's the first time that we run the predict:
            running_time_model = days_in_season - 12
        else:                   # If we don't want to shift forward 12 days:
            # We subtract 1 because the first day is reflected in the start conditions of the model
            running_time_model = days_in_season - 1
        if continuous_predict:
            running_time_model = days_in_season

        for t in np.arange(running_time_model):
            # Calculating the multiplication of the contact matrix with the recent I compartments
            contact_force_matrix, vector_to_plot = self.calculate_force_matrix(final_contact_matrix, beta_isolation, t)

            # Calculating the force of infection, Lambda_j_k:
            lambda_t = (self.alpha_variant_vec[t]) * self.delta_variant_vec[t] * self.omicron_variant_vec[t] * \
                    (beta_lockdown ** self.is_lockdown[t]) * (beta_school ** self.is_school[t]) * \
                    beta_j * (fertility_rate_orthodox * self.is_haredi + fertility_rate_arabs *
                    self.is_arab+fertility_rate_sacular * (1 - self.is_haredi - self.is_arab)) * contact_force_matrix


            # Expanding the array to match the dimensions of the S compartment (1X540),
            lambda_t = expand_partial_array(
                mapping_dic=self.ind.region_age_dict,
                array_to_expand=lambda_t,
                size=len(self.ind.N),
            )

            # Updating the transition rate of v1 and v2 to e - determine these by the force of infection (lambda_t):
            self.v1_to_e_transition = lambda_t * (
                    1 - 0.90)  # Force of infection, multiplied by (1 - first_dose_efficiency as reported by Pfizer)
            self.v2_to_e_transition = lambda_t * (
                    1 - 0.94)  # Force of infection, multiplied by (1 - second_dose_efficiency as reported by Pfizer)
            self.v3_to_e_transition = lambda_t * (
                    1 - self.booster_efficiency)  # Force of infection, multiplied by (1 - third_dose_efficiency as reported by Pfizer)

            ## UPDATING THE STATES OF THE DYNAMIC MODEL
            # Keeping track on the second force of infection
            self.L_2.append(lambda_t)

            # R_2(t)
            self.R_2.append(self.R_2[-1] + self.gama * (self.Is_2[-1] + self.Ia_2[-1]) - self.omega * self.R_2[-1])

            # For tracking: H_2(t)- Hospitalized patients now (snap-shot)
            self.H_2.append(self.H_2[-1] + eta * self.Hosp_latent_2[-1] - nu * self.H_2[-1])

            # For tracking: new_H_2(t)- new (daily, non accumulated) Hospitalized patients
            self.new_H_2.append(self.new_Is_2[-1] * self.rho * self.alpha)

            # For tracking: Vents_2(t) - ventilated patients (who are hospitalized as well) now (snap-shot)
            self.Vents_2.append(self.Vents_2[-1] + xi * self.Vents_latent_2[-1] - mu * self.Vents_2[-1])

            # For tracking: H_latent_2(t) - people who are waiting to be hospitalized
            self.Hosp_latent_2.append(self.Hosp_latent_2[-1] + (self.rho * self.gama * self.alpha) * self.Is_2[-1] -
                                      eta * self.Hosp_latent_2[-1])

            # For tracking: Vents_latent_2(t) - people who are waiting to be ventilated
            self.Vents_latent_2.append(self.Vents_latent_2[-1] + (self.chi * self.gama) * self.Is_2[-1] -
                                       xi * self.Vents_latent_2[-1])

            # For tracking: save new_Is_2
            self.new_Is_2.append((1 - self.f[self.scen]) * self.delta * self.E_2[-1])

            # Is_2(t)
            self.Is_2.append(self.Is_2[-1] + self.new_Is_2[-1] - self.gama * self.Is_2[-1])

            # Ia_2(t)
            self.Ia_2.append(
                self.Ia_2[-1] + (self.f[self.scen] * self.delta * self.E_2[-1]) - self.gama * self.Ia_2[-1])

            # E_2(t)
            self.E_2.append(self.E_2[-1] + lambda_t * self.S_2[-1] - self.delta * self.E_2[-1] +
                            self.v1_to_e_transition * self.V_1[-1] + self.v2_to_e_transition * self.V_2[-1] +
                            self.v3_to_e_transition * self.V_3[-1])
            # Deleted this component: + self.eps[self.scen][t - start]
            # This is responsible for the ignition of the model and belongs only to E_1

            # S_2(t)
            self.S_2.append( self.S_2[-1] + self.omega * (self.R_1[-1] + self.R_2[-2]) + self.tau * self.V_2[-1]
                             + self.omega * self.V_3[-1] - lambda_t * self.S_2[-1] - self.s_2_to_v3_transition_t[t] * self.S_2[-1])

            # Keeping track on the force of infection
            self.L_1.append(lambda_t)

            # R_1(t)
            self.R_1.append(self.R_1[-1] + self.gama * (self.Is_1[-1] + self.Ia_1[-1]) - self.omega * self.R_1[-1])

            # For tracking: H_1(t)- Hospitalized patients now (snap-shot)
            self.H_1.append(self.H_1[-1] + eta * self.Hosp_latent_1[-1] - nu * self.H_1[-1])

            # For tracking: new_H_1(t)- new (daily, non accumulated) Hospitalized patients
            self.new_H_1.append(self.new_Is_1[-1] * self.rho)

            # For tracking: Vents(t) - ventilated patients (who are hospitalized as well) now (snap-shot)
            self.Vents_1.append(self.Vents_1[-1] + xi * self.Vents_latent_1[-1] - mu * self.Vents_1[-1])

            # For tracking: H_latent(t) - people who are waiting to be hospitalized
            self.Hosp_latent_1.append(self.Hosp_latent_1[-1] + (self.rho * self.gama) * self.Is_1[-1] -
                                      eta * self.Hosp_latent_1[-1])

            # For tracking: Vents_latent(t) - people who are waiting to be ventilated
            self.Vents_latent_1.append(self.Vents_latent_1[-1] + (self.chi * self.gama) * self.Is_1[-1] -
                                       xi * self.Vents_latent_1[-1])

            # Save new_Is_1 - for tracking
            self.new_Is_1.append((1 - self.f[self.scen]) * self.delta * self.E_1[-1])

            # Is_1(t)
            self.Is_1.append(self.Is_1[-1] + self.new_Is_1[-1] - self.gama * self.Is_1[-1])

            # Ia_1(t)
            self.Ia_1.append(
                self.Ia_1[-1] + (self.f[self.scen] * self.delta * self.E_1[-1]) - self.gama * self.Ia_1[-1])

            # E_1(t) ( ERASED: + self.eps[self.scen][t-start] )
            self.E_1.append(self.E_1[-1] + lambda_t * self.S_1[-1] - self.delta * self.E_1[-1])

            # V_1(t)
            self.V_1.append(
                self.V_1[-1] + self.s_1_to_v1_transition_t[t] * self.S_1[-1] - self.v1_to_e_transition * self.V_1[
                    -1] - self.v1_to_v2_transition * self.V_1[-1])

            # V_2(t)
            # We take the [-2] element because the inserting state was updated before the receiving state, and therefore proceeded to the t(n+1) day
            self.V_2.append(
                self.V_2[-1] + self.v1_to_v2_transition * self.V_1[-2] - self.v2_to_e_transition * self.V_2[-1] -
                self.v2_to_v3_transition_t[t] * self.V_2[-1] - self.tau * self.V_2[-1])

            # V_3(t)
            self.V_3.append(self.V_3[-1] + self.v2_to_v3_transition_t[t] * self.V_2[-2]
                            + self.s_2_to_v3_transition_t[t] * self.S_2[-2] - self.v3_to_e_transition * self.V_3[-1]
                            - self.omega * self.V_3[-1] )

            # S_1(t)
            self.S_1.append(self.S_1[-1] - lambda_t * self.S_1[-1] - self.s_1_to_v1_transition_t[t] * self.S_1[-1])

        # Return the model results
        return {
            'S_1': np.array(self.S_1),
            'E_1': np.array(self.E_1),
            'Ia_1': np.array(self.Ia_1),
            'Is_1': np.array(self.Is_1),
            'new_Is_1': np.array(self.new_Is_1),
            'R_1': np.array(self.R_1),
            'V_1': np.array(self.V_1),
            'V_2': np.array(self.V_2),
            'V_3': np.array(self.V_3),
            'L_1': np.array(self.L_1),
            'H_1': np.array(self.H_1),
            'new_H_1': np.array(self.new_H_1),
            'Hosp_latent_1': np.array(self.Hosp_latent_1),
            'Vents_1': np.array(self.Vents_1),
            'Vents_latent_1': np.array(self.Vents_latent_1),
            'S_2': np.array(self.S_2),
            'E_2': np.array(self.E_2),
            'Ia_2': np.array(self.Ia_2),
            'Is_2': np.array(self.Is_2),
            'new_Is_2': np.array(self.new_Is_2),
            'R_2': np.array(self.R_2),
            'L_2': np.array(self.L_2),
            'H_2': np.array(self.H_2),
            'new_H_2': np.array(self.new_H_2),
            'Hosp_latent_2': np.array(self.Hosp_latent_2),
            'Vents_2': np.array(self.Vents_2),
            'Vents_latent_2': np.array(self.Vents_latent_2)

        }

    def get_res(self):
        return {
            'S_1': np.array(self.S_1),
            'E_1': np.array(self.E_1),
            'Ia_1': np.array(self.Ia_1),
            'Is_1': np.array(self.Is_1),
            'R_1': np.array(self.R_1),
            'V_1': np.array(self.V_1),
            'V_2': np.array(self.V_2),
            'V_3': np.array(self.V_3),
            'new_Is_1': np.array(self.new_Is_1),
            'L_1': np.array(self.L_1),
            'H_1': np.array(self.H_1),
            'new_H_1': np.array(self.new_H_1),
            'Hosp_latent_1': np.array(self.Hosp_latent_1),
            'Vents_1': np.array(self.Vents_1),
            'Vents_latent_1': np.array(self.Vents_latent_1),
            'S_2': np.array(self.S_2),
            'E_2': np.array(self.E_2),
            'Ia_2': np.array(self.Ia_2),
            'Is_2': np.array(self.Is_2),
            'new_Is_2': np.array(self.new_Is_2),
            'R_2': np.array(self.R_2),
            'L_2': np.array(self.L_2),
            'H_2': np.array(self.H_2),
            'new_H_2': np.array(self.new_H_2),
            'Hosp_latent_2': np.array(self.Hosp_latent_2),
            'Vents_2': np.array(self.Vents_2),
            'Vents_latent_2': np.array(self.Vents_latent_2)
        }

    def calculate_force_matrix(self, final_contact_matrix, beta_isolation, t):
        # Multiply the contact matrix by the different Is
        # reshape(-1) == operating transpose on a vector
        force_contact_infectious = final_contact_matrix.T.dot((
                self.Is_1[-1][self.ind.risk_dict['High']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                              * beta_isolation)) +
                self.Is_1[-1][self.ind.risk_dict['Low']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                             * beta_isolation)) +
                self.Ia_1[-1][self.ind.risk_dict['High']] +
                self.Ia_1[-1][self.ind.risk_dict['Low']] +
                self.Is_2[-1][self.ind.risk_dict['High']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                              * (beta_isolation))) +
                self.Is_2[-1][self.ind.risk_dict['Low']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                             * (beta_isolation))) +
                self.Ia_2[-1][self.ind.risk_dict['High']] +
                self.Ia_2[-1][self.ind.risk_dict['Low']]
        )).reshape(-1)
        vector_to_plot = (self.Is_1[-1][self.ind.risk_dict['High']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                                        * (beta_isolation))) +
                          self.Is_1[-1][self.ind.risk_dict['Low']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                                       * (beta_isolation))) +
                          self.Ia_1[-1][self.ind.risk_dict['High']] +
                          self.Ia_1[-1][self.ind.risk_dict['Low']] +
                          self.Is_2[-1][self.ind.risk_dict['High']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                                        * (beta_isolation))) +
                          self.Is_2[-1][self.ind.risk_dict['Low']] * ((1 - self.isolation_morbidity_ratio_vector[t]
                                                                       * (beta_isolation))) +
                          self.Ia_2[-1][self.ind.risk_dict['High']] +
                          self.Ia_2[-1][self.ind.risk_dict['Low']]).sum()
        return force_contact_infectious, vector_to_plot
