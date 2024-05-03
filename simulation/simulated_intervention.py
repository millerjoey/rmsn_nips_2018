# Import 
import simulation.cancer_simulation as sim
import pickle
import numpy as np
import logging

def _generate(save = False, assigned_actions = None):
    chemo_coeff = 10
    radio_coeff = 3
    window_size = 15
    num_time_steps = 60
    np.random.seed(100)
    num_patients = 10000
    pickle_file = 'models/cancer_sim_intervention_{}_{}.pkl'.format(chemo_coeff, radio_coeff)

    params = sim.get_confounding_params(num_patients, chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    training_data = simulate(params, num_time_steps, assigned_actions)

    params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    validation_data = simulate(params, num_time_steps, assigned_actions)

    params = sim.get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                        radio_coeff=radio_coeff)
    params['window_size'] = window_size
    test_data = simulate(params, num_time_steps, assigned_actions)

    scaling_data = sim.get_scaling_params(training_data)

    pickle_map = {'chemo_coeff': chemo_coeff,
                    'radio_coeff': radio_coeff,
                    'num_time_steps': num_time_steps,
                    'training_data': training_data,
                    'validation_data': validation_data,
                    'test_data': test_data,
                    'scaling_data': scaling_data,
                    'window_size': window_size}

    logging.info("Saving pickle map to {}".format(pickle_file))
    if save:
        pickle.dump(pickle_map, open(pickle_file, 'wb'))
    return pickle_map

def simulate(simulation_params, num_time_steps, assigned_actions=None):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_patients,
                                         num_time_steps)  # 5% cell variability
    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):

        logging.info("Simulating patient {} of {}".format(i + 1, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]


        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(1, num_time_steps):

            cancer_volume[i, t] = abs(cancer_volume[i, t - 1] * (1 + \
                                  + rho * np.log(K / cancer_volume[i, t - 1]) \
                                  - beta_c * chemo_dosage[i, t - 1] \
                                  - (alpha * radio_dosage[i, t - 1] + beta * radio_dosage[i, t - 1] ** 2)))  # add noise to fit residuals

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t-1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t]
            cancer_diameter_used = np.array([sim.calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:

                radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i]
                                             *(cancer_metric_used - radio_sigmoid_intercepts[i]))))
                chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                                  (cancer_metric_used - chemo_sigmoid_intercepts[i]))))
            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob :

                    radio_application_point[i, t] = 1
                    radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:

                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # if cancer_volume[i, t] > sim.tumour_death_threshold:
            #     b_death = True
                
            # # recovery threshold as defined by the previous stuff
            # if recovery_rvs[i, t] < np.exp(-cancer_volume[i, t] * sim.tumour_cell_density):
            #     b_recover = True

        # Package outputs
        sequence_lengths[i] = int(t+1)
        death_flags[i, t] = 1 if b_death else 0
        recovery_flags[i, t] = 1 if b_recover else 0

    outputs = {'cancer_volume': cancer_volume,
               'chemo_dosage': chemo_dosage,
               'radio_dosage': radio_dosage,
               'chemo_application': chemo_application_point,
               'radio_application': radio_application_point,
               'chemo_probabilities': chemo_probabilities,
               'radio_probabilities': radio_probabilities,
               'sequence_lengths': sequence_lengths,
               'death_flags': death_flags,
               'recovery_flags': recovery_flags,
               'patient_types': patient_types
               }

    return outputs


assigned_actions = np.random.randint(0, 2, (10000, 60, 2))

tumor_data = _generate(True, assigned_actions=assigned_actions)

# Using a modified version of simulate that removes noise on the cancer volume after the nth time step