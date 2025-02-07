import pharmaceutical_example_with_multiple_init_states_modified_obs

# Run experiments

# Experiment with dynamic masking -> sensor noise 0.15 (beta = 0.85) and threshold 70.
pharmaceutical_example_with_multiple_init_states_modified_obs.run_pharma_example_with_dynamic_maskin(iter_num=3000,
                                                                                                     batch_size=100,
                                                                                                     V=1500,
                                                                                                     T=10, threshold=70,
                                                                                                     prior_compute_flag=0,
                                                                                                     exp_number=6,
                                                                                                     sensor_noise=0.15,
                                                                                                     eta=8.2,
                                                                                                     kappa=0.25)
# Experiment with no masking -> sensor noise 0.25 (beta = 0.75) and threshold 70.
pharmaceutical_example_with_multiple_init_states_modified_obs.run_pharma_example_with_dynamic_maskin(iter_num=3000,
                                                                                                     batch_size=100,
                                                                                                     V=1500,
                                                                                                     T=10, threshold=70,
                                                                                                     prior_compute_flag=0,
                                                                                                     exp_number=7,
                                                                                                     sensor_noise=0.25,
                                                                                                     eta=8.2,
                                                                                                     kappa=0.25)

# Experiment with final state masking -> sensor noise 0.15 (beta = 0.85) and threshold 35.
pharmaceutical_example_with_multiple_init_states_modified_obs.run_pharma_example_with_dynamic_maskin(iter_num=3000,
                                                                                                     batch_size=100,
                                                                                                     V=1500,
                                                                                                     T=10, threshold=35,
                                                                                                     prior_compute_flag=0,
                                                                                                     exp_number=8,
                                                                                                     sensor_noise=0.15,
                                                                                                     eta=8.2,
                                                                                                     kappa=0.25)

# Experiment with final state masking -> sensor noise 0.25 (beta = 0.75) and threshold 35.
pharmaceutical_example_with_multiple_init_states_modified_obs.run_pharma_example_with_dynamic_maskin(iter_num=3000,
                                                                                                     batch_size=100,
                                                                                                     V=1500,
                                                                                                     T=10, threshold=35,
                                                                                                     prior_compute_flag=0,
                                                                                                     exp_number=9,
                                                                                                     sensor_noise=0.25,
                                                                                                     eta=8.2,
                                                                                                     kappa=0.25)

print("End of Pharma example for dynamic masking.")
