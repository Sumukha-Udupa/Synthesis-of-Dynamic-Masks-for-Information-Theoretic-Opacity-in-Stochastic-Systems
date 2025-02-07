import pharmaceutical_example_multi_init_states_no_masking
import pharmaceutical_example_with_multiple_init_states_final_state_masking

# Run experiments

# Experiment with no masking -> sensor noise 0.15 (beta = 0.85).
pharmaceutical_example_multi_init_states_no_masking.run_pharma_example_no_maskin(iter_num=1000, batch_size=100, V=100,
                                                                                 T=10, threshold=70,
                                                                                 prior_compute_flag=0, exp_number=4,
                                                                                 sensor_noise=0.15)
# Experiment with no masking -> sensor noise 0.25 (beta = 0.75).
pharmaceutical_example_multi_init_states_no_masking.run_pharma_example_no_maskin(iter_num=1000, batch_size=100, V=100,
                                                                                 T=10, threshold=70,
                                                                                 prior_compute_flag=0, exp_number=5,
                                                                                 sensor_noise=0.25)

# Experiment with final state masking -> sensor noise 0.15 (beta = 0.85)
pharmaceutical_example_with_multiple_init_states_final_state_masking.run_pharma_example_final_state_maskin(
    iter_num=1000, batch_size=100, V=100, T=10, threshold=70, prior_compute_flag=0, exp_number=6, sensor_noise=0.15)

# Experiment with final state masking -> sensor noise 0.25 (beta = 0.75)
pharmaceutical_example_with_multiple_init_states_final_state_masking.run_pharma_example_final_state_maskin(
    iter_num=1000, batch_size=100, V=100, T=10, threshold=70, prior_compute_flag=0, exp_number=7, sensor_noise=0.25)

print("End of Pharma example for no masking and final state masking.")
