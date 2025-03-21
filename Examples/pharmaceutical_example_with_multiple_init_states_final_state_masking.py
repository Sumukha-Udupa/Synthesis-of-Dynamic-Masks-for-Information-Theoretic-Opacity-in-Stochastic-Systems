import math

from setup_and_solvers.gridworld_env_multi_init_states import *
from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_calculation_with_final_masking_policy import *


def run_pharma_example_final_state_maskin(iter_num=1000, batch_size=100, V=100, T=10, eta=8.2, kappa=0.25, threshold=70,
                                          prior_compute_flag=0,
                                          exp_number=4, sensor_noise=0.15):
    logger.add("logs_for_examples/log_E_threshold_70_final_state_masking_policy_sensor_noise_0.75.log")

    logger.info("This is the log file for the 6X6 gridworld with goal states 9, 20, 23 with no masking and planning "
                "gamma=0.9.")

    # Initial set-up for a 6x6 gridworld.
    ncols = 6
    nrows = 6
    target = [9, 20, 23]

    secret_goal_states = [9, 20, 23]
    obstacles = [17, 19]
    unsafe_u = [1, 13, 15, 35]
    non_init_states = [1, 25, 9, 14, 15, 17, 19, 23, 35]
    initial = {30, 12}

    initial_dist = dict([])
    # considering a single initial state.
    for state in range(36):
        if state in initial:
            initial_dist[state] = 1 / len(initial)
        else:
            initial_dist[state] = 0

    robot_ts = read_from_file_MDP_old('robotmdp.txt')

    # sensor setup
    sensors = {'A', 'B', 'C', 'D', 'NO'}

    setA = {3, 4, 9, 10}
    setB = {21, 22, 28}
    setC = {23, 29, 35}
    setD = {6, 7, 8, 12, 13, 14}
    setNO = {1, 2, 5, 11, 15, 16, 17, 19, 24, 25, 26, 30, 31, 32, 33, 34, 20, 27, 18, 0}

    # masking actions
    masking_action = dict([])

    masking_action[0] = {'A'}
    masking_action[1] = {'C'}
    masking_action[2] = {'F'}  # 'F' is the no masking action.

    no_mask_act = 2

    # sensor noise
    sensor_noise = sensor_noise

    # sensor costs
    sensor_cost = dict([])
    sensor_cost['A'] = 20
    sensor_cost['B'] = 25
    sensor_cost['C'] = 15
    sensor_cost['D'] = 5
    sensor_cost['F'] = 0  # Cost for not masking.

    # Define a threshold for sensor masking.
    threshold = threshold

    sensor_cost_normalization = sum(abs(cost) for cost in sensor_cost.values())

    # updating the sensor costs with normalized costs.
    for sens in sensor_cost:
        sensor_cost[sens] = sensor_cost[sens] / sensor_cost_normalization

    # normalized threshold.
    threshold = threshold / sensor_cost_normalization

    sensor_net = Sensor()
    sensor_net.sensors = sensors

    sensor_net.set_coverage('A', setA)
    sensor_net.set_coverage('B', setB)
    sensor_net.set_coverage('C', setC)
    sensor_net.set_coverage('D', setD)
    sensor_net.set_coverage('NO', setNO)

    sensor_net.jamming_actions = masking_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    agent_gw_1 = GridworldGui(initial, nrows, ncols, robot_ts, target, obstacles, unsafe_u, initial_dist)
    agent_gw_1.mdp.get_supp()
    agent_gw_1.mdp.gettrans()
    agent_gw_1.mdp.get_reward()
    agent_gw_1.draw_state_labels()

    goal_policy_file = "goal_policy.pickle"

    # Load from goal policy.
    with open(goal_policy_file, "rb") as f:
        goal_policy = pickle.load(f)

    if prior_compute_flag == 1:

        # Computing the prior entropy.
        # Monte carlo simulation to obtain the approximate probability of being in the final state in T=10.

        total_prior = 0

        for iterations in range(1000):
            prior_entropy = 0
            counter = 0
            horizon = 12
            final_state_goal_state = 0
            final_state_not_goal_state = 0

            while counter <= 1000:
                new_init_state = random.choice(list(agent_gw_1.mdp.init))

                for i in range(horizon):
                    weights_list = list()
                    for action in agent_gw_1.mdp.actlist:
                        weights_list.append(goal_policy[(new_init_state, action)])
                    action_to_play = random.choices(agent_gw_1.actlist, weights_list)[0]

                    post_states = list(agent_gw_1.mdp.suppDict[(new_init_state, action_to_play)])
                    states_weights_list = list()
                    for st in post_states:
                        states_weights_list.append(agent_gw_1.mdp.trans[new_init_state][action_to_play][st])

                    next_state = random.choices(post_states, states_weights_list)[0]
                    new_init_state = next_state

                if new_init_state in target:
                    final_state_goal_state += 1
                else:
                    final_state_not_goal_state += 1

                counter += 1

            probability_of_raching_final_state = final_state_goal_state / 1001
            # print(f"Probability of reaching goal state within T steps: {probability_of_raching_final_state}")
            prior_entropy = probability_of_raching_final_state * math.log2(probability_of_raching_final_state) + (
                    (1 - probability_of_raching_final_state) * math.log2(1 - probability_of_raching_final_state))

            # print(f"Prior entropy: {-prior_entropy}")

            # prior_list.append(-prior_entropy)
            total_prior += (-prior_entropy)

        print(f"Mean prior entropy = {total_prior / 1000}")
        # print(f"Final state not goal state = {final_state_not_goal_state}")

        logger.debug(f"Mean prior entropy = {total_prior / 1000}.")

    hmm_p2 = HiddenMarkovModelP2(agent_gw_1.mdp, sensor_net, goal_policy, secret_goal_states=secret_goal_states,
                                 no_mask_act=no_mask_act)

    # masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
    # masking_policy_gradient.solver()

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=3000, batch_size=100, V=100, T=10,
    #                                                        eta=3.2,
    #                                                        kappa=0.25,
    #                                                        epsilon=threshold)

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=10,
    #                                                        eta=8.2,
    #                                                        kappa=0.25,
    #                                                        epsilon=threshold)

    masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=iter_num, batch_size=batch_size, V=V,
                                                           T=T,
                                                           eta=eta,
                                                           kappa=kappa,
                                                           epsilon=threshold, exp_number=exp_number)

    masking_policy_gradient.solver()
