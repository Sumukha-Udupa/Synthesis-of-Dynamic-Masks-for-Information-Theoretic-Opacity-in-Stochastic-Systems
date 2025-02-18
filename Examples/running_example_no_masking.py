import math

from setup_and_solvers.LP_for_nominal_policy import *
from setup_and_solvers.test_gradient_entropy_calculations_modified_obs import *
from setup_and_solvers.markov_decision_process import *


def run_experiment_with_no_masking(iter_num=1000, batch_size=100, V=100, T=3, eta=8.2, kappa=4.1, threshold=20,
                                   exp_number=1):
    logger.add("logs_for_examples/log_file_for_running_example_no_masking.log")

    logger.info("This is the log file for the running example with secret goal states s4 and s6 and no masking.")

    # Initial set-up for the MDP.

    states = [0, 1, 2, 3, 4, 5, 6]
    actions = ['a', 'b']

    prob = {
        'a': np.array([[0, (1 / 3), (1 / 3), (1 / 3), 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]),
        'b': np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
    }

    transitions = {
        (0, 'a'): [(1, 1 / 3), (2, 1 / 3), (3, 1 / 3)],
        (0, 'b'): [(0, 1)],
        # (1, 'a'): [(2, 0.5), (4, 0.5)],
        (1, 'a'): [(4, 1)],
        (1, 'b'): [(5, 1)],
        (2, 'a'): [(5, 1)],
        (2, 'b'): [(5, 1)],
        (3, 'a'): [(5, 1)],
        # (3, 'b'): [(3, 0.5), (6, 0.5)],
        (3, 'b'): [(6, 1)],
        (4, 'a'): [(4, 1)],
        (4, 'b'): [(4, 1)],
        (5, 'a'): [(5, 1)],
        (5, 'b'): [(5, 1)],
        (6, 'a'): [(6, 1)],
        (6, 'b'): [(6, 1)]
    }

    target = [4, 6]
    secret_goal_states = [4, 6]

    initial = [0]

    initial_dist = dict([])
    # Initial distribution is uniform over the initial states. This is a simplification for the current example
    for state in states:
        if state in initial:
            initial_dist[state] = 1 / len(initial)
        else:
            initial_dist[state] = 0

    # sensor setup
    sensors = {'R', 'G', 'P', 'B', 'NO'}

    # coverage sets
    setR = {1}
    setG = {3}
    setP = {4}
    setB = {6}
    setNO = {0, 2, 5}

    # masking actions
    masking_action = dict([])

    masking_action[0] = {'F'}  # 'F' is the no masking action.

    no_mask_act = 0

    # sensor noise
    sensor_noise = 0.15

    # sensor costs
    sensor_cost = dict([])
    sensor_cost['R'] = 10
    sensor_cost['G'] = 10
    sensor_cost['P'] = 10
    sensor_cost['B'] = 30
    sensor_cost['F'] = 0  # Cost for not masking.

    sensor_cost_normalization = sum(abs(cost) for cost in sensor_cost.values())

    # updating the sensor costs with normalized costs.
    for sens in sensor_cost:
        sensor_cost[sens] = sensor_cost[sens] / sensor_cost_normalization

    # normalized threshold.
    threshold = threshold / sensor_cost_normalization

    sensor_net = Sensor()
    sensor_net.sensors = sensors
    sensor_net.set_coverage('R', setR)
    sensor_net.set_coverage('G', setG)
    sensor_net.set_coverage('P', setP)
    sensor_net.set_coverage('B', setB)
    sensor_net.set_coverage('NO', setNO)

    sensor_net.jamming_actions = masking_action
    sensor_net.sensor_noise = sensor_noise
    sensor_net.sensor_cost_dict = sensor_cost

    agent_mdp = MDP(init=initial, actlist=actions, states=states, prob=prob, trans=transitions, init_dist=initial_dist,
                    goal_states=target)
    agent_mdp.get_supp()
    agent_mdp.gettrans()
    agent_mdp.get_reward()

    # Using the following agent policy.

    goal_policy = dict([])
    goal_policy[(0, 'a')] = 1
    goal_policy[(0, 'b')] = 0
    goal_policy[(1, 'a')] = 1
    goal_policy[(1, 'b')] = 0
    goal_policy[(2, 'a')] = 0.5
    goal_policy[(2, 'b')] = 0.5
    goal_policy[(3, 'a')] = 0
    goal_policy[(3, 'b')] = 1
    goal_policy[(4, 'a')] = 1
    goal_policy[(4, 'b')] = 0
    goal_policy[(5, 'a')] = 1
    goal_policy[(5, 'b')] = 0
    goal_policy[(6, 'a')] = 0
    goal_policy[(6, 'b')] = 1

    logger.debug("Goal policy:")
    logger.debug(goal_policy)

    hmm_p2 = HiddenMarkovModelP2(agent_mdp, sensor_net, goal_policy, secret_goal_states=secret_goal_states,
                                 no_mask_act=no_mask_act)

    # masking_policy_gradient = PrimalDualPolicyGradient(hmm=hmm_p2, iter_num=1000, V=10, T=10, eta=1.5, kappa=0.1, epsilon=threshold)
    # masking_policy_gradient.solver()

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=3,
    #                                                        eta=8.2,
    #                                                        kappa=0.25,
    #                                                        epsilon=threshold, sensor_cost_normalization=sensor_cost_normalization)

    # masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=1000, batch_size=100, V=100, T=3,
    #                                                        eta=8.2,
    #                                                        kappa=4.1,
    #                                                        epsilon=threshold,
    #                                                        sensor_cost_normalization=sensor_cost_normalization)

    masking_policy_gradient = PrimalDualPolicyGradientTest(hmm=hmm_p2, iter_num=iter_num, batch_size=batch_size, V=V,
                                                           T=T,
                                                           eta=eta,
                                                           kappa=kappa,
                                                           epsilon=threshold,
                                                           sensor_cost_normalization=sensor_cost_normalization,
                                                           exp_number=exp_number)

    masking_policy_gradient.solver()
