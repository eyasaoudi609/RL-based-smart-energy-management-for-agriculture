import numpy as np

# Agent params
AGENT_IDS = [661, 1642, 2335]
NUM_AGENTS = len(AGENT_IDS)
NUM_RL_AGENTS = 3              # Number of agents that are trained. The others are dummy agents (always choosing power rate 1.0)
# TRAINING_START_DAY = 182        # Day of the year defining the start of the training period
# TRAINING_END_DAY = 244          # Day of the year defining the end of the training period
TRAINING_START_DAY = 151        # Day of the year defining the start of the training period
TRAINING_END_DAY = 162          # Day of the year defining the end of the training period
TESTING_START_DAY = 162
TESTING_END_DAY = 165
BASELINE_START_DAY = 151
TRAINING_PERIOD = TRAINING_END_DAY - TRAINING_START_DAY     # The length of the training period
#print(TRAINING_PERIOD)
TESTING_PERIOD = TESTING_END_DAY - TESTING_START_DAY     # The length of the training period
#print(TESTING_PERIOD)

# RL params
EPSILON = 0.1                   # Fixed epsilon
EPSILON_START = 1.0             # Epsilon start when using epsilon decay
EPSILON_MIN = 0.01              # Epsilon minimum when using epsilon decay
EPSILON_DECAY = 0.999           # Epsilon is multiplied by this decay every step (depends on number of episodes)
DISCOUNT_RATE = 0.9              # Discount rate (gamma) of the Q-learning algorithm
EPISODES = 10                 # 5000 Number of episodes to train

# DQN params
BUFFER_SIZE = 10000             # The maximum number of SARS samples in the replay buffer
BATCH_SIZE = 32                 # The batch size for training of the Q-network
LEARNING_RATE_DQN = 0.001      # The learning rate for training the Q-network
TAU = 0.001                     # The soft-update parameter for updating the target network
TRAINING_INTERVAL = 16           # After so many steps the agent performs a training update on the network
REPLACE_TARGET_INTERVAL = 50    # After so many episodes the target network is replaced
HIDDEN_LAYER_SIZE = 32          # Size of the hidden layers

# Environment params
#TIME_STEPS_TRAIN = 144                       # Number of time steps per episode in training
#TIME_STEPS_TEST = 144  
DEVICE_LOADS = 9                # Number of device loads
LOAD_TIME_INTERVAL_MINUTES = 10  # Time interval between each load in minutes
# Calculate the total simulation time for each episode (in minutes)
TOTAL_SIMULATION_TIME_MINUTES = DEVICE_LOADS * LOAD_TIME_INTERVAL_MINUTES
# Calculate the total number of time steps per episode
TIME_STEPS_TRAIN = int(TOTAL_SIMULATION_TIME_MINUTES / LOAD_TIME_INTERVAL_MINUTES)
TIME_STEPS_TEST = TIME_STEPS_TRAIN                     # Number of time steps per episode in testing is 9

POWER_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # Set of actions (fraction of the demand the agent uses)
CUSTOMER_ACTION_SIZE = len(POWER_RATES)     # Number of actions for the customers
CUSTOMER_STATE_SIZE = 12 #8                 # Number of state variables for the customers
RHO = 0.5                       # Weight of the incentive term (the weight of the dissatisfaction term is 1-RHO)
RHO_COMMON = 1.0
CRITICAL_THRESHOLD = 70         # Above this threshold for the total demand the agents receive incentives > 0
CRITICAL_THRESHOLD_RELATIVE = 0.8         # Above this threshold for the total demand the agents receive incentives > 0
MAX_TOTAL_DEMAND = 110          # Incentives will not increase when total demand is higher than this value
MINIMUM_CUSTOMER_REWARD = -10  # Minimum reward to avoid too large negative rewards

# Aggregator params
RHO_AGGREGATOR = 0.5              # Weight of the consumption term (the weight of the incentive term is 1-RHO)
INCENTIVE_RATES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]    # Set of actions (incentives)
AGGREGATOR_ACTION_SIZE = len(INCENTIVE_RATES)   # Number of actions for the aggregator
AGGREGATOR_STATE_SIZE = 3      # Number of state variables for the aggregator
MAX_INCENTIVE = 10
DISCOUNT_RATE_AGGREGATOR = 0.9  # Discount rate for the aggregator

# Device params
DEVICES = ['cleaning_system', 'cooling_system', 'lightning_system', 'milk_pump', 'RO','vacuum_pump' , 'osmose', 'coldroom','water_pump']
#DEVICE_CONSUMPTION = np.array([45.0, 95.0, 10.0, 3.0, 0.2, 147.0, 19.0, 2.5, 14.0  ])  
DEVICE_CONSUMPTION = np.array([3.0, 2.0, 2.5, 4.0, 1.0, 2.0, 2.0, 2.5, 4.0, ])                           # Fixed consumption of the devices in kW
# DISSATISFACTION_COEFFICIENTS = np.array([3.0, 0.04, 0.1, 0.06, 0.2])                  # Delay coefficients
# DISSATISFACTION_COEFFICIENTS = np.array([10.0, 0.2, 0.4, 0.3, 0.6])                  # Delay coefficients
DISSATISFACTION_COEFFICIENTS = np.array([6.0, 0.2, 0.4, 0.1, 0.3, 0.05, 0.2, 0.1, 0.4])  # Delay coefficients
DISSATISFACTION_COEFFICIENTS_STD = np.array([2.0, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1])  # Delay coefficients
DISSATISFACTION_COEFFICIENTS_MIN = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
DEVICE_NON_INTERRUPTIBLE = np.array([False, False, True, True, True, True, True, True, True])  # If the device is non-interruptible
