import numpy as np
import pandas as pd
import random 
from load_demand import load_requests, load_day, get_device_demands, load_baselines, get_peak_demand
from params import RHO, CUSTOMER_ACTION_SIZE, TRAINING_START_DAY, TRAINING_END_DAY, INCENTIVE_RATES,AGENT_IDS,AGGREGATOR_ACTION_SIZE, TIME_STEPS_TRAIN,TIME_STEPS_TEST, DEVICE_CONSUMPTION, DISSATISFACTION_COEFFICIENTS,DEVICE_NON_INTERRUPTIBLE, RHO_AGGREGATOR, DEVICES, CRITICAL_THRESHOLD_RELATIVE, MAX_TOTAL_DEMAND, MAX_INCENTIVE, RHO_COMMON, POWER_RATES, BASELINE_START_DAY,NUM_AGENTS, DISSATISFACTION_COEFFICIENTS_STD, DISSATISFACTION_COEFFICIENTS_MIN, TESTING_START_DAY


TESTING_START_DAY = 162
TESTING_END_DAY = 165
NUM_RL_AGENTS = 3   


def knapsack(values, weights, capacity):
    """ Schedule the devices based on their dissatisfaction (the values) and their consumption (the weights). The
    customer selected a fraction of demand as consumption (capacity). This function brute-forces the optimal knapsack
    solution. """
    non_zero_values = np.nonzero(values)[0]
    n = len(non_zero_values)
    max_value = 0
    max_weight = capacity
    max_actions = np.zeros(len(values), dtype=bool)

    if n == 0:
        return 0, 0, max_actions

    for i in range(2 ** n):
        actions = np.array([int(x) for x in list(f'{i:b}'.zfill(n))], dtype=bool)
        action_indices = non_zero_values[actions]
        value = values[action_indices].sum()
        weight = weights[action_indices].sum()
        if (weight <= capacity and value > max_value) or (weight <= max_weight and value == max_value):
            max_value = value
            max_weight = weight
            max_actions = np.zeros(len(values), dtype=bool)
            max_actions[action_indices] = True

    return max_value, max_weight, max_actions



def sample_action_customer():
    """ Sample a random action for the customer. """
    return np.random.randint(0, CUSTOMER_ACTION_SIZE)


def sample_action_aggregator():
    """ Sample a random action for the aggregator. """
    return np.random.randint(0, AGGREGATOR_ACTION_SIZE)





pv_power_csv_data = 'C:\\Users\\saoudi\\Desktop\\RL_Algo\\Data\\pv_data_10min.csv'

def knapsack_ensemble(values, weights, capacity, dissatisfaction_coefficients):
    max_values = []
    max_weights = []
    max_actionss = []
    rates = []

    for rate in POWER_RATES[1:]:
        adjusted_values = values.copy()
        adjusted_weights = weights.copy()

        # Calculate the dissatisfaction value for each device using the provided coefficients
        dissatisfaction_values = dissatisfaction_coefficients * np.square(weights)
        adjusted_values -= dissatisfaction_values

        # Calculate the reduced weight for each device using the power rate
        reduced_weights = weights * rate
        adjusted_weights -= reduced_weights

        # Perform knapsack optimization for the adjusted values and weights
        max_value, max_weight, max_actions = knapsack(adjusted_values, adjusted_weights, capacity)
        max_values.append(max_value)
        max_weights.append(max_weight)
        max_actionss.append(max_actions)
        rates.append(rate)

    sorted_values = sorted(zip(max_values, max_weights, max_actionss, rates), key=lambda elem: (-elem[0], elem[1]))
    return sorted_values[0]


class Environment:
    """ The AggregatorAgent and the CustomerAgents interact with the Environment. The Environment controls input of
    device requests and demands. It schedules devices with the knapsack algorithm for the CustomerAgents. Finally,
    it calculates the rewards."""

    def __init__(self, data_ids, heterogeneous=False, baseline=False):
        self.data_ids = data_ids
        self.episode = 0
        self.df = load_requests()
        self.heterogeneous = heterogeneous
        self.baseline = baseline
        self.pv_power_data = pd.read_csv(pv_power_csv_data)['power'].to_numpy()
        self.dissatisfaction_coefficients = np.full((len(data_ids), len(DEVICES)), DISSATISFACTION_COEFFICIENTS)

        # self.dissatisfaction_coefficients = np.full((len(data_ids), len(DEVICES)), DISSATISFACTION_COEFFICIENTS)
        if heterogeneous:
            dissatisfaction_coefficients = np.random.normal(
                loc=DISSATISFACTION_COEFFICIENTS, scale=DISSATISFACTION_COEFFICIENTS_STD, size=(NUM_AGENTS, len(DEVICES)))
            self.dissatisfaction_coefficients = np.maximum(DISSATISFACTION_COEFFICIENTS_MIN, dissatisfaction_coefficients)




    def reset(self, day=None, max_steps=TIME_STEPS_TRAIN):
        self.day = day
        import random

        self.day = day
        if day is None:
            training_day_range = (TRAINING_START_DAY, TESTING_START_DAY)
            testing_day_range = (TESTING_START_DAY, TRAINING_END_DAY)
            selected_range = random.choice([training_day_range, testing_day_range])
            low, high = selected_range

        #if low >= high:
            #raise ValueError("Invalid day range selected.")

            #self.day = np.random.randint(low, high)

        self.curr_step = 0
        self.episode += 1
        self.done = False
        self.max_steps = max_steps      

        # New PV-related parameters
        self.pv_power = np.zeros(max_steps)
        self.grid_power = np.zeros(max_steps)
        self.power_sources = np.zeros((max_steps, len(self.data_ids)))  # 0 for grid, 1 for PV



        # Customer agent params
        self.demand = np.zeros((max_steps, len(self.data_ids)))
        self.non_shiftable_load = np.zeros((max_steps, len(self.data_ids)))
        self.requests_new = np.zeros((max_steps, len(self.data_ids), len(DEVICES)), dtype=bool)         # Incoming requests from PecanStreet
        self.request_loads = np.zeros((max_steps, len(self.data_ids), len(DEVICES)))                        # The load in kW for open requests
        self.requests_started = np.zeros((max_steps, len(self.data_ids), len(DEVICES)), dtype=bool)     # Request for Non-interruptible devices that have been started but are still running
        self.requests_open = np.zeros((max_steps, len(self.data_ids), len(DEVICES)))                    # How many time steps are still unfulfilled for a request for a device (the length of the request)
        self.requests_delayed = np.zeros((max_steps, len(self.data_ids), len(DEVICES)))                 # How many time steps a device has been delayed

        self.possible_actions = np.zeros((max_steps, len(self.data_ids)))      # The devices scheduled by knapsack in each time step
        self.power_rates = np.zeros((max_steps, len(self.data_ids)))      # The devices scheduled by knapsack in each time step
        self.request_actions = np.zeros((max_steps, len(self.data_ids), len(DEVICES)), dtype=bool)      # The devices scheduled by knapsack in each time step
        self.ac_rates = np.zeros((max_steps, len(self.data_ids)))
        self.consumptions = np.zeros((max_steps, len(self.data_ids)))                                       # Total consumtpion by each agent in each time step                                     # Total consumtpion by each agent in each time step
        self.incentive_received = np.zeros((max_steps, len(self.data_ids)))
        self.rewards_customers = np.zeros((max_steps, len(self.data_ids)))
        self.dissatisfaction = np.zeros((max_steps, len(self.data_ids), len(DEVICES)))
        self.customer_reward_matrix = np.zeros((max_steps, len(INCENTIVE_RATES), len(self.data_ids), len(POWER_RATES)))
        self.aggregator_reward_matrix = np.zeros((max_steps, len(INCENTIVE_RATES)))

        # Aggregator agent params
        self.incentives = np.zeros(max_steps)
        self.rewards_aggregator = np.zeros(max_steps)

        # Demand data params
        self.day_df = load_day(self.df, self.day, max_steps)
        self.baselines = load_baselines()
        self.set_demands(self.curr_step)
        # self.capacity_threshold = CRITICAL_THRESHOLD
        self.capacity_threshold = get_peak_demand(self.day_df) * CRITICAL_THRESHOLD_RELATIVE



    def last_customer(self, agent_id):
        """ The CustomerAgent can call this method to receive the previous reward and the next observation.
        The observation consists of the state of the household appliances and the offered incentive. The state of the
        household appliances is defined as an integer, 0 for no request or requests for non-interruptible devices that
        have been started, 1 for a new request and > 1 if the request has been delayed. """
        incentive = self.incentives[self.curr_step]
        if self.day is not None and BASELINE_START_DAY is not None:
            baseline = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
        else:
            # Handle the case where either self.day or BASELINE_START_DAY is None
            baseline = 0  # Provide a default value if needed
            #baseline = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
        new_requests = self.requests_new[self.curr_step][agent_id]
        started_requests = self.requests_started[self.curr_step][agent_id]
        open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
        delays = self.requests_delayed[self.curr_step][agent_id]
        new_delays = new_requests + delays
        open_delays = new_delays * np.invert(started_requests)
        #ac_consumption = self.request_loads[self.curr_step][agent_id][0]
        non_shiftable = self.non_shiftable_load[self.curr_step][agent_id]
        non_interruptible = (np.logical_and(open_requests, started_requests) * DEVICE_CONSUMPTION).sum()
        #observation = np.array(np.concatenate(([ac_consumption], open_delays[1:], [non_shiftable + non_interruptible, incentive, baseline])))
        reward = self.rewards_customers[self.curr_step][agent_id]
        done = self.done
        power_source = self.power_sources[self.curr_step][agent_id]
        observation = np.array(np.concatenate((open_delays[1:], [non_shiftable + non_interruptible, incentive, baseline, power_source])))
        return observation, reward, done, None
    



    def last_aggregator(self):
        """ The AggregatorAgent can call this method to receive the previous reward and the next observation.
        The observation only contains the total demand of the customer together. """
        total_demand = self.get_total_demand(self.curr_step)
        # total_demand = self.baselines[:, self.day - TRAINING_START_DAY, self.curr_step].sum(axis=0)
        threshold = self.capacity_threshold
        reduction = self.get_total_reduction(self.curr_step-1 if self.curr_step > 0 else 0)
        observation = np.array([total_demand, threshold, reduction])
        # Check if the list is not empty before accessing its last element
        if len(self.rewards_aggregator) > 0:
            reward = self.rewards_aggregator[-1]  # Access the last element
        else:
            # Handle the case when the list is empty (no rewards available)
            reward = 0  # Provide a default value or handle it according to your logic


        #reward = self.rewards_aggregator[self.curr_step]
        done = self.done
        return observation, reward, done, None
    
    def act_aggregator(self, action):
        """ Apply the action selected by the AggregatorAgent.
        The agent selects the incentive rate and sends it to the environment. The environment saves the incentive to
        send it to the CustomerAgents later. """
        incentive_rate = INCENTIVE_RATES[action]
        # Define the maximum valid index (one less than the size of the array)
        max_valid_index = len(self.incentives) - 1

        # Clip or limit self.curr_step to stay within valid index range
        self.curr_step = min(self.curr_step, max_valid_index)

        # Now you can safely assign the incentive_rate to the array
        self.incentives[self.curr_step] = incentive_rate
        
        #self.incentives[self.curr_step] = incentive_rate
        if self.baseline:
            print('Computing baseline step:', self.curr_step)
            self.compute_best_responses()
            self.incentives[self.curr_step] = np.argmax(self.aggregator_reward_matrix[self.curr_step])


    def act(self, agent_id, action):
        """ Apply the action selected by a CustomerAgent.
        The agent selects a power rate and sends it to the environment. Based on this power rate this method calls the
        knapsack algorithm and determines the devices scheduled for this time step. Afterwards this method calculates
        the new state of the appliances taking device-specific constraints into account. """
        # Get power rate and incentive rate
        incentive_rate = self.incentives[self.curr_step]
        if self.day is not None and BASELINE_START_DAY is not None:
            baseline_day = self.day - BASELINE_START_DAY
            if baseline_day >= 0 and baseline_day < len(self.baselines[agent_id]):
                baseline_demand = self.baselines[agent_id][baseline_day][self.curr_step]
            else:
                baseline_demand = 0  # Provide a default value if baseline_day is out of bounds
        else:
            baseline_demand = 0  # Provide a default value if self.day or BASELINE_START_DAY is None

        #baseline_demand = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
        #car_index = DEVICES.index('car')
        #ac_index = DEVICES.index('air')
        power_rate = POWER_RATES[action]
        if self.baseline:
            power_rate = POWER_RATES[np.argmax(self.customer_reward_matrix[self.curr_step][int(incentive_rate)][agent_id])]

        # Get requests and demands
        started_requests = self.requests_started[self.curr_step][agent_id]
        new_requests = self.requests_new[self.curr_step][agent_id]
        open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
        delayed_requests = self.requests_delayed[self.curr_step][agent_id]
        selectable_requests = np.logical_and(open_requests, np.invert(started_requests))
        non_interruptible_requests = np.logical_and(open_requests, started_requests)
        non_interruptible_demand = (non_interruptible_requests * DEVICE_CONSUMPTION).sum()
        non_shiftable_demand = self.non_shiftable_load[self.curr_step][agent_id]
        device_consumptions = selectable_requests * DEVICE_CONSUMPTION
        #device_consumptions[ac_index] = self.request_loads[self.curr_step][agent_id][ac_index]

        # Brute-force knapsack
        dissatisfaction_values = self.dissatisfaction_coefficients[agent_id] * np.square(delayed_requests + 1)
        #dissatisfaction_values[ac_index] = self.dissatisfaction_coefficients[agent_id][ac_index] * np.square(device_consumptions[ac_index])
        device_values = dissatisfaction_values * selectable_requests
        shiftable_demand = (selectable_requests * device_consumptions).sum()
        capacity = power_rate * shiftable_demand
        result = knapsack_ensemble(device_values, device_consumptions, capacity, self.dissatisfaction_coefficients[agent_id])
        value, weight, actions, rate = result
        # Update power rate based on the selected rate
        power_rate = POWER_RATES[action] if not self.baseline else rate

        #value, weight, actions = knapsack_ensemble(device_values, device_consumptions, capacity, self.dissatisfaction_coefficients[agent_id])
        delayed_devices = np.invert(actions) * selectable_requests
        dissatisfaction = device_values.sum() - value

        # Calculate received incentive
        consumption = weight + non_interruptible_demand + non_shiftable_demand
        energy_diff = baseline_demand - consumption
        # energy_diff = self.demand[self.curr_step, agent_id] - consumption
        # energy_diff = shiftable_demand - weight
        incentive_received = incentive_rate * max(0, energy_diff)

        # Calculate reward
        incentive_term = RHO * incentive_received
        dissatisfaction_term = (1 - RHO) * -dissatisfaction
        reward = incentive_term + dissatisfaction_term
        # reward = max(MINIMUM_CUSTOMER_REWARD, incentive_term + dissatisfaction_term)

        # Save selected devices, energy consumption and received incentive
        fulfilled_requests = np.logical_or(actions, non_interruptible_requests)
        self.possible_actions[self.curr_step][agent_id] = np.count_nonzero(selectable_requests)
        self.request_actions[self.curr_step][agent_id] = fulfilled_requests
        self.consumptions[self.curr_step][agent_id] = consumption
        self.incentive_received[self.curr_step][agent_id] = incentive_received
        self.power_rates[self.curr_step][agent_id] = power_rate if selectable_requests.any() else 1
        #self.ac_rates[self.curr_step][agent_id] = ac_rate * actions[ac_index] if selectable_requests[ac_index] else 1
        self.dissatisfaction[self.curr_step][agent_id] = device_values * delayed_devices
        #self.dissatisfaction[self.curr_step][agent_id][ac_index] = self.dissatisfaction_coefficients[agent_id][ac_index] * np.square((1 - ac_rate) * device_consumptions[ac_index])

        # Update parameters for use in the next time step
        if self.curr_step < self.max_steps - 1:
            #started_non_interruptibles = actions * DEVICE_NON_INTERRUPTIBLE
            #num_devices = len(DEVICES)
            #non_interruptible_requests = np.array([self.requests_new[self.curr_step, agent_id] & DEVICE_NON_INTERRUPTIBLE[:num_devices] for agent_id in AGENT_IDS[:NUM_RL_AGENTS]])

            #started_non_interruptibles = actions * non_interruptible_requests
            


            open_requests_next = open_requests - fulfilled_requests

            self.rewards_customers[self.curr_step + 1][agent_id] = reward
            self.requests_open[self.curr_step + 1][agent_id] = open_requests_next
            #self.requests_started[self.curr_step + 1][agent_id] = started_non_interruptibles + non_interruptible_requests
            self.requests_delayed[self.curr_step + 1][agent_id] = delayed_requests + delayed_devices
            #self.requests_delayed[self.curr_step + 1][agent_id][started_non_interruptibles] = 0

            # If all requested time slots for the EV are fulfilled reset the delay
            #if open_requests_next[car_index] == 0:
                #self.requests_delayed[self.curr_step + 1][agent_id][car_index] = 0

            # AC has no delay
            #self.requests_delayed[self.curr_step + 1][agent_id][ac_index] = 0
            #self.requests_open[self.curr_step + 1][agent_id][ac_index] = 0

        
        # Update PV and grid power sources
        if self.pv_power[self.curr_step] >= consumption:
            self.power_sources[self.curr_step][agent_id] = 1  # PV power
        else:
            self.power_sources[self.curr_step][agent_id] = 0  # Grid power



        



    def step(self):
        """ This method is called at the end of a time step.
    If it was not the final time step, the reward for the aggregator is calculated and demands for the next
    time step are retrieved. """
        self.curr_step += 1
        self.done = self.curr_step == self.max_steps
        if not self.done:
            self.reward_aggregator()

            if self.curr_step <= TIME_STEPS_TEST:
                step = self.curr_step % TIME_STEPS_TEST  # Ensure step is within [0, TIME_STEPS_TEST - 1]
            else:
                step = self.curr_step % TIME_STEPS_TRAIN  # Ensure step is within [0, TIME_STEPS_TRAIN - 1]

            self.set_demands(step)
            self.set_incentive()


        



    def reward_aggregator(self):
        """ Calculate the reward for the aggregator.
        The reward consists of a consumption term, indicating how much the total consumption exceeds the threshold, and
        an incentive term, indicating how much each agent received on average. The term is normalized instead of taking
        the total, because the number of RL agents may differ. """
        consumption_term = max(0, self.get_total_consumption(self.curr_step - 1) - self.capacity_threshold)
        incentive_term = self.incentive_received[self.curr_step - 1].sum() / 100
        reward = - RHO_AGGREGATOR * consumption_term - (1 - RHO_AGGREGATOR) * incentive_term
        self.rewards_aggregator[self.curr_step] = reward

        customer_reward = self.rewards_customers[self.curr_step]
        customer_bonus = RHO_COMMON * customer_reward - (1 - RHO_COMMON) * consumption_term
        self.rewards_customers[self.curr_step] = customer_bonus  



    def load_pv_power_data(self, csv_file_path):
        csv_file_path = 'C:\\Users\\saoudi\\Desktop\\RL_Algo\\Data\\pv_data_10min.csv'
        # Load PV power data from CSV file into a NumPy array
        pv_power_df = pd.read_csv(csv_file_path)
        self.pv_power_data = pv_power_df['power'].to_numpy()

    def get_pv_power(self, step):
        if self.pv_power_data is not None and 0 <= step < len(self.pv_power_data):
            return self.pv_power_data[step]
        else:
            return 0.0  # Return 0.0 if step is out of range or pv_power_data is not available   







    def set_demands(self, step):
        """ Retrieve the demands per customer and per device for the specified time step from the demands DataFrame.
    If the demand is larger than a certain threshold the device is considered requested by the user. The actual
    load in kW that is requested for the device is fixed, except for the total non-shiftable devices. """
        df = get_device_demands(self.day_df, self.data_ids, self.day, self.curr_step)
        # Convert 'step' to an integer
        step = int(step)
        # Initialize 'non_shiftable' with a default value
        non_shiftable = 0.0
        total = 0.0
        requests = np.zeros(len(DEVICES))


        # Check if 'step' exists in the DataFrame's index
        if step in df.index:
            non_shiftable = df.loc[step, 'non-shiftable']
            total = df.loc[step, 'total']
            requests = df.loc[step, DEVICES].to_numpy()

    # Check if 'non_shiftable' is NaN and set it to 0.0 if it is
        if np.isnan(non_shiftable):
            non_shiftable = 0.0

    # Check if 'total' is NaN and set it to 0.0 if it is
        if np.isnan(total):
            total = 0.0

    # Check if 'requests' array is empty (contains NaN values) and set it to zeros if it is
        if np.isnan(requests).all():
            requests = np.zeros(len(DEVICES))

    # Rest of your code remains the same
    # ...

        else:
    # Handle the case where 'step' does not exist in the index by setting default values
            non_shiftable = 0.0
            total = 0.0
            requests = np.zeros(len(DEVICES))



    #self.grid_power[self.curr_step] = self.get_grid_power(self.curr_step)

        request_new = np.greater(requests, 0)
        self.non_shiftable_load[self.curr_step] = non_shiftable
        self.requests_new[self.curr_step] = request_new
        self.request_loads[self.curr_step] = requests
        self.demand[self.curr_step] = total  
        # Calculate PV power
        self.pv_power[self.curr_step] = self.get_pv_power(self.curr_step)





    def get_total_demand(self, step=None):
        """ Sum the demands of the customer agents. """
        if step is None:
            return self.demand.sum(axis=1)
        if step == 9:
            # Handle the case for step 144
            return self.demand[-1].sum()
        return self.demand[step].sum()

    def get_total_consumption(self, step=None):
        """ Sum the consumptions of the customer agents. """
        if step is None:
            return self.consumptions.sum(axis=1)
        if step == 9:
            # Handle the case for step 144
            return self.consumptions[-1].sum()
        return self.consumptions[step].sum()


    def get_total_reduction(self, step=None):
        if step is None:
            return self.get_total_demand() - self.get_total_consumption()
        return self.get_total_demand(step) - self.get_total_consumption(step)

    def set_incentive(self):
        """ A simple heuristic for calculating incentives without the aggregator as an agent.
        The incentive is a linear relation to the demand exceeding the capacity. """
        total_demand = min(MAX_TOTAL_DEMAND, self.get_total_demand(self.curr_step))
        demand_range = MAX_TOTAL_DEMAND - self.capacity_threshold
        demand_overflow = max(0, total_demand - self.capacity_threshold)
        incentive = np.ceil((demand_overflow / demand_range) * MAX_INCENTIVE)
        self.incentives[self.curr_step] = incentive

    def set_baseline(self):
        """ Average the pre-computed baseline with the consumption of the last time step for a more accurate result. """
        baseline_demand = self.baselines[:, self.day - BASELINE_START_DAY, self.curr_step]
        new_baseline_demand = (baseline_demand + self.consumptions[self.curr_step - 1]) / 2
        self.baselines[:, self.day - BASELINE_START_DAY, self.curr_step] = new_baseline_demand

    def compute_best_responses(self):
        rewards = np.zeros((AGGREGATOR_ACTION_SIZE, len(self.data_ids), len(POWER_RATES)))
        profits = np.zeros((AGGREGATOR_ACTION_SIZE, len(self.data_ids), len(POWER_RATES)))
        consumptions = np.zeros((AGGREGATOR_ACTION_SIZE, len(self.data_ids), len(POWER_RATES)))
        best_profits = np.zeros((AGGREGATOR_ACTION_SIZE, len(self.data_ids)))
        best_consumptions = np.zeros((AGGREGATOR_ACTION_SIZE, len(self.data_ids)))
        for i, incentive_rate in enumerate(INCENTIVE_RATES):
            for agent_id in range(len(self.data_ids)):
                for j, power_rate in enumerate(POWER_RATES):
                    baseline_demand = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
                    started_requests = self.requests_started[self.curr_step][agent_id]
                    new_requests = self.requests_new[self.curr_step][agent_id]
                    open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
                    delayed_requests = self.requests_delayed[self.curr_step][agent_id]
                    selectable_requests = np.logical_and(open_requests, np.invert(started_requests))
                    non_interruptible_requests = np.logical_and(open_requests, started_requests)
                    non_interruptible_demand = (non_interruptible_requests * DEVICE_CONSUMPTION).sum()
                    non_shiftable_demand = self.non_shiftable_load[self.curr_step][agent_id]
                    device_consumptions = selectable_requests * DEVICE_CONSUMPTION
                    device_consumptions[0] = self.request_loads[self.curr_step][agent_id][0]
                    dissatisfaction_values = self.dissatisfaction_coefficients[agent_id] * np.square(delayed_requests + 1)
                    dissatisfaction_values[0] = self.dissatisfaction_coefficients[agent_id][0] * np.square(device_consumptions[0])
                    device_values = dissatisfaction_values * selectable_requests
                    shiftable_demand = (selectable_requests * device_consumptions).sum()
                    capacity = power_rate * shiftable_demand
                    value, weight, actions, ac_rate = knapsack_ensemble(device_values, device_consumptions, capacity, self.dissatisfaction_coefficients[agent_id])
                    dissatisfaction = device_values.sum() - value
                    consumption = weight + non_interruptible_demand + non_shiftable_demand
                    profit = incentive_rate * max(0, baseline_demand - consumption)
                    reward = profit - dissatisfaction
                    rewards[i, agent_id, j] = reward
                    profits[i, agent_id, j] = profit
                    consumptions[i, agent_id, j] = consumption

                best_profits[i][agent_id] = profits[i][agent_id][np.argmax(rewards[i, agent_id])]
                best_consumptions[i][agent_id] = consumptions[i][agent_id][np.argmax(rewards[i, agent_id])]

        aggregator_rewards = - best_profits.sum(axis=1) / 100 - np.maximum(0, best_consumptions.sum(axis=1) - self.capacity_threshold)
        self.customer_reward_matrix[self.curr_step] = rewards
        self.aggregator_reward_matrix[self.curr_step] = aggregator_rewards     