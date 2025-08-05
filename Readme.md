# Multi-Agent Deep Reinforcement Learning for Smart Grid Demand Response

This project applies **Multi-Agent Deep Q-Networks (DQN)** to optimize **energy consumption scheduling** in agricultural households. It addresses **peak load reduction** and **demand-response (DR)** incentives while maintaining user comfort.

Developed as part of an **end-of-study internship**, this work simulates realistic energy profiles and learns optimal load control policies for customers and an energy aggregator.


##  Project Objectives

- **Reduce peak demand** using demand-response strategies
- **Optimize scheduling** of shiftable and time-shiftable loads
- **Use reinforcement learning** (DQN) for decentralized and scalable energy management
- **Preserve comfort** using dissatisfaction-aware reward shaping


##  Architecture

- **Customer Agents**: Control individual household appliances
- **Aggregator Agent**: Broadcasts incentive signals to shape collective consumption
- **Environment**: Simulates grid, demand, consumption, and rewards
- **Experience Replay**: For stabilizing training in DQN
- **Baseline vs DR**: Compare system performance with and without DR


##  Project Structure

.
├── main.py # Main training and testing logic

├── agent_customer.py # Customer agent implementation (DQN)

├── agent_aggregator_dqn.py # Aggregator agent logic

├── environment.py # Environment simulating the smart grid

├── params.py # All configurable parameters

├── data_preprocessing.py # Clean and format consumption data

├── utils/ # Plotting, metrics, evaluation

├── save_files/ # Saved trained models

└── logs/ # TensorBoard logs


## Data

Real energy usage data is cleaned and processed into:
        
- **10-minute resolution**
- Aggregated per device: cleaning system, RO, cooling, lighting, etc.
- Three household profiles
- Time window: days 151 to 165 of the year

The preprocessed data is saved as:

/Data/10min_all_data_new.csv


## Configuration

All system settings are defined in `params.py`, including:

- Agents: `AGENT_IDS`, `NUM_AGENTS`, `NUM_RL_AGENTS`
- Devices: `DEVICE_CONSUMPTION`, `DISSATISFACTION_COEFFICIENTS`
- Environment: `TIME_STEPS_TRAIN`, `POWER_RATES`
- DQN: `BUFFER_SIZE`, `LEARNING_RATE_DQN`, `HIDDEN_LAYER_SIZE`
- Rewards: `RHO`, `CRITICAL_THRESHOLD`

You can customize these to fit other grid or building profiles.


## How to Run

### 1. Install Requirements
pip install -r requirements.txt

### 2. Train the Model
python main.py
Trained models and logs will be saved under /save_files/ and /logs/.

### 3. Test the Model
Evaluation is done over multiple days using:

test_average(path="save_files/MARL_IDR_6")

Key Results
Metric	Without DR	With DR
Peak Load (kW)	High	Lower
Mean Load (kW)	Similar	Balanced
Incentives Paid	0	Moderate
PAR (Peak-to-Average Ratio)	High	Reduced

