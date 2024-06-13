## Automated Trading using Reinforcement Learning

This project focuses on predicting optimal times for traders to open and close positions in the forex market, whether to buy or sell.

### Training Design
The proposed training design involves two distinct procedures:

1. **LSTM Model Training**: The LSTM model is trained to predict actions—buy, hold, or sell—based on input data. During training, the model iteratively updates its weights to improve prediction accuracy.

2. **Reinforcement Learning (RL) Training**: The RL component updates the Q-table, which is used for decision-making. Instead of using random actions to update the Q-table, this research leverages the predicted actions from the trained LSTM model.

### Environment
The Environment component is responsible for executing the positions based on actions provided by either the LSTM model or the RL Agent. It calculates rewards as feedback, indicating the success of the actions taken.

### Testing Process
During testing, the agent utilizes the Q-table generated from the training process to make decisions based on the state inputs received from the environment. The Q-table, optimized with information from the LSTM model, negates the need for the LSTM during testing. After a specified number of timesteps, the agent's final profit is calculated from the testing simulation.
