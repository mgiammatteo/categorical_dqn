# Categorical DQN implementation on CartPole-v0 from OpenAi-Gym
This notebook contains an attempt at implementing categorical dqn as described [here](https://arxiv.org/pdf/1707.06887.pdf).
I have experienced issues on more complex environments such as 'LunarLander-v2'. Whether this is caused by bugs in the code or tuning of the hyperparameters and the structure of the neural network approximator I am not sure. 
Hopefully I will find time to investigate this further.
The project includes:
* **Categorical_DQN.ipynb**, which is the main notebook used to setup the model,agent and the training)
* **categorical_dqn_agent.py**, which containg the agent itself
* **model.py**, which contains the neural network approximator of the action value function.