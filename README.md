# LTL Reward Specification in a CMDP
This repository is for my Master's Thesis I am working on under the supervision of Professor Yves Lesperance. 

## Abstract

This thesis will focus on improving two of the main concerns with reinforcement learning (RL), safety and performance. On one hand, the goal in RL is to learn a policy that will maximize the returned reward by exploring an unknown environment, but such exploration typically does not concern itself with the potential safety of an agent and therefore can not guarantee its safety. On the other hand, finding an optimal policy using reinforcement learning can require a lot of exploration of the environment. Sometimes this can be of very high cost and in some cases not possible to perform, such as physical environments.

In order to address said issues I propose the use of Constrained Markov Decision Processes with Linear Temporal Logic as a reward specifier. Constrained Markov Decision Processes is a framework for reinforcement learning problems that allows for the separation of safety specifications and the reward function by naturally encoding the safety concerns as constraints. Linear Temporal Logic is meant to be used to specify the reward function in the RL problem as a Reward Machine, first introduced by Icarte et al. [2018], and map the CMDP to a CMDPRM. The idea is that the RM will allow the agent to decompose the task (and specify multiple tasks) allowing it it to learn an optimal policy while minimizing exploration, and by the same principle minimize the amount of times the agent will encounter the unsafe states specified by the original CMDP. I will introduce a novel algorithm, Constrained Q-Learning for Reward Machines (CQRM), meant to solve the Reinforcement Learning problem in a CMDPRM. The goal is to test and evaluate the proposed methodology in the Safety Gym benchmark suite developed by Ray et al. [2019]., which consists of high-dimensional continuous control environments meant to measure the performance and safety of agents in Constrained Markov Decision Processes.

## References

Rodrigo Toro Icarte, Toryn Klassen, Richard Valenzano, and Sheila McIlraith. Using reward machines for high-level task specification and decomposition in reinforcement learning. In *International Conference on Machine Learning*, pages 2107-2116, 2018. 

Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. *arXiv preprint arXiv:1910.01708*, 2019.
