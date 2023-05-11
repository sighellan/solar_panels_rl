from simple_rl.agents.AgentClass import Agent

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
"""
Example BO agent, using BOTorch
"""



class BOAgent(Agent):
    def __init__(self, actions, name="BO", num_initial_random=25, subset_size=100):
        Agent.__init__(self, name, actions)
        self.num_initial_random = num_initial_random
        self.subset_size = subset_size
        self.train_X = []
        self.train_Y = []
        self.successes = 0
        self.failures = 0
        
    def reset(self):
        Agent.reset(self)
        self.train_X = []
        self.train_Y = []
        self.successes = 0
        self.failures = 0
        
    def update(self, reward):
        prev_context = self.get_context(self.prev_state)
        prev_action = self.make_action_flt(self.prev_action)
        self.train_X.append(np.hstack([prev_context, prev_action]))
        self.train_Y.append(reward)
        
    def make_action_flt(self, action_str):
        return [float(ss) for ss in action_str.split(',')]
    
    def make_action_string(self, action):
        return ",".join([str(int(aa)) for aa in action])
    
    def get_context(self, state):
        return [float(cc) for cc in state]

    def act(self, state, reward):
        if self.prev_action is not None:
            self.update(reward)
            
        actions_flt = np.array([self.make_action_flt(aa) 
                                for aa in self.actions])
        context = self.get_context(state)
        
        if len(self.train_Y) < self.num_initial_random:
            # Pick an action randomly
            action = actions_flt[np.random.randint(len(actions_flt))]
            action_str = self.make_action_string(action)
            
        else:
            # Subsample training examples to speed up GP fitting
            if len(self.train_Y) < self.subset_size:
                subset_X = self.train_X
                subset_Y = self.train_Y
            else:
                idx = np.arange(0, len(self.train_Y))
                np.random.shuffle(idx)
                subset_X = np.array(self.train_X)[idx[:self.subset_size]]
                subset_Y = np.array(self.train_Y)[idx[:self.subset_size]]

            tensor_X = torch.tensor(np.array(subset_X))
            tensor_Y = torch.tensor(np.array([subset_Y])).T
            # Build a GP and use that to select next action

            gp = SingleTaskGP(tensor_X, tensor_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_model(mll, max_retries=0);
            
                UCB = UpperConfidenceBound(gp, beta=0.1)
                
                actions_extended = np.hstack([actions_flt, np.tile(context, (len(actions_flt), 1))])
                # Evaluate the acquisition function on the available values
                acq_values = [UCB(torch.tensor(np.array([aa])))[0].item() for aa in actions_extended]
                candidate_action = actions_flt[np.argmax(acq_values)]
                
                action_str = self.make_action_string(candidate_action)
                self.successes += 1
            except:
                # If the model fails to train, revert to picking randomly
                action = actions_flt[np.random.randint(len(actions_flt))]
                action_str = self.make_action_string(action)
                self.failures += 1
                # If there are many failures, print update
                if self.failures % 100 == 0:
                    print('Failures: %s' %self.failures)
                    print('Successes: %s' %self.successes)
            
        self.prev_action = action_str
        self.prev_state = state
        return action_str