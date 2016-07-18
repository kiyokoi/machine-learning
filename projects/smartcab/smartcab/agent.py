import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qtable = {}
        # Track agent's learning ability for each trial
        self.total_reward = 0.0
        self.alpha = 1.0
        self.gamma = 1.0
        
        # Populate Q-table with place-holders
        for light in ['red', 'green']:
            for oncoming in self.env.valid_actions:
                for left in self.env.valid_actions:
                    for right in self.env.valid_actions:
                        for waypoint in self.env.valid_actions:
                                state = ((light, oncoming, left, right), waypoint)
                                for action in self.env.valid_actions:
                                    self.qtable[(state, action)] = -float('inf')


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0.0
        self.alpha = 1.0
        self.gamma = 1.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        inputs_tuple = tuple(inputs.values())
        self.state = (inputs_tuple, self.next_waypoint, deadline)

        # TODO: Select action according to your policy
        # action = random.choice(self.env.valid_actions)    # [for Q2]

        # Execute action and get reward
        # reward = self.env.act(self, action)   # [for Q2]

        # TODO: Learn policy based on state, action, reward
        def argmax(state, qtable, waypoint):        
            max_reward = -float('inf')
            best_action = None
            for action in self.env.valid_actions:
                #print 'action: {}, qtable: {}, max: {}'.format(action, self.qtable[(self.state, action)], max_reward)   # [debug]
                if qtable[(state, action)] > max_reward:
                    best_action = action
                    max_reward = qtable[(state, action)]
        
            # Take the planned action if there is no positive reward        
            if max_reward < 0.0:
                best_action = waypoint
            
            return best_action, max_reward
            
        best = argmax(self.state, self.qtable, self.next_waypoint)
        best_action = best[0]
        max_reward = best[1]
        
        if max_reward < 0.0:
            qval = 0.0
        else:
            qval = max_reward
            
        reward = self.env.act(self, best_action)
        
        # Update reward with learning rate, alpha, and discount factor, gamma
        # alpha and gamma are reduced by 20% in each iteration
        self.next_waypoint = self.planner.next_waypoint()        
        inputs = self.env.sense(self)
        inputs_tuple = tuple(inputs.values())
        s_prime = (inputs_tuple, self.next_waypoint)
        argmaxprime = argmax(s_prime, self.qtable, self.next_waypoint)        
        a_prime = argmaxprime[0]
        q_prime = self.qtable[(s_prime, a_prime)]
        if q_prime < -9999:
            q_prime = 0.0
        reward -= (self.alpha * qval)
        reward += (self.alpha * reward)
        reward += (self.alpha * self.gamma * q_prime)
        self.total_reward += reward
        self.alpha *= 0.8
        self.gamma *= 0.8
        
        if self.next_waypoint == None:        
            print 'Cumulative reward = {}'.format(self.total_reward)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
