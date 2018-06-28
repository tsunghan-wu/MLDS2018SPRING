import numpy as np 
import random
import pickle
import torch
import sys
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from agent_dir.agent import Agent
from agent_dir.util2 import Net, ReplayBuffer, Schedule

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.manual_seed(39)
            np.random.seed(39)
        else:
            torch.manual_seed(39)
            np.random.seed(39)
        # DQN network parameter
        self.action_class = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape

        # training 
        self.batch_size = 32
        self.buffer_size = 10000
        self.timestamp = 5000000

        if args.test_dqn:
            #you can load your model here
            if self.use_cuda:
                self.online_net = Net(self.action_class, self.state_shape).cuda()
                self.online_net.load_state_dict(torch.load('./model/dqn_model.pkl'))
            else:
                self.online_net = Net(self.action_class, self.state_shape)
                self.online_net.load_state_dict(torch.load('./model/dqn_model.pkl', map_location='cpu'))

            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        torch.manual_seed(39)
        np.random.seed(39)
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def optimize_online_network(self):
        
        # batch = [state, action, next_state, reward]
        batch = list(self.replay_buffer.sample(self.batch_size))
        batch = [torch.from_numpy(np.array(x)) for x in batch]
        if self.use_cuda:
            batch = [x.cuda() for x in batch]
            
        state_batch, action_batch, next_state_batch, reward_batch, terminate_batch = batch
        
        # Q(s_i, a)       
        sa_val = self.online_net.forward(Variable(state_batch.float())).gather(1, Variable(action_batch.unsqueeze(1)))

        # y = r_i + GAMMA * max_a Q_target (s_{i+1}, a) ( -> live ) or r_i ( -> done)
        max_action = self.online_net.forward(Variable(next_state_batch.float())).max(1)[1]
        new_sa_val = self.target_net.forward(Variable(next_state_batch.float())).gather(1, max_action.unsqueeze(-1)).squeeze(-1).detach()

        y = Variable(reward_batch.float()) + (1.0 - Variable(terminate_batch.float()))*(0.99 * new_sa_val)

        # loss function and backprop
        loss = self.loss_func(sa_val, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy()


    def train(self):
        """
        Implement your training algorithm here
        """

        self.logfile = open("rmsprop_training_log.txt", "w")
        self.reward_file = open("rmsprop_reward_file.pickle", "w")


        # initialize target, online network
        if self.use_cuda:
            self.target_net = Net(self.action_class, self.state_shape).cuda()
            self.online_net = Net(self.action_class, self.state_shape).cuda()
        else:
            self.target_net = Net(self.action_class, self.state_shape)
            self.online_net = Net(self.action_class, self.state_shape)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.optimizer = torch.optim.RMSprop(self.online_net.parameters(), lr=1.5e-4)
        self.loss_func = nn.MSELoss()
        self.schedule = Schedule(self.timestamp)
        

        s0 = self.env.reset()
        episode_reward = []
        current_reward = 0
        best_reward = 0
        for game in range(self.timestamp):           
            # play game
            action = self.make_action(s0, test=False)
            s1, r, done, _ = self.env.step(action)
            done = 1 if done == True else 0
            current_reward += r
            self.replay_buffer.push(s0, action, s1, r, done)

            # train online network
            if game % 4 == 3 and game > 5000:
                loss = self.optimize_online_network()

            # update target network
            if game % 1000 == 999 and game > 5000:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if done == True:
                s0 = self.env.reset()

                episode_reward.append(current_reward)
                past_reward = sum(episode_reward[-100:])
                print ("Timestamp = {} Episode = {} Past 100 reward = {}".format(game,len(episode_reward), past_reward), file=self.logfile)  
                if past_reward > 3000.0 and past_reward > best_reward:
                    filename = os.path.join("./model", "reward_%d.pkl" % past_reward)
                    torch.save(self.online_net.state_dict(), filename)
                current_reward = 0
                best_reawrd = max(best_reward, past_reward)
            if game % 1000000 == 999999:
                iteration_filename = os.path.join("./model", "rmsprop_iteration_%d.pkl" % game)
                torch.save(self.online_net.state_dict(), iteration_filename)
                
            else:
                s0 = s1
            self.logfile.flush()
        torch.save(self.online_net.state_dict(), "./model/final_model.pkl")
        pickle.dump(episode_reward, self.reward_file)



    def make_action(self, observation, test=True):

        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        observation = Variable(torch.from_numpy(np.expand_dims(observation, axis=0)))
        if self.use_cuda:
            observation = observation.cuda()
        if test == False:
            threshold = max(0.025, self.schedule.take_action())
        else:
            threshold = 0.005
        explore = np.random.random()
        # print ("explore = ", explore, "threshold = ", threshold, file=self.logfile)
        if explore < threshold:
            return self.env.get_random_action()
        else:
            action_val = self.online_net.forward(observation)
            best_action = action_val.data.max(1)[1].cpu().numpy()
            return best_action[0]

        return self.env.get_random_action()



