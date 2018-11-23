#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
from gym import wrappers
import ffmpeg
import random
import os
import matplotlib.pyplot as plt
import math
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import nel
os.environ["CUDA_VISIBLE_DEVICES"]="0"
USE_TF = False
TERM_STEP = 250

class Actor_Network( nn.Module ):
    def __init__(self, goal_size ):
        super( Actor_Network, self ).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l0 = nn.Linear( 16*5*5 + 3 + 1, 512 )
        self.l1 = nn.Linear( 512 , 256 )
        self.l2 = nn.Linear( 256, 256 )
        self.l_a_0 = nn.Linear( 256, 256 )
        self.l_a_2 = nn.Linear( 256, goal_size )
    def forward( self, scent_batch, vision_batch, moved_batch ):
        scent_batch = torch.tensor( scent_batch, dtype = torch.float32 ).cuda()
        vision_batch = torch.tensor( vision_batch, dtype = torch.float32 ).cuda()
        moved_batch = torch.tensor( moved_batch, dtype = torch.float32 ).cuda()

        """ Compute vision"""
        conv = F.relu( self.conv1( vision_batch ) )
        # print( conv.size() )
        conv = F.relu( self.conv2( conv ) )
        # print( conv.size() )
        conv = conv.view( -1, 16*5*5)
        conv = torch.cat( ( conv, scent_batch, moved_batch ), dim = 1 )

        conv = F.relu( self.l0( conv ) )
        conv = F.relu( self.l1( conv ) )
        conv = F.relu( self.l2( conv ) )
        conv = F.relu( self.l_a_0( conv ) )
        conv = torch.nn.softmax( conv, dim = 1 )
        return conv

class Critic_Network( nn.Module ):
    def __init__(self, goal_size, action_size ):
        super( Critic_Network, self ).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l0 = nn.Linear( 16*5*5 + 3 + 1 + goal_size, 512 )
        self.l1 = nn.Linear( 512 , 256 )
        self.l2 = nn.Linear( 256, 256 )
        self.l_a_0 = nn.Linear( 256, 256 )
        self.l_a_2 = nn.Linear( 256, action_size )
    def forward( self, scent_batch, vision_batch, moved_batch, goal_batch ):
        scent_batch = torch.tensor( scent_batch, dtype = torch.float32 ).cuda()
        vision_batch = torch.tensor( vision_batch, dtype = torch.float32 ).cuda()
        moved_batch = torch.tensor( moved_batch, dtype = torch.float32 ).cuda()
        goal_batch = torch.tensor( goal_batch, dtype = torch.float32 ).cuda()

        """ Compute vision"""
        conv = F.relu( self.conv1( vision_batch ) )
        # print( conv.size() )
        conv = F.relu( self.conv2( conv ) )
        # print( conv.size() )
        conv = conv.view( -1, 16*5*5)
        conv = torch.cat( ( conv, scent_batch, moved_batch, goal_batch ), dim = 1 )

        conv = F.relu( self.l0( conv ) )
        conv = F.relu( self.l1( conv ) )
        conv = F.relu( self.l2( conv ) )
        conv = F.relu( self.l_a_0( conv ) )
        conv = self.l_a_2( conv )
        return conv

class Goal_Q( object ):

    def __init__(self, environment_name,  goal_size, lr ): 
        self.env_name = environment_name
        self.goal_size = goal_size

        # one fully connected layer
        if self.env_name == "NEL-v0":
            self.network = Goal_Network( goal_size )

        self.network = self.network.cuda()

        self.loss = nn.MSELoss().cuda()

        self.optim = optim.Adam( self.network.parameters(), lr=lr )

    def get_q_val( self, sess, scent_batch, vision_batch, moved_batch ):
        # print( scent_batch.shape, vision_batch.shape, moved_batch.shape )
        q_val_next = self.network( scent_batch, vision_batch, moved_batch )
        return q_val_next.cpu().detach().numpy()

    def get_loss( self, sess, sb, vb, mb, goal_batch, reward_batch ):
        goal_batch = goal_batch
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        goal_batch = torch.tensor( goal_batch, dtype = torch.float32 ).cuda()
        q_val = self.network( sb, vb, mb )
        q_val = ( q_val * goal_batch ).sum( dim = 1 ).view(-1, 1)
        loss = self.loss( q_val, reward_batch )
        return loss

    def optimize( self, sess, sb, vb, mb, goal_batch, reward_batch, learning_rate ):
        self.optim.zero_grad()
        goal_batch = goal_batch
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        loss = self.get_loss( sess, sb, vb, mb, goal_batch, reward_batch )
        loss.backward()
        self.optim.step()
        return loss.cpu().detach().item()

class Action_Q( object ):

    def __init__(self, environment_name, goal_size, action_size, lr ):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.env_name = environment_name
        self.goal_size = goal_size
        self.action_size = action_size

        # one fully connected layer
        if self.env_name == "NEL-v0":
            self.network = Action_Network( goal_size, action_size )

        self.network = self.network.cuda()

        self.loss = nn.MSELoss().cuda()

        self.optim = optim.Adam( self.network.parameters(), lr=lr )

    def get_q_val( self, sess, scent_batch, vision_batch, moved_batch, goal_batch ):
        # print( scent_batch.shape, vision_batch.shape, moved_batch.shape, goal_batch.shape )
        q_val_next = self.network( scent_batch, vision_batch, moved_batch, goal_batch )
        return q_val_next.cpu().detach().numpy()

    def get_loss( self, sess, sb, vb, mb, action_batch, reward_batch, goal_batch ):
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        action_batch = torch.tensor( action_batch, dtype = torch.float32 ).cuda()
        q_val = self.network( sb, vb, mb, goal_batch )
        # print( q_val.size(), action_batch.size(), reward_batch.size() )
        q_val = ( q_val * action_batch ).sum( dim = 1 ).view(-1, 1)
        loss = self.loss( q_val, reward_batch )
        return loss

    def optimize( self, sess, sb, vb, mb, action_batch, reward_batch, goal_batch, learning_rate ):
        self.optim.zero_grad()
        goal_batch = goal_batch
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        loss = self.get_loss( sess, sb, vb, mb, action_batch, reward_batch, goal_batch )
        loss.backward()
        self.optim.step()
        return loss.cpu().detach().item()

# memory storage: ( st, a, r, st+1, is_term )
class Replay_Memory():

    def __init__(self, max_memory_size=50000):
        self.max_memory_size = max_memory_size
        self.memory = [ [] for i in range( max_memory_size ) ]
        self.cur_index = 0

    def get_memory_size( self ):
        return len( self.memory )

    def sample_batch(self, batch_size=32):
        assert( len( self.memory ) >= batch_size, "memory does not have enough to sample" )
        # return random.sample( self.memory, batch_size )
        if self.cur_index < self.max_memory_size:
        	return random.sample( self.memory[ :self.cur_index ], batch_size )
        else:
        	return random.sample( self.memory, batch_size )

    def get_one_episode( self, num_epi ):
        pass

    def append(self, transition):
        # Appends transition to the memory.     
        assert( len( transition ) == 5, "invalid transition format" )
        self.memory[ self.cur_index % self.max_memory_size ] = transition
        self.cur_index += 1

class DQN_Agent():
    
    def __init__(self, environment_name, lr, render=False):
        self.sess = None

        self.environment_name = environment_name
        self.log_file = open( "h_dqn.txt", "w" )
        self.test_file = open("h_dqn_test.txt", "w" )

        self.base_dir = "h_dqn"
        # make working directories
        if not os.path.isdir( self.base_dir ): os.mkdir( self.base_dir )
        self.env_dir = os.path.join( self.base_dir, self.environment_name + str( TERM_STEP ) )
        if not os.path.isdir( self.env_dir ): os.mkdir( self.env_dir )
        self.tmp_dir = os.path.join( self.base_dir, self.environment_name + "_tmp" )
        if not os.path.isdir( self.tmp_dir ): os.mkdir( self.tmp_dir )
        
        self.env =  gym.make( environment_name )
        if render:
            self.env = wrappers.Monitor(self.env, self.tmp_dir + "/monitor_1", force = True )
        if environment_name == "NEL-v0":
            self.action_size = 3
            self.goal_size = 3
            self.learning_rate = lr
            self.state_size = 2
            self.discount_factor = 1.0
            self.start_epi = 0.99
            self.end_epi = 0.3
            self.sample_batch = 64
            self.burn_in_amount = 200
            self.memory_size = 50000
            self.lr_decay = 0.99
            self.reward_to_goal_dict = { 20:0, -1:1, 100:2 }
            self.goal_space = [ 0, 2 ]

        self.goal_memory = Replay_Memory( max_memory_size = self.memory_size )
        self.action_memory = Replay_Memory( max_memory_size = self.memory_size )
        self.burn_in_memory( self.burn_in_amount )
        self.goal_q = Goal_Q( environment_name, self.goal_size, lr )
        self.action_q = Action_Q( environment_name, self.goal_size, self.action_size, lr )
        
        # epi greedy
        self.max_episode = 10000
        self.evaluate_every = 100 # evaluate reward for every * episode
    
    def state_dict_to_batch( self, state_dict_list ):
        scent_batch, vision_batch, moved_batch = [], [], []
        if type( state_dict_list ) == dict:
            scent_batch.append( np.array( state_dict_list[ "scent" ] ).reshape( ( 1, 3 ) ) )
            vision_batch.append( state_dict_list[ "vision" ].reshape( 1, 3, 11, 11 ) )
            m = [ float( state_dict_list[ "moved" ] ) ]
            moved_batch.append( np.array( m ).reshape( 1, 1 ) )
        else:
            for d in state_dict_list:
                scent_batch.append( np.array( d[ "scent" ] ).reshape( ( 1, 3 ) ) )
                vision_batch.append( d[ "vision" ].reshape( 1, 3, 11, 11 ) )
                m = [ float( d[ "moved" ] ) ]
                moved_batch.append( np.array( m ).reshape( 1, 1 ) )
            
        sr = np.concatenate( scent_batch, axis = 0 )
        sv = np.concatenate( vision_batch, axis = 0 )
        sm = np.concatenate( moved_batch, axis = 0 )
        return sr, sv, sm

    def action_to_onehot( self, action_batch ):
        if type( action_batch ) == int or type( action_batch ) == float or type( action_batch ) == np.int64 or type( action_batch ) == np.int:
            action_batch = int( action_batch )
            res = np.zeros( self.action_size, dtype = np.float32 )
            res[ action_batch ] = 1
            return res.reshape( -1, self.action_size )
        batch_size = len( action_batch )
        res = np.zeros( ( batch_size, self.action_size ), dtype = np.float32 )
        for i in range( len( action_batch ) ):
            res[ i, action_batch[ i ] ] = 1
        return res

    def reward_to_goal( self, reward ):
        """
        jelly: 0
        tong: 1
        diamond: 2
        in one hot format
        """
        reward = int( reward )
        return self.reward_to_goal_dict[ reward ]

    def collected_item_to_goal(self, prev_list, cur_list):
        # d, t, j, always 0
        # check jelly
        if cur_list[ 2 ] - prev_list[ 2 ] > 0:
            return cur_list, 0
        # check tong
        if cur_list[ 1 ] > prev_list[ 1 ]:
            return cur_list, 1
        # check dismond
        if cur_list[ 0 ] > prev_list[ 0 ]:
            return cur_list, 2

        # assert cur_list == prev_list
        return cur_list, -1

    def reward_to_goal_batch( self, reward_batch ):
        
        if type( reward_batch ) == int or type( reward_batch ) == float:
            reward_batch = int( reward_batch )
            res = np.zeros( self.goal_size, dtype = np.float32 )
            res[ self.reward_to_goal(reward_batch ) ] = 1
            return res.reshape( -1, self.goal_size )
        batch_size = len( reward_batch )
        res = np.zeros( ( batch_size, self.goal_size ), dtype = np.float32 )
        for i in range( len( reward_batch ) ):
            res[ i, self.reward_to_goal(reward_batch[ i ] ) ] = 1
        return res

    def goal_to_onehot( self, goal_batch ):
        
        if type( goal_batch ) == int or type( goal_batch ) == float or type( goal_batch ) == np.int64 or type( goal_batch ) == np.int:
            goal_batch = int( goal_batch )
            res = np.zeros( self.goal_size, dtype = np.float32 )
            res[ goal_batch ] = 1
            return res.reshape( -1, self.goal_size )
        batch_size = len( goal_batch )
        res = np.zeros( ( batch_size, self.goal_size ), dtype = np.float32 )
        for i in range( len( goal_batch ) ):
            res[ i, goal_batch[ i ] ] = 1
        return res

    def construct_goal_q_batch( self, sample_list ):
        state_batch = []
        goal_batch = []
        is_term_batch = []
        reward_batch = []
        next_state_batch = []

        for transitions in sample_list:
            [ cur, r, _next, _is_term, goal ] = transitions
            state_batch.append( cur )
            goal_batch.append( goal )
            reward_batch.append( r )
            is_term_batch.append( _is_term )
            next_state_batch.append( _next )

        # convert everything to np array
        cur_bs, cur_bv, cur_bm = self.state_dict_to_batch( state_batch )
        goal_batch = self.goal_to_onehot( goal_batch )
        is_term_batch = np.array( is_term_batch ).reshape( -1, 1 )
        reward_batch = np.array( reward_batch ).reshape( -1, 1 )
        next_bs, next_bv, next_bm = self.state_dict_to_batch( next_state_batch )

        return cur_bs, cur_bv, cur_bm, reward_batch, goal_batch, is_term_batch, next_bs, next_bv, next_bm

    def construct_action_q_batch( self, sample_list ):
        state_batch = []
        action_batch = []
        is_term_batch = []
        reward_batch = []
        next_state_batch = []
        cur_goal_batch = []
        next_goal_batch = []

        for transitions in sample_list:
            [ cur, a, r, _next, _is_term, cur_goal, next_goal ] = transitions
            state_batch.append( cur )
            action_batch.append( a )
            reward_batch.append( r )
            is_term_batch.append( _is_term )
            next_state_batch.append( _next )
            cur_goal_batch.append( cur_goal )
            next_goal_batch.append( next_goal )

        # convert everything to np array
        cur_bs, cur_bv, cur_bm = self.state_dict_to_batch( state_batch )
        action_batch = self.action_to_onehot( action_batch )
        reward_batch = np.array( reward_batch ).reshape( -1, 1 )
        is_term_batch = np.array( is_term_batch ).reshape( -1, 1 )
        next_bs, next_bv, next_bm = self.state_dict_to_batch( next_state_batch )
        cur_goal_batch = self.goal_to_onehot( cur_goal_batch )
        next_goal_batch = self.goal_to_onehot( next_goal_batch )

        return cur_bs, cur_bv, cur_bm, action_batch, reward_batch, cur_goal_batch, next_goal_batch, is_term_batch, next_bs, next_bv, next_bm

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.     

        prob = np.ones( self.action_size )
        # reduce all prob to epi
        prob = prob * self.epi / self.action_size
        remainder_prob = 1 - np.sum( prob )
        optim_action = np.argmax( np.array( q_values ) )
        prob[ optim_action ] += remainder_prob
        return prob

        # Creating greedy policy for test time. 
    def greedy_policy(self, q_values):
        
        prob = np.zeros( self.action_size )
        optim_action = np.argmax( np.array( q_values ) )
        prob[ optim_action ] = 1
        return prob
    
    # epi-greedy
    def train(self):
        loss_list = []
        reward_accu = 0
        cur_state = self.env.reset()
        epi = np.linspace( self.start_epi, self.end_epi, num = 100000 )
        _iter = 0
        is_term = False
        cur_episode = 1
        print( "Enter training" )
        while cur_episode < self.max_episode + 1:

            if _iter < 100000: self.epi = epi[ _iter ]
            else: self.epi = 0.3

            bs, bm, bv = self.state_dict_to_batch( cur_state )
            # sample goal with epi-greedy
            if random.random() < self.epi:
                goal = random.choice( self.goal_space )
            else:
                goal_values = self.goal_q.get_q_val( self.sess, bs, bm, bv )
                goal = np.argmax( goal_values.flatten() )

            prev_goal_list = np.zeros( self.goal_size + 1, dtype = np.int64 )
            while not is_term:
                extrinsic_reward = 0
                start_state = cur_state
                last_goal = -1
                while not ( is_term or ( last_goal == goal ) ):
                    bs, bm, bv = self.state_dict_to_batch( cur_state )
                    if random.random() < self.epi:
                        action = self.env.action_space.sample()
                    # else select action from q Net
                    else:
                        q_values = self.action_q.get_q_val( self.sess, bs, bm, bv, self.goal_to_onehot( goal ) )
                        action = np.argmax( q_values.flatten() )

                    next_state, reward, is_term, _ = self.env.step( action )
                    reward_accu += reward
                    extrinsic_reward += reward
                    prev_goal_list, last_goal = self.collected_item_to_goal( prev_goal_list, self.env._agent.collected_items() )

                    _iter += 1
                    if _iter % TERM_STEP == 0:
                        is_term = True
                    # store transition into memory
                    self.action_memory.append( [ cur_state, action, reward, next_state, is_term, goal, goal ] )
                    # sample random minibatch drom D
                    """ Update action q """
                    sample_list = self.action_memory.sample_batch( batch_size = self.sample_batch )
                    # compile states into batch
                    cur_bs, cur_bv, cur_bm, action_batch, reward_batch, cur_goal_batch, next_goal_batch, is_term_batch, next_bs, next_bv, next_bm = \
                    self.construct_action_q_batch( sample_list )
                    q_val_next = self.action_q.get_q_val( self.sess, next_bs, next_bv, next_bm, next_goal_batch )
                    q_next_max = np.amax( q_val_next, axis = 1 ).reshape( -1, 1 )
                    # print("q next max", q_next_max.shape)
                    # invert the is terminate to mask out terminated from loss being added
                    is_term_mask = np.invert( is_term_batch ).astype( np.float )
                    reward_batch = reward_batch + self.discount_factor * is_term_mask * q_next_max
                    # gradient update
                    self.action_q.optimize( self.sess, cur_bs, cur_bv, cur_bm, 
                                                            action_batch, reward_batch, cur_goal_batch, self.learning_rate )
                    """ End update action q """

                    """ Update goal q """
                    sample_list = self.goal_memory.sample_batch( batch_size = self.sample_batch )
                    # compile states into batch
                    cur_bs, cur_bv, cur_bm, reward_batch, goal_batch, is_term_batch, next_bs, next_bv, next_bm = \
                    self.construct_goal_q_batch( sample_list )

                    q_val_next = self.goal_q.get_q_val( self.sess, next_bs, next_bv, next_bm )
                    q_next_max = np.amax( q_val_next, axis = 1 ).reshape( -1, 1 )
                    # print("q next max", q_next_max.shape)
                    # invert the is terminate to mask out terminated from loss being added
                    is_term_mask = np.invert( is_term_batch ).astype( np.float )
                    reward_batch = reward_batch + self.discount_factor * is_term_mask * q_next_max
                    # gradient update
                    self.goal_q.optimize( self.sess, cur_bs, cur_bv, cur_bm, goal_batch, reward_batch, self.learning_rate )
                    """ End update goal q """

                    cur_state = next_state
                
                self.goal_memory.append( [ start_state, extrinsic_reward, cur_state, is_term, goal ] )
                if not is_term:
                     # sample goal with epi-greedy
                    if random.random() < self.epi:
                        goal = random.choice( self.goal_space )
                    else:
                        goal_values = self.goal_q.get_q_val( self.sess, bs, bm, bv )
                        goal = np.argmax( goal_values.flatten() )

                

            print( os.environ["CUDA_VISIBLE_DEVICES"], TERM_STEP, cur_episode, "reward accu", reward_accu )
            if ( cur_episode ) % self.evaluate_every == 0:
                self.save_model( episode_num = cur_episode )

                self.learning_rate *= self.lr_decay
                loss_list.append( reward_accu )
                one = self.evaluate_total_reward()
                print( " Episide {} one_step {}".format( cur_episode, one ) )
                self.test_file.write( str( cur_episode ) + " " + str( one ) + "\n" )
                print( "----------------------------" )
                print(" ")
            # reset environment and retrain for next episode
            cur_state = self.env.reset()
            # print( "Env reset" )
            is_term = False
            self.log_file.write( str( cur_episode ) + " " + str( reward_accu ) + "\n" )
            cur_episode += 1
            reward_accu = 0
                
        x = list( range( len( loss_list ) ) )
        plt.plot( x, loss_list )
        plt.title( "Train Reward of {}".format( self.environment_name ) )
        plt.xlabel( "number of episodes for every 100 step" )
        plt.ylabel( "Reward" )
        plt.show()


    # evaluate total rewards given current q network
    # evaluate in two different fashion: one_step look ahead( regular estimate ) and two step look ahead
    # use epi greedy
    def evaluate_total_reward( self, epi = 0.05 , num_sepi = 10):
        total =[]
        for _ in range( num_sepi ):
            # evaluate based on value function ( one step look ahead )
            cur_state = self.env.reset()
            is_term = False
            reward_list = []
            step_num = 0
            prev_goal_list = np.zeros( self.goal_size + 1, dtype = np.int64 )
            # simulate the environment till is_term
            while not is_term:
                extrinsic_reward = 0
                bs, bv, bm = self.state_dict_to_batch( cur_state )
                goal = self.goal_q.get_q_val( self.sess, bs, bv, bm )
                goal = np.argmax( goal.flatten() )
                last_goal = -1
                while not( is_term or last_goal == goal ):
                    bs, bv, bm = self.state_dict_to_batch( cur_state )
                    action = self.action_q.get_q_val( self.sess, bs, bv, bm, self.goal_to_onehot( goal ) )
                    next_state, reward, is_term, _ = self.env.step( np.argmax( action.flatten() ) )
                    extrinsic_reward += reward
                    reward_list.append( reward )
                    step_num += 1
                    if step_num % TERM_STEP == 0:
                        is_term = True 
                    prev_goal_list, last_goal = self.collected_item_to_goal( prev_goal_list, self.env._agent.collected_items() )
                    cur_state = next_state
            one_step_total = np.sum( np.array( reward_list ) )
            total.append( one_step_total )

        return sum( total ) / num_sepi

    # call evaluate total award with all the given checkpoints
    def evaluate_wrapper( self, times_avg = 3 ):
        one_step_list, two_step_list = [], []
        for episode in range( self.evaluate_every, self.max_episode, self.evaluate_every ):
            self.load_model( os.path.join( self.env_dir, str( episode ) ) + "/" + "model.bin" )
            total_one, total_two = 0,0
            for _ in range( times_avg ):
            	one = self.evaluate_total_reward()
            	total_one += one

            one_step_list.append( total_one / times_avg )
            two_step_list.append( total_two  / times_avg )

        return one_step_list, two_step_list

    def burn_in_memory( self, burn_in_size ):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        burn_in_num = 0
        while burn_in_num < burn_in_size:
            cur_state = self.env.reset()
            is_term = False
            step_num = 0
            prev_goal_list = np.zeros( self.goal_size + 1, dtype = np.int64 )
            while not is_term:
                extrinsic_reward = 0
                start_state = cur_state
                goal = random.choice( self.goal_space )
                last_goal = -1
                while not( is_term or last_goal == goal ):
                    action = self.env.action_space.sample()
                    next_state, reward, is_term, _ = self.env.step( action )
                    extrinsic_reward += reward
                    step_num += 1
                    if step_num % TERM_STEP == 0:
                        is_term = True 
                    self.action_memory.append( [ cur_state, action, reward, next_state, is_term, goal, goal ] )
                    prev_goal_list, last_goal = self.collected_item_to_goal( prev_goal_list, self.env._agent.collected_items() )
                    cur_state = next_state
                self.goal_memory.append( [ start_state, extrinsic_reward, cur_state, is_term, goal ] )
                if not is_term:
                    goal = random.choice( self.goal_space )
                burn_in_num += 1
        print( "Done with burn in, memory size: {}".format( burn_in_size ) )

    def save_model(self, episode_num = 100 ):
        # Helper function to save your model / weights. 
        path = os.path.join( self.env_dir, str( episode_num ) ) + "/"
        os.mkdir( path )
        # self.saver.save( self.sess, path )
        # print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )
        cpt = dict()
        cpt[ "action_q" ] = self.action_q.network.state_dict()
        cpt[ "action_optim" ] = self.action_q.optim.state_dict()
        cpt[ "goal_q" ] = self.goal_q.network.state_dict()
        cpt[ "goal_optim" ] = self.goal_q.optim.state_dict()
        torch.save( cpt, path + "model.bin" )
        print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )


    def load_model(self, model_file):
        # Helper function to load an existing model.
        # self.saver.restore( self.sess, model_file)
        cpt = torch.load( model_file )
        self.action_q.network.load_state_dict( cpt[ "action_q" ] )
        self.action_q.optim.load_state_dict( cpt[ "action_optim" ] )
        self.goal_q.network.load_state_dict( cpt[ "goal_q" ] )
        self.goal_q.optim.load_state_dict( cpt[ "goal_optim" ] )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default="NEL-v0")
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str, default = "")
    parser.add_argument('--lr',dest='lr',type=float, default = 1e-3)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    if environment_name == "NEL-v0":
        agent = DQN_Agent( args.env , args.lr, render = args.render )
    if args.train:
        agent.train()
    else:
        agent.draw()

if __name__ == '__main__':
    main(sys.argv)
