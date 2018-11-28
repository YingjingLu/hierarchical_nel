import numpy as np, gym, sys, copy, argparse
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
from models import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
np.random.seed( 2333 )
random.seed( 2333 )
USE_TF = False
TERM_STEP = 250

class A2C( object ):
    
    def __init__(self, environment_name, lr, render=False):
        self.sess = None

        self.environment_name = environment_name
        self.log_file = open( "a2c{}-2.txt".format(TERM_STEP), "w" )
        self.test_file = open("a2c_test{}.txt".format( TERM_STEP ), "w" )

        self.base_dir = "a2c"
        # make working directories
        if not os.path.isdir( self.base_dir ): os.mkdir( self.base_dir )
        self.env_dir = os.path.join( self.base_dir, self.environment_name + str( TERM_STEP ) )
        if not os.path.isdir( self.env_dir ): os.mkdir( self.env_dir )
        self.tmp_dir = os.path.join( self.base_dir, self.environment_name + "_tmp" )
        if not os.path.isdir( self.tmp_dir ): os.mkdir( self.tmp_dir )
        
        self.env =  gym.make( environment_name )
        if environment_name == "NEL-v0":
            self.action_size = 3
            self.learning_rate = lr
            self.state_size = 2
            self.discount_factor = 1.0
            self.start_epi = 0.99
            self.end_epi = 0.3
            self.burn_in_amount = 200
            self.memory_size = 50000
            self.lr_decay = 0.99

            self.prev_state = None 
            self.available_tong = 0

        self.actor = Actor( lr )
        self.critic = Critic( lr )
        
        # epi greedy
        self.max_episode = 10000
        self.evaluate_every = 100 # evaluate reward for every * episode
    
    # epi-greedy
    def train(self):
        loss_list = []
        reward_accu = 0
        cur_state = self.env.reset()
        epi = np.linspace( self.start_epi, self.end_epi, num = 100000 )
        _iter = 0
        is_term = False
        cur_episode = 5201
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

    def generate_episode( self, num_step = TERM_STEP ):
        # continue from previous location and step an episode amount of steps
        # compile the episode data and return in the format of list of [ prev_state, action, reward, cur_state, tong ]
        # self.prev_state : record prev_state
        # self.available_tong: record available tong
        res = []
        if self.prev_state is None:
            self.prev_state = self.env.reset()
        
        return res

    def save_model(self, episode_num = 100 ):
        # Helper function to save your model / weights. 
        path = os.path.join( self.env_dir, str( episode_num ) ) + "/"
        os.mkdir( path ) if not os.path.exists( path ) else print( "Warning: saving in an existing path for episode {}".format( episode_num ) )
        cpt = dict()
        cpt[ "actor" ] = self.actor.network.state_dict()
        cpt[ "actor_optim" ] = self.actor.optim.state_dict()
        cpt[ "critic" ] = self.critic.network.state_dict()
        cpt[ "critic_optim" ] = self.critic.optim.state_dict()
        torch.save( cpt, path + "model.bin" )
        print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )


    def load_model(self, model_file):
        # Helper function to load an existing model.
        # self.saver.restore( self.sess, model_file)
        cpt = torch.load( model_file )
        self.actor.network.load_state_dict( cpt[ "actor" ] )
        self.actor.optim.load_state_dict( cpt[ "actor_optim" ] )
        self.critic.network.load_state_dict( cpt[ "critic" ] )
        self.critic.optim.load_state_dict( cpt[ "critic_optim" ] )