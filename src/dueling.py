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

class NELNetwork( nn.Module ):
    def __init__(self, action_size ):
        super( NELNetwork, self ).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l0 = nn.Linear( 16*5*5 + 3 + 1, 512 )
        self.l1 = nn.Linear( 512 , 256 )
        self.l2 = nn.Linear( 256, 128 )
        self.l_a_0 = nn.Linear( 128, 256 )
        self.l_a_1 = nn.Linear( 256, action_size )
        self.l_v_0 = nn.Linear( 128 , 256 )
        self.l_v_1 = nn.Linear( 256, 1 )

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
        advantage_out = F.relu( self.l_a_0( conv ) )
        advantage_out = self.l_a_1( advantage_out )
        value_out = F.relu( self.l_v_0( conv ) )
        value_out = self.l_v_1( value_out )

        advantage_out = advantage_out - torch.mean( advantage_out, dim = 1, keepdim = True )
        # print("dvantage out ", advantage_out.size())
        conv = advantage_out + value_out
        # print("action out", conv.size())
        return conv

class QNetwork( object ):

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name,  action_size, lr ):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.env_name = environment_name
        self.action_size = action_size

        # one fully connected layer
        if self.env_name == "NEL-v0":
            self.network = NELNetwork(  action_size )

        self.network.cuda()

        self.loss = nn.MSELoss()

        self.optim = optim.Adam( self.network.parameters(), lr=lr )

    def get_q_val( self, sess, scent_batch, vision_batch, moved_batch ):
        # print( scent_batch.shape, vision_batch.shape, moved_batch.shape )
        q_val_next = self.network( scent_batch, vision_batch, moved_batch )
        return q_val_next.cpu().detach().numpy()

    def get_loss( self, sess, sb, vb, mb, action_batch, reward_batch ):
        batch_size, _ = action_batch.shape
        action_batch = action_batch
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        action_onehot = torch.zeros( batch_size, self.action_size ).cuda()
        for i in range( len( action_batch ) ):
            action_onehot[ i, action_batch[i][0] ] = 1
        q_val = self.network( sb, vb, mb )
        q_val = ( q_val * action_onehot ).sum( dim = 1 ).view(-1, 1)
        loss = self.loss( q_val, reward_batch )
        return loss

    def optimize( self, sess, sb, vb, mb, action_batch, reward_batch, learning_rate ):
        self.optim.zero_grad()
        action_batch = action_batch
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        loss = self.get_loss( sess, sb, vb, mb, action_batch, reward_batch )
        loss.backward()
        self.optim.step()
        return loss.cpu().detach().item()

# memory storage: ( st, a, r, st+1, is_term )
class Replay_Memory():

    def __init__(self, max_memory_size=50000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory his as a list of transitions. 
        self.max_memory_size = max_memory_size
        self.memory = [ [] for i in range( max_memory_size ) ]
        self.cur_index = 0

    def get_memory_size( self ):
        return len( self.memory )

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        assert( len( self.memory ) >= batch_size, "memory does not have enough to sample" )
        # return random.sample( self.memory, batch_size )
        if self.cur_index < self.max_memory_size:
        	return random.sample( self.memory[ :self.cur_index ], batch_size )
        else:
        	return random.sample( self.memory, batch_size )

    def append(self, transition):
        # Appends transition to the memory.     
        assert( len( transition ) == 5, "invalid transition format" )
        # if transition not in self.memory_hash:
        #     if self.get_memory_size() == self.max_memory_size:
        #         transition_poped = self.memory.pop( 0 )
        #         self.memory_hash.remove( transition_poped )
        #     self.memory.append( transition )
        # if self.get_memory_size() == self.max_memory_size:
        #     transition_poped = self.memory.pop( 0 )
        # self.memory.append( transition )
        self.memory[ self.cur_index % self.max_memory_size ] = transition
        self.cur_index += 1



class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, lr, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
         # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
        
        self.sess = None

        self.environment_name = environment_name
        

        self.base_dir = "duel_work_dir"
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
            self.learning_rate = lr
            self.state_size = 2
            self.discount_factor = 1.0
            self.start_epi = 0.99
            self.end_epi = 0.3
            self.sample_batch = 32
            self.burn_in_amount = 256
            self.memory_size = 50000
            self.lr_decay = 0.99

        self.replay_memory = Replay_Memory( max_memory_size = self.memory_size )
        self.burn_in_memory( self.burn_in_amount )
        self.source_action_value = QNetwork( environment_name, self.action_size, lr )
        
        # update target paeam per C episodes
        self.C = 100
        # epi greedy
        self.max_episode = 10000
        self.evaluate_every = 50 # evaluate reward for every * episode
    
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
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 
        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        cur_state = self.env.reset()
        epi = np.linspace( self.start_epi, self.end_epi, num = 100000 )
        _iter = 0
        is_term = False
        cur_episode = 1
        print( "Enter training" )
        # sample action and record state, action reward until termination
        while cur_episode < self.max_episode + 1:
            # take action according to epi-greedy
            # with episilon select random action

            if _iter < 100000:
                self.epi = epi[ _iter ]
            else:
                self.epi = 0.3

            bs, bm, bv = self.state_dict_to_batch( cur_state )
            if random.random() < self.epi:
                action = self.env.action_space.sample()
            # else select action from q Net
            else:
                
                q_values = self.source_action_value.get_q_val( self.sess, bs, bm, bv )
                action = np.argmax( q_values.flatten() )

            next_state, reward, is_term, _ = self.env.step( action )
            reward_accu += reward
            # store transition into memory
            self.replay_memory.append( ( cur_state, action, reward, next_state, is_term ) )
            # sample random minibatch drom D
            sample_list = self.replay_memory.sample_batch( batch_size = self.sample_batch )
            # compile states into batch
            state_batch = []
            action_batch = []
            is_term_batch = []
            reward_batch = []
            next_state_batch = []

            for transitions in sample_list:
                [ cur, a, r, _next, _is_term ] = transitions
                state_batch.append( cur )
                action_batch.append( a )
                reward_batch.append( r )
                is_term_batch.append( _is_term )
                next_state_batch.append( _next )

            # convert everything to np array
            cur_bs, cur_bv, cur_bm = self.state_dict_to_batch( state_batch )
            action_batch = np.array( action_batch )
            reward_batch = np.array( reward_batch )
            is_term_batch = np.array( is_term_batch )
            next_bs, next_bv, next_bm = self.state_dict_to_batch( next_state_batch )

            q_val_next = self.source_action_value.get_q_val( self.sess, next_bs, next_bv, next_bm )
            q_next_max = np.amax( q_val_next, axis = 1 )
            # print("q next max", q_next_max.shape)
            # invert the is terminate to mask out terminated from loss being added
            is_term_mask = np.invert( is_term_batch ).astype( np.float )
            reward_batch = reward_batch + self.discount_factor * is_term_mask * q_next_max
            # print("reward_batch", reward_batch.shape )
            reward_batch = reward_batch.reshape( ( -1, 1 ) )
            # print("reward_batch", reward_batch.shape )
            # print( "state_batch", state_batch.shape, state_batch )
            action_batch = action_batch.reshape( ( -1, 1 ) )
            # gradient update
            loss = self.source_action_value.optimize( self.sess, cur_bs, cur_bv, cur_bm, 
                                                      action_batch, reward_batch, self.learning_rate )
            loss_list.append( loss )
            cur_state = next_state

            _iter += 1
            # print( "Step" )

            if _iter % TERM_STEP == 0:
                is_term = True

            # check if an episode ends according to env criteria
            if is_term:
                print( os.environ["CUDA_VISIBLE_DEVICES"], TERM_STEP, cur_episode, "reward accu", reward_accu )
                # print( "Position: ", cur_state[ 0 ] )
                # loss_list.append( cur_state[ 0 ] )
                # print( "Finish training episode {}".format( cur_episode ) )
                # save for every evaluate, evaluate if necessary
                if ( cur_episode ) % self.evaluate_every == 0:
                    self.save_model( episode_num = cur_episode )

                    self.learning_rate *= self.lr_decay
                    loss_list.append( reward_accu )

                # save loss and plot
                # loss = self.source_action_value.get_loss( self.sess, state_batch, action_batch, reward_batch ) 
                # loss_list.append( loss )
                # if cur_episode % 100 == 0:
                #     print( "Episode {}, loss {}, epi {}".format( cur_episode, loss, self.epi ) )
                    one = self.evaluate_total_reward()
                    print( " Episide {} one_step {}".format( cur_episode, one ) )
                    print( "----------------------------" )
                    print(" ")
                # reset environment and retrain for next episode
                cur_state = self.env.reset()
                # print( "Env reset" )
                is_term = False
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
    def evaluate_total_reward( self, epi = 0.05 , num_sepi = 20):
        total =[]
        for _ in range( num_sepi ):
            # evaluate based on value function ( one step look ahead )
            cur_state = self.env.reset()
            is_term = False
            reward_list = []
            step_num = 0
            # simulate the environment till is_term
            while not is_term:
                bs, bv, bm = self.state_dict_to_batch( cur_state )
                # epi greedy
                q_val_list = self.source_action_value.get_q_val( self.sess, bs, bv, bm )
                action = np.argmax( q_val_list.flatten() )
                [ next_state, reward, is_term, _ ] = self.env.step( action )
                reward_list.append( reward )
                cur_state = next_state
                step_num += 1
                if step_num % TERM_STEP == 0 or is_term:
                    is_term = True
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

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        self.load_model( model_file )
        reward = []
        for i in range( 100 ):
            reward_accu = 0
            cur_state = self.env.reset()
            is_term = False
            while not is_term:
                bs, bv, bm = self.state_dict_to_batch( cur_state )
                q_val = self.source_action_value.get_q_val( None, bs, bv, bm )
                action = np.argmax( q_val )
                next_state, r, _, _ = self.env.step( action )
                reward_accu += r
                cur_state = next_state
            reward.append( reward_accu )
        reward = np.array( reward )
        mean, std =  np.mean( reward ), np.std( reward )
        print( "mean: {} std {}".format( mean, std ) )
        return mean, std

    def draw( self, model_file = None ):
        # self.load_model( model_file )
        cur_state = self.env.reset()
        is_term = False
        rese = 0
        cur_step= 0
        while not is_term:
            bs, bv, bm = self.state_dict_to_batch( cur_state )
            q_val = self.source_action_value.get_q_val( None, bs, bv, bm )
            action = np.argmax( q_val )
            action = self.env.action_space.sample()
            next_state, r, is_term, _ = self.env.step( action )
            if r > 0:
                print( "reward", r, "before", rese )
                rese = 0
            else:
                rese += 1

            cur_state = next_state
            cur_step += 1
            if cur_step % TERM_STEP == 0:
                print( "end", cur_step )
                is_term = 1



    def burn_in_memory( self, burn_in_size ):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        burn_in_num = 0
        cur_state = self.env.reset()
        is_term = False
        step_num = 0
        while burn_in_num < burn_in_size:
            if is_term:
                cur_state = self.env.reset()
                is_term = False
                continue
            action = self.env.action_space.sample()
            next_state, reward, is_term, _ = self.env.step( action )
            self.replay_memory.append( ( cur_state, action, reward, next_state, is_term ) )
            cur_state = next_state
            burn_in_num += 1
            step_num += 1
            if step_num % TERM_STEP == 0:
                is_term = 1
                print( "term" )
        self.env.reset()
        print( "Done with burn in, memory size: {}".format( burn_in_size ) )

    def save_model(self, episode_num = 100 ):
        # Helper function to save your model / weights. 
        path = os.path.join( self.env_dir, str( episode_num ) ) + "/"
        os.mkdir( path )
        # self.saver.save( self.sess, path )
        # print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )
        cpt = dict()
        cpt[ "q_net" ] = self.source_action_value.network.state_dict()
        cpt[ "q_optim" ] = self.source_action_value.optim.state_dict()
        torch.save( cpt, path + "model.bin" )
        print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )


    def load_model(self, model_file):
        # Helper function to load an existing model.
        # self.saver.restore( self.sess, model_file)
        cpt = torch.load( model_file )
        self.source_action_value.network.load_state_dict( cpt[ "q_net" ] )
        self.source_action_value.optim.load_state_dict( cpt[ "q_optim" ] )

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
