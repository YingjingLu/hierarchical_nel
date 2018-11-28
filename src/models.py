import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import pickle
from networks import *

class Actor( object ):

    def __init__(self, lr ): 

        self.network = Actor_Network( ).cuda()

        self.optim = optim.Adam( self.network.parameters(), lr=lr )

    def get_q_val( self, delta_scent_batch, prev_vision_batch, 
                         cur_vision_batch, moved_batch, tong_batch ):
        # print( scent_batch.shape, vision_batch.shape, moved_batch.shape )
        q_val_next = self.network( delta_scent_batch, prev_vision_batch, 
                                   cur_vision_batch, moved_batch, tong_batch )
        return q_val_next.cpu().detach().numpy()

    def get_loss( self, delta_scent_batch, prev_vision_batch, 
                        cur_vision_batch, moved_batch, tong_batch, R_batch, V_batch ):
        R_batch = torch.tensor( R_batch, dtype = torch.float32 ).cuda()
        V_batch = torch.tensor( V_batch, dtype = torch.float32 ).cuda()
        q_val = self.network( delta_scent_batch, prev_vision_batch, 
                              cur_vision_batch, moved_batch, tong_batch )
        one_hot_action = torch.zeros( q_val.size(), dtype = torch.float32 ).cuda()
        action = torch.argmax( q_val, dim = 1 )
        for i in range( action.size( 0 ) ):
            one_hot_action[ i, action[ i ] ] = 1
        reward = ( R_batch - V_batch ) * one_hot_action 

        loss = torch.mean( torch.sum( torch.log( q_val ) * reward, dim = 1 ) ) 
        return loss

    def optimize( self, delta_scent_batch, prev_vision_batch, 
                        cur_vision_batch, moved_batch, tong_batch, R_batch, V_batch ):
        self.optim.zero_grad()
        loss = -1.0*self.get_loss( delta_scent_batch, prev_vision_batch, 
                                   cur_vision_batch, moved_batch, tong_batch, R_batch, V_batch )
        loss.backward()
        self.optim.step()
        return loss.cpu().detach().item()

class Critic( object ):

    def __init__(self, lr ):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.network = Critic_Network( )

        self.network = self.network.cuda()

        self.loss = nn.MSELoss().cuda()

        self.optim = optim.Adam( self.network.parameters(), lr=lr )

    def get_q_val( self, delta_scent_batch, prev_vision_batch, 
                         cur_vision_batch, moved_batch, tong_batch, action_batch ):
        # print( scent_batch.shape, vision_batch.shape, moved_batch.shape, goal_batch.shape )
        q_val_next = self.network( delta_scent_batch, prev_vision_batch, 
                                   cur_vision_batch, moved_batch, tong_batch, action_batch )
        return q_val_next.cpu().detach().numpy()

    def get_loss( self, delta_scent_batch, prev_vision_batch, 
                        cur_vision_batch, moved_batch, tong_batch, action_batch, reward_batch ):
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda().view( -1, 1 )
        q_val = self.network( delta_scent_batch, prev_vision_batch, 
                              cur_vision_batch, moved_batch, tong_batch, action_batch )
        q_val = ( q_val ).view(-1, 1)
        loss = self.loss( q_val, reward_batch )
        return loss

    def optimize( self, delta_scent_batch, prev_vision_batch, 
                        cur_vision_batch, moved_batch, tong_batch, action_batch, reward_batch ):
        self.optim.zero_grad()
        reward_batch = torch.tensor( reward_batch, dtype = torch.float32 ).cuda()
        loss = self.get_loss( delta_scent_batch, prev_vision_batch, 
                              cur_vision_batch, moved_batch, tong_batch, action_batch, reward_batch )
        loss.backward()
        self.optim.step()
        return loss.cpu().detach().item()