import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class Actor_Network( nn.Module ):
    def __init__( self ):
        super( Actor_Network, self ).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l0 = nn.Linear( 16*5*5*3 + 3 + 1 + 1, 512 )
        self.l1 = nn.Linear( 512 , 256 )
        self.l2 = nn.Linear( 256, 256 )
        self.l_a_0 = nn.Linear( 256, 256 )
        self.l_a_2 = nn.Linear( 256, 3 )
        self.softmax = torch.nn.softmax( dim = 1 )

    def forward( self, delta_scent_batch, prev_vision_batch, cur_vision_batch, moved_batch, tong_batch ):
        delta_scent_batch = torch.tensor( delta_scent_batch, dtype = torch.float32 ).cuda()
        prev_vision_batch = torch.tensor( prev_vision_batch, dtype = torch.float32 ).cuda()
        cur_vision_batch = torch.tensor( cur_vision_batch, dtype = torch.float32 ).cuda()
        moved_batch = torch.tensor( moved_batch, dtype = torch.float32 ).cuda()
        tong_batch = torch.tensor( tong_batch, dtype = torch.float32 ).cuda()

        """ Compute vision"""
        conv = F.relu( self.conv1( cur_vision_batch ) )
        # print( conv.size() )
        conv = F.relu( self.conv2( conv ) )
        # print( conv.size() )
        conv = conv.view( conv.size( 0 ), -1 )

        prev_conv = F.relu( self.conv1( prev_vision_batch ) )
        # print( conv.size() )
        prev_conv = F.relu( self.conv2( prev_conv ) )
        # print( conv.size() )
        prev_conv = prev_conv.view( prev_conv.size( 0 ), -1 )

        conv = torch.cat( ( conv, prev_conv, delta_scent_batch, moved_batch, tong_batch ), dim = 1 )

        conv = F.relu( self.l0( conv ) )
        conv = F.relu( self.l1( conv ) )
        conv = F.relu( self.l2( conv ) )
        conv = F.relu( self.l_a_0( conv ) )
        conv = self.softmax( self.l_a_2( conv ) )
        return conv

class Critic_Network( nn.Module ):
    def __init__(self ):
        super( Critic_Network, self ).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.l0 = nn.Linear( 16*5*5*2 + 3 + 1 + 1 + 3, 512 )
        self.l1 = nn.Linear( 512 , 256 )
        self.l2 = nn.Linear( 256, 256 )
        self.l_a_0 = nn.Linear( 256, 256 )
        self.l_a_2 = nn.Linear( 256, 1 )

    def forward( self, delta_scent_batch, prev_vision_batch, cur_vision_batch, moved_batch, tong_batch, action_batch ):
        delta_scent_batch = torch.tensor( delta_scent_batch, dtype = torch.float32 ).cuda()
        prev_vision_batch = torch.tensor( prev_vision_batch, dtype = torch.float32 ).cuda()
        cur_vision_batch = torch.tensor( cur_vision_batch, dtype = torch.float32 ).cuda()
        moved_batch = torch.tensor( moved_batch, dtype = torch.float32 ).cuda()
        tong_batch = torch.tensor( tong_batch, dtype = torch.float32 ).cuda()
        action_batch = torch.tensor( action_batch, dtype = torch.float32 ).cuda()
        """ Compute vision"""
        conv = F.relu( self.conv1( cur_vision_batch ) )
        # print( conv.size() )
        conv = F.relu( self.conv2( conv ) )
        # print( conv.size() )
        conv = conv.view( conv.size( 0 ), -1 )

        prev_conv = F.relu( self.conv1( prev_vision_batch ) )
        # print( conv.size() )
        prev_conv = F.relu( self.conv2( prev_conv ) )
        # print( conv.size() )
        prev_conv = prev_conv.view( prev_conv.size( 0 ), -1 )

        conv = torch.cat( ( conv, prev_conv, delta_scent_batch, moved_batch, tong_batch, action_batch ), dim = 1 )

        conv = F.relu( self.l0( conv ) )
        conv = F.relu( self.l1( conv ) )
        conv = F.relu( self.l2( conv ) )
        conv = F.relu( self.l_a_0( conv ) )
        conv = self.l_a_2( conv )
        return conv

