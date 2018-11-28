import numpy as np

def state_dict_to_batch( state_dict_list ):
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

def action_to_onehot( action_batch, action_size = 3 ):
    if type( action_batch ) == int or type( action_batch ) == float or type( action_batch ) == np.int64 or type( action_batch ) == np.int:
        action_batch = int( action_batch )
        res = np.zeros( action_size, dtype = np.float32 )
        res[ action_batch ] = 1
        return res.reshape( -1, action_size )
    batch_size = len( action_batch )
    res = np.zeros( ( batch_size, action_size ), dtype = np.float32 )
    for i in range( len( action_batch ) ):
        res[ i, action_batch[ i ] ] = 1
    return res

def goal_to_onehot( goal_batch, goal_size = 3 ):
    
    if type( goal_batch ) == int or type( goal_batch ) == float or type( goal_batch ) == np.int64 or type( goal_batch ) == np.int:
        goal_batch = int( goal_batch )
        res = np.zeros( goal_size, dtype = np.float32 )
        res[ goal_batch ] = 1
        return res.reshape( -1, goal_size )
    batch_size = len( goal_batch )
    res = np.zeros( ( batch_size, goal_size ), dtype = np.float32 )
    for i in range( len( goal_batch ) ):
        res[ i, goal_batch[ i ] ] = 1
    return res

def construct_batch( sample_list ):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    tong_batch = []

    for transitions in sample_list:
        [ cur, a, r, _next, tong ] = transitions
        state_batch.append( cur )
        action_batch.append( a )
        reward_batch.append( r )
        next_state_batch.append( _next )
        tong_batch.append( tong )

    # convert everything to np array
    cur_bs, cur_bv, cur_bm = state_dict_to_batch( state_batch )
    next_bs, next_bv, next_bm = state_dict_to_batch( next_state_batch )
    action_batch = action_to_onehot( action_batch )
    reward_batch = np.array( reward_batch ).reshape( -1, 1 )
    tong_batch = np.array( tong_batch ).reshape( -1, 1 )

    return next_bs-cur_bs, cur_bv,next_bv, next_bm, tong_batch