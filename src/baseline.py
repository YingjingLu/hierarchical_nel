import gym
import nel
import keras, numpy as np, gym, sys, copy, argparse
#from baselines import deepq

env = gym.make('NEL-v0')
#model = deepq.learn(env, "mlp")


class Network():
    #This class essentially defines the network architecture
    def __init__(self, lr_cnn = 0.001, lr_mlp = 0.001):
        self.alpha_cnn = lr_cnn
        self.alpha_mlp = lr_mlp
        self.num_out = 5
        self.cnn = self.generate_cnn(self.num_out) #Defines a CNN architecture for visual oberservation
        self.mlp = self.generate_mlp(self.num_out+4) #Defines a MLP architecture for state.



    def generate_cnn(self, num_out = 5):
        #input RGB image of size 11x11x3, output vector of length num_out.
        model = keras.models.Sequential()
        model.add(Conv3D(4, kernel_size=(6, 6, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(num_out, activation='linear'))
        model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(lr=self.alpha_cnn))
        return model

    def generate_mlp(self, num_inp = 9):
        #num_inp = output from CNN + scent + moved
        mlp_model = keras.models.Sequential()
        mlp_model.add(keras.layers.Dense(
                units=16, activation='relu', input_dim=4
        ))
        mlp_model.add(keras.layers.Dense(
                units=32, activation='relu'
        ))
        mlp_model.add(keras.layers.Dense(
                units=128, activation='relu'
        ))
        mlp_model.add(keras.layers.Dense(
                units=3, activation='linear'
        ))
        mlp_model.compile(
                loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(lr=self.alpha_mlp)
        )
        return mlp_model

    def predict(self, state):
        vision = []
        scent = []
        moved = []
        for i in len(state):
            vision.append(state[i]['vision'])
            scent.append(state[i]['scent'])
            moved.append([state[i]['moved']])
        vision = np.asarray(vision)
        scent = np.asarray(scent)
        moved = np.asarray(moved)
        cnn_output = self.cnn.predict(vision)
        state = np.concatenate((scent,cnn_opt,moved), axis = 1)
        res = self.mlp.predict(state)
        return res

    def train_on_batch(self, state, y):
        vision = []
        scent = []
        moved = []
        for i in len(state):
            vision.append(state[i]['vision'])
            scent.append(state[i]['scent'])
            moved.append([state[i]['moved']])
        vision = np.asarray(vision)
        scent = np.asarray(scent)
        moved = np.asarray(moved)
        
        
        

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.cnn.save_weights(suffix + 'cnn.h5')
        self.mlp.save_weights(suffix + 'mlp.h5')
        return 0

    def load_model_weights(self,weight_file_cnn, weight_file_mlp):
        # Helper funciton to load model weights. 
        self.cnn.load_weights(weight_file_cnn)
        self.mlp.load_weights(weight_file_mlp)
        return 0


class Replay_Memory():
    #This class saves and replays the memory, same implementation as in HW2
    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.memory_dims = None
        self.memory = None
        # memory_load: number of transitions currently stored in memory
        self.memory_load = 0
        self.memory_size = memory_size
        self.burn_in = burn_in
        
    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        # Don't worry about the case where memory doesn't have at least 32 entries: burn in episodes sufficient
        samp = min([self.memory_load, self.memory_size])
        sample_indices = np.random.choice(samp,
                                          size=batch_size,
                                          replace=False)
        res = []
        for i in sample_indices:
            res.append(self.memory[i])
        return res

    def append(self, transition):
        # Induces the dimension of transition from the data fed in
        # In the case of CartPole
        # memory[i][0:4] : State[i]
        # memory[i][4] : Action[i]
        # memory[i][5] : Reward[i]
        # memory[i][6:10] : State'[i]
        # memory[i][10] : isterminate
        self.memory[self.memory_load % self.memory_size] = transition
        self.memory_load += 1


class DDQN_agent():
# This class implements the double DQN agent
    
    
    def __init__(self, environment_name, render=False):
        #initializing the network approximation for Q function,
        #the structure is the same as used by DQN_agent.
        self.inner = Network()
        #initializing the network for selecting action Q function as described in paper.
        self.outter = Network()
        self.render = render
        self.use_memory = True
        self.batchsize = 32
        self.epsilon = np.linspace(0.5,0.05,100000)
        self.replace_freq = 200
        #We replace the outter network by inner work every 200 iteration.
        if self.use_memory:
            self.memory = Replay_Memory()
            self.max_iter = 10000
            self.num_iter = 0
        else:
            self.max_episodes = 10000
            self.num_episodes = 0
        self.env = gym.make(environment_name)
        self.state = self.env.reset()

    def epsilon_greedy_policy(self, q_values, iteration):
        # Creating epsilon greedy probabilities to sample from.
        if iteration >= 100000:
            epsilon = 0.05
        else:
            epsilon = self.epsilon[iteration]
        choice = int(np.random.choice(2, 1, p = [epsilon, 1-epsilon]))
        if choice == 0:
            return self.env.action_space.sample()
        else:
            q = q_values.predict([self.state])
            return np.argmax(q)
            
        

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        choice = int(np.random.choice(2, 1, p = [0.05, 0.95]))
        if choice == 0:
            return self.env.action_space.sample()
        else:
            q = q_values.predict([self.state])
            return (np.argmax(q))

    def train
    
    def burn_in_memory(self):
        #We burn in 10000 memory before training the model. 
        Q = self.inner.model
        ind = 0
        for episode in range(self.max_iter):
            s0 = self.env.reset()
            self.state = s0
            while True:
                ind += 1
                a = self.env.action_space.sample()
                temp = self.env.step(a)
                transition = dict()
                transition["curr_state"] = self.state
                transition["action"] = a
                transition["reward"] = temp[1]
                transition["next_state"] = temp[0]
                transition["is_term"] = temp[2]
                self.memory.append(transition)
                if temp[2]: #terminate
                    break
                else:
                    self.state = temp[0]
            if ind >= self.memory.burn_in:
                break
    











        
