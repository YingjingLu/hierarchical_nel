import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


def costum_loss(yTrue, yPred):
    epsilon = 1e-6
    s = K.mean(tf.multiply(yTrue, K.log(yPred + epsilon)))
    return -1 * s


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n
        self.lr = lr
        self.critic_lr = critic_lr
        self.num_action = 4
        self.num_state = 8
        self.max_iter = 50000

        self.model.compile(loss=costum_loss,
                            optimizer=keras.optimizers.Adam(lr=self.lr))

        self.critic_model.compile(loss='mean_squared_error',
                                  optimizer=keras.optimizers.Adam(lr=self.critic_lr))

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def generate_batch(self, states, actions, R, V):
        X = states
        pred_y = self.model.predict(X)
        y = np.zeros((len(actions), self.num_action))
        y[np.arange(len(y)), actions.astype(int)] = R-V
        return (X, y)
    
    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        test_mean = []
        test_std = []
        loss = np.zeros(self.max_iter)
        loss2 = np.zeros(self.max_iter)
        for episode in range(self.max_iter):
            (states, actions, rewards) = self.generate_episode(env)
            V = self.critic_model.predict(states)
            T = len(actions)
            V = V.reshape((T,))
            R = np.zeros(T) 
            discount = gamma ** np.arange(self.n)
            for i in range(T-1):
                if i + self.n >= T:
                    R[i] = np.sum(discount[0:T-i]*rewards[i:])
                else:
                    R[i] = np.sum(discount*rewards[i:i+self.n]) + gamma ** self.n * V[i+self.n]
            (X, y) = self.generate_batch(states, actions, R, V)
            loss[episode] = self.model.train_on_batch(X, y)
            loss2[episode] = self.critic_model.train_on_batch(X, R)
            if episode % 300 == 0 or episode == self.max_iter-1:
                if episode != 0:
                    print(np.mean(loss[(episode - 300):episode]))
                    print(np.mean(loss2[(episode - 300):episode]))
                res = self.test(env, 100)
                print(res)
                test_mean.append(res[0])
                test_std.append(res[1])
                if res[0] > 200 and res[1] < 40:
                    break
        return test_mean, test_std

    def test(self, env, num_epi = 100):
        rewards = np.zeros(num_epi)
        for i in range(num_epi):
            curr_s = np.asarray(env.reset())
            self.num_state = len(curr_s)
            while True:
                curr_s = curr_s.reshape((1,self.num_state))
                prob_a = self.model.predict(curr_s)[0]
                a = np.random.choice(self.num_action, p=prob_a)
                next_s, reward, done, _ = env.step(a)
                rewards[i] += reward
                if done:
                    break
                else:
                    curr_s = next_s
        return np.mean(rewards), np.std(rewards)
    
    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        curr_s = np.asarray(env.reset())
        self.num_state = len(curr_s)
        while True:
            states.append(curr_s)
            curr_s = curr_s.reshape((1,self.num_state))
            prob_a = self.model.predict(curr_s)[0]
            a = np.random.choice(self.num_action, p=prob_a)
            actions.append(a)
            next_s, reward, done, _ = env.step(a)
            rewards.append(reward/100)
            if done:
                break
            else:
                curr_s = next_s
            
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
            
        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

def construct_critic_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
                units=16, activation='sigmoid', input_dim=8
            ))
    model.add(keras.layers.Dense(
                units=32, activation='sigmoid'
            ))
    model.add(keras.layers.Dense(
                units=128, activation='sigmoid'
            ))
    model.add(keras.layers.Dense(
                units=1, activation='linear'
            ))
    return model


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    critic_model = construct_critic_model()
    AC = A2C(model, lr, critic_model, critic_lr, n = 20)
    test_mean, test_std = AC.train(env)
    np.save(f"mean_{n}", test_mean)
    np.save(f"std_{n}", test_mean)

    

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
