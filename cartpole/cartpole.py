import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, flag
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import collections

# chainer model used as Q function
class DeepQModel(Chain):
    def __init__(self , isz, osz):
        super(DeepQModel, self).__init__(
            mid=L.Linear(isz, 128),  # dense layer with relu activation
            out=L.Linear(128, osz),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, x):
        h = F.relu(self.mid(x))
        y = self.out(h)
        return y

# converts np array to chainer variable
def tv(x, v = flag.OFF):
    return Variable(x.astype('float32'), volatile=v)

# agent class
class DeepQ_Discrete():
    def __init__(self, isz, n_a):

        self.s_buff = collections.deque( [], 1 ) # num of states to consider as one big MDP state

        self.n_a = n_a  # num of actions
        self.isz = isz  # size of input vector
        self.training = True # is training?

        self.Q = DeepQModel(isz * self.s_buff.maxlen, n_a)
        self.Qp = DeepQModel(isz * self.s_buff.maxlen, n_a)

        self.Qp.copyparams(self.Q)

        self.pr = 0.0

        class MSE(Chain):
            def __init__(self):
                super(MSE, self).__init__()

            def __call__(self, X, Y, A, Q):
                P = Q(X)
                P = F.select_item(P, Variable(np.array(A).astype('int32')))
                return F.mean_squared_error(Y, P)

        if self.training:
            self.d = 0.99  # discount
            self.idx = 0 # steps counter
            self.upd = 0 # updates counter
            self.batch = 32 # batch size
            self.batches = 64 # number of update batches
            self.random_exp = 256.0 # the larger the value the more random exploration will be done

            self.pr = 1.0 # probability of choosing random action

            self.loss = MSE()
            self.opt = optimizers.Adam()
            self.opt.setup(self.Q)

            self.Q.zerograds()
            self.r_buff = collections.deque([], 1000) # num of games to keep in replay buffer
            self.r = [] # replay array for 1 game; nice to have it separately, in case of multithreading later

    def reset(self):
        # resets the network.  if training, resets the state buffer

        if self.training:
            # add game array in the replay buffer
            if self.r:
                self.r_buff.append(self.r)
            # empty game replay array
            self.r = []
            # zero input buffer
            for _ in range(self.s_buff.maxlen):
                self.s_buff.append( np.zeros((1, self.isz)) )

    def get_mdp_obs(self, obs):
        # add observation to buffer, concatenate buffer into vector
        self.s_buff.append(obs)
        cc = np.column_stack(self.s_buff)

        return cc

    def next(self, obs):

        # add observation to buffer
        mdp_obs = self.get_mdp_obs(obs)

        x = Variable(mdp_obs.astype('float32'))
        pa = self.Q(x)  # q distribution over actions

        # choose action, with self.pr probability at random
        a = np.argmax(pa.data) if np.random.rand() > self.pr else np.random.randint(0, self.n_a)

        if self.training:
            self.r.append([mdp_obs, a]) # save observation and action
            self.idx += 1

        return a

    def feedback(self, reward):
        # associate feedback with recent action
        self.r[-1].append(reward)

    def train(self, par=None):

        self.pr = min(1.0, 0.02 + self.random_exp / (self.idx + 1.0))
        self.upd = self.upd + 1

        # generate dataset from replay buffer
        if self.upd % 3 == 0:
            self.Qp.copyparams(self.Q)

        # save the buffer if any
        if self.r:
            self.r_buff.append(self.r)
            self.r = []
        
        # process batches
        for repeat in range(self.batches):

            X = []
            A = []
            Y = []

            ln = len(self.r_buff)

            I = np.random.choice(ln, min(ln, self.batch), replace=False)

            XQ = []

            for i in I:
                game = self.r_buff[i]
                for x, a, r in reversed(game):
                    XQ.append(x)

            XQ = tv(np.row_stack(XQ), v=flag.ON)

            Qmax = F.max( self.Qp(XQ), axis=1)
            Qmax = Qmax.data

            idx = 0

            for i in I:
                game = self.r_buff[i]
                q_max = 0.0

                for x, a, r in reversed(game):

                    y = q_max + r

                    X.append(x)
                    Y.append(y)
                    A.append(a)

                    # update q max
                    q_max = self.d * Qmax[idx]
                    idx += 1

            X = tv(np.row_stack(X))
            Y = tv(np.squeeze(np.row_stack(Y)))

            self.Q.zerograds()
            loss = self.loss(X, Y, A, self.Q)

            # update the parameters of the agent
            loss.backward()
            self.opt.update()

import gym

buff = collections.deque([], 100)
env = gym.make('CartPole-v0')
env.monitor.start("cartpole", force=True)

MAX_STEPS = env.spec.timestep_limit

# create a deep q network
actor = DeepQ_Discrete(4,2)

for episode in xrange(200):

    actor.reset()
    observation = env.reset()
    buff.append(0)

    for t in xrange(MAX_STEPS):
        # act in the environment
        action = actor.next([observation])
        observation, reward, done, info = env.step(action)
        buff[-1] += reward

        actor.feedback(reward)

        if done:
            break
    
    # update the neural net after every episode
    actor.train()
    print buff[-1], "avg. reward:", np.mean(buff), "iter:", episode

env.monitor.close()