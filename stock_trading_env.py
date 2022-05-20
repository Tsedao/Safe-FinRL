import gym
import numpy as np
import math


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history,
                       steps=730,
                       step_size=1,
                       look_back=50,
                       start_idx=0,
                       feature_num=4,
                       valid_env=False):
        """
        Args:
            history: (num_stocks, timestamp, feaure_num)
                    sp-open, bp-open, sp_close, bp_close are the last four feature
                    which are used to calculate return
            steps: the total number of steps to simulate
            look_back: observation window, must be less than 50
            start_idx: the idx to start. Default is None and random pick one.

        Returns:
            reset():
            _step(): obs, done, ground_truth_obs

        """
        if len(history.shape) == 2:
            history = np.expand_dims(history,axis=0)


        import copy
        self.step_size = step_size
        self.steps = steps
        self.look_back = look_back
        self.start_idx = start_idx

        # make immutable class
        self._data = history.copy()[...,-feature_num:]  # all data
        self.valid_env = valid_env
    def _step(self):

        self.step += self.step_size

        obs = self.data[:, self.step:self.step + self.look_back, :].copy()

        # used for compute optimal action and reward
        ground_truth_obs = self.data[:, self.step + self.look_back:self.step + self.look_back + 1, :].copy()
        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0
        # get data for this episode, each episode might be different.
        if self.valid_env:
            self.idx = self.look_back
        else:
            self.idx = np.random.randint(
                low=self.look_back, high=self._data.shape[1] - self.steps)

        # print('Start date: %s, End date: %s, Length: %s' %(self.idx,self.idx+self.steps,self.steps))
        data = self._data[:, self.idx - self.look_back:self.idx + self.steps + 1, :]
        self.data = data
        # self.timestamp = self.timestamp[self.idx - self.window_length:self.idx + self.steps + 1]
        return self.data[:, self.step:self.step + self.look_back, :].copy(), \
               self.data[:, self.step + self.look_back:self.step + self.look_back + 1, :].copy()

class StockTradingSim(object):
    """
    Multi-Stock Trading Simulation from time t to t+1
    """

    def __init__(self, steps=730,
                       trading_cost=0.0004,
                       num_stock = 1,
                       balance=100000,
                       unit = 0.0001):

        self.trading_cost = trading_cost
        self.num_stock = num_stock
        self.steps = steps
        self.balance_init = balance
        self.balance = balance
        self.unit = unit

    def step(self, action, price, next_price):
        """

        Args:
            action - (num_stock, ) ranging in [-h_max, h_max]
            balance - (1, ),       t-th balance
            shares - (num_stock,) t-th holding position for each asset
            prices - (num_stock, 2) t-th time of selling (bid) price and buying (ask) price
            next_prices - (num_stock, 2) t+1-th time of buying (ask) price and selling (bid) price

        Return:
            balance - at t+1
            shares - at t+1
            reward - at t, calculated by (amount t+1) / amount t
            done - at t, if total wealth samller than 0 than it is done

        """
        return self._step(action, price, next_price)

    def _step(self, action, price, next_price):

        assert action.shape == self.shares.shape

        # calculate initial wealth before action at time t
        initial_total_assets = self.balance + sum(s * p for s, p in zip(self.shares,price[:,0]) if s >= 0) + \
                                            sum(s * p for s, p in zip(self.shares,price[:,1]) if s < 0)

        argsort_actions = np.argsort(action)
        sell_index = argsort_actions[: np.where(action < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(action > 0)[0].shape[0]]

        for index in sell_index:
            # action[index] = self._do_sell_normal(action,price,index)

            sell_position, short_position = self._do_short_normal(action, price, index)

        for index in buy_index:
            # action[index] = self._do_buy_normal(action,price,index)

            buy_position, long_position = self._do_long_normal(action, price, index)

        # calculate updated wealth after action at time t + 1
        updated_total_assets = self.balance + sum(s * p for s, p in zip(self.shares,next_price[:,0]) if s>=0) + \
                                                sum(s * p for s, p in zip(self.shares,next_price[:,1]) if s<0)

        reward = 100*math.log(updated_total_assets / initial_total_assets) if updated_total_assets > 0 and initial_total_assets > 0 else -1



        return self.balance.copy(), self.shares.copy(), reward, updated_total_assets



    def _do_short_normal(self, action, price, index):
        action = action[index]
        shares = self.shares[index]
        assert action < 0

        action_unit = np.array(action) // np.array(self.unit)
        action = action_unit * self.unit
        # price 0: selling price 1: buying price

        # CONSTRAINTS
        max_unit  = (self.balance_init / (price[index,0]* (1 + self.trading_cost))) // self.unit
        max_share = max_unit * self.unit
        action = -min(max_share,abs(action))

        # action [-1]  shares [0.5]
        short_position = abs(max(min(action + shares,0),action))

        sell_position = max(min(abs(action),shares),0)

        # print('Action {}, Selling {} shares, shorting {} shares, original holding shares {}'.format(action,
        #                                                                                         sell_position,
        #                                                                                         short_position,
        #                                                                                         self.shares[index]))

        #

        sell_amount = price[index,0] * (sell_position) * (1- self.trading_cost)         # sell the current position
        short_amount = price[index,0] * (short_position) * (1- self.trading_cost)       # sell the broker's position
        # update balance
        self.balance += sell_amount
        self.balance += short_amount

        # update shares
        self.shares[index] += action

        return sell_position, short_position

    def _do_long_normal(self, action, price,index):

        action = action[index]
        assert action > 0

        action_unit = np.array(action) // np.array(self.unit)
        action = action_unit * self.unit


        # TODO: Modify the code to suitable for doing long first when shares <0
        # error may happen here

        # CONSTRAINTS
        max_unit  = (self.balance / (price[index,0]* (1 + self.trading_cost))) // self.unit
        max_share = max_unit * self.unit
        action = min(max_share,action)


        # action 1 shares [-0.5]
        long_position = min(abs(min(self.shares[index],0)),action)
        buy_position = action - long_position

        # print('Action {}, Buying {} shares, longing {} shares, original holding shares {}'.format(action,
        #                                                                                         buy_position,
        #                                                                                         long_position,
        #                                                                                         self.shares[index]))

        long_amount = price[index,1] * (long_position) * (1 + self.trading_cost)     # buy the broker's positions back
        buy_amount = price[index,1] * (buy_position) *  (1 + self.trading_cost)      #  buy us extra positions

        self.balance -= long_amount
        self.balance -= buy_amount

        # update shares
        self.shares[index] += action

        return buy_position, long_position




    def reset(self):
        self.infos = []
        self.balance = self.balance_init
        self.shares = np.zeros(self.num_stock)


class StockTradingEnv(gym.Env):
    """
    Multi-Stock Trading Environment From time t=0 to t=T
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 steps=730,  # 2 years
                 step_size=1,
                 trading_cost=0.0004,
                 look_back=50,
                 start_idx=0,
                 feature_num=4,
                 balance_init = 100000,
                 h_max = 1,
                 valid_env = False,
                 verbose = False
                 ):
        """
        An environment for multistock trading.
        Params:
            history -
            steps - steps in episode
            trading_cost - cost of trade as a fraction
            look_back - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            balance -
            feature_num -
            h_max -
            valid_env -
        """
        self.look_back = look_back

        if len(history.shape) == 2:
            history = np.expand_dims(history, axis=0)
        self.num_stock = history.shape[0]
        self.start_idx = start_idx
        self.verbose = verbose
        self.h_max = h_max
        self.balance_init = balance_init
        # self.balance = balance

        self.src = DataGenerator(history,
                                steps=steps,
                                step_size=step_size,
                                look_back=look_back,
                                start_idx=start_idx,
                                feature_num=feature_num,
                                valid_env = valid_env)

        self.sim = StockTradingSim(steps=steps,
                                   trading_cost=trading_cost,
                                   num_stock=self.num_stock,
                                   balance=balance_init)
        # openai gym attributes
        # action will be the selling/buying shares from -1 to 1 for each asset
        self.action_space = gym.spaces.Box(-self.h_max, self.h_max, shape=(self.num_stock,),
                                           dtype=np.float32)

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.num_stock, look_back,
                                                        history.shape[-1]), dtype=np.float32)

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        We use close price to calculate the total wealth
        Args:
            action - [num_stock,]

        """

        obs, done1, ground_truth_obs = self.src._step()

        ## obs (aseet_num, look_back, feature)
        price = obs[:,-1,-2:]
        next_price = ground_truth_obs[:,0,-2:]
        # 0 sell 1 sell
        market_gain = next_price[:,-1] / price[:,-1]

        action = action * self.h_max
        action = np.clip(action, -self.h_max, self.h_max)

        balance, share, reward,update_total_asset = self.sim.step(action,price,next_price)

        done2 = update_total_asset <= 0

        # update balance and share states

        temp = np.copy(self.obs_balance[1:])
        self.obs_balance[:-1],  self.obs_balance[-1:] = temp, balance

        temp = np.copy(self.obs_shares[:,1:])
        self.obs_shares[:,:-1],  self.obs_shares[:,-1:] = temp, share.copy()

        info = {}
        info['market_gain'] = 100*np.log(market_gain)
        info['action_reward'] = reward
        info['total_wealth'] = update_total_asset
        info['balance'] =  balance
        info['share'] =  share.copy()
        info['steps'] = self.src.step

        self.infos.append(info)

        return (self.obs_balance.copy(),self.obs_shares.copy(),obs.copy()), reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []

        self.sim.reset()

        obs, ground_truth_obs = self.src.reset()


        info = {}
        info['balance'] = self.sim.balance

        self.obs_balance = np.ones(self.look_back)*self.sim.balance
        self.obs_shares = np.zeros(shape = (self.num_stock, self.look_back))
        return (self.obs_balance.copy(),self.obs_shares.copy(),obs.copy()), info
