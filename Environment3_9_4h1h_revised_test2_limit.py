from importlib.resources import path
from socket import close
#from Env import FLAT, HOLD
#import data_processing2
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path
import torch
from sklearn.preprocessing import normalize, MinMaxScaler
import math
import pickle

#position constant
LONG = 1
SHORT = -1 
FLAT = 0 #ga masuk pasar

#action constant
BUY = 1
SELL = -1
HOLD = 0

class Env(gym.Env):

    def __init__(self,path):
        self.path = path
        self.actions = ["BUY", "SELL", "HOLD"]
        #self.fee = 0.1 #website tradimo
        self.seed(1234)
        self.file_list = []
        self.load_csv()

        #n_features
        self.window_size = 1
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features+2)#5)

        #define action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_csv(self):
        #df = data_processing2.ExtractFeature2(self.path)
        #self.df = df.features4()
        self.df = pd.read_csv(self.path)
        #self.df = pd.DataFrame(columns=['close'])
        #self.df['close'] = df['close'].values
        ## selected manual features
        #feature_list = ['open1', 'close1', 'high1','low1','open2', 'close2', 'high2','low2','open3', 'close3', 'high3','low3']
        #feature_list = ['upper1', 'body1', 'lower1','color1','upper2', 'body2', 'lower2','color2','upper3', 'body3', 'lower3','color3']
        #feature_list = ['close1', 'close2', 'close3']
        #feature_list = ['upper', 'body', 'lower','color']
        #feature_list = ['open', 'close', 'high','low']
        #self.df.dropna(inplace=True) # drops Nan rows
        #self.df = self.df.reset_index(drop=True)
        self.closingPrices = self.df['close'].values
        #self.positionDf = self.df['action'].values
        #self.rewDf = self.df['PL'].values
        self.colors = self.df['color'].values
        self.colorsD = self.df['colorD'].values
        self.colors4h = self.df['color4h'].values
        self.colors1h = self.df['color1h'].values
        #self.df = self.df.loc[:,['color', '']]#self.df.drop(columns=['close'])
          
        print(self.df)     

    def seed(self, SEED):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        
    def step(self, action):

        #if self.done:
        #    return self.state, self.reward, self.done, {}
        #self.reward = 0

        print('action env:', action)
        #self.action = HOLD
        if action == BUY:
            print('buy')
            if self.position == FLAT and self.delta_time >= 48-2:
                print('open long at', self.closingPrice)
                self.arr_long_open_prices.append(self.closingPrice)
                self.arr_long_open_index.append(self.current_tick-1)
                profit = 0 #initial reward kalo di-sigmoid jadi 0.5
                self.position = LONG
                pos = self.position
                self.action = BUY
                self.entry_price = self.closingPrice
                #self.exit_price = self.closingPrice
                #self.lowest_price = self.entry_price
                #self.open_position_balance = 0.1*self.balance
                #self.position_balance = 0.1*self.balance
                self.lowest_price = self.closingPrice
                #self.clone_balance -= self.open_position_balance
                #self.krw_balance -= self.open_position_balance
                #profit = self.get_profit()
                #self.time_end = self.current_tick
                self.sum_delta_time += self.delta_time
                #self.arr_delta_time.append(self.delta_time)
                if self.delta_time > self.highest_delta_time:
                    self.highest_delta_time = self.delta_time
                if self.delta_time < self.lowest_delta_time:
                    self.lowest_delta_time = self.delta_time
                self.delta_time = 0
                self.holding_time += 1#0
                drawdown = 0
                self.reward = 0
            elif self.position == FLAT:
                self.position = FLAT
                pos = self.position
                print('no position')
                profit = 0# self.get_profit()
                self.action = HOLD
                self.delta_time += 1
                drawdown = 0
                self.reward = 0
            #hold position long
            elif self.position == LONG:
                print('long')
                self.position = LONG
                pos = self.position
                profit = self.get_profit()
                self.action = BUY
                if self.closingPrice < self.lowest_price:
                    self.lowest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            #close position short
            elif self.position == SHORT and self.holding_time >= 48-1:
                pos = self.position
                print('close short at ',self.closingPrice)
                profit = self.get_profit()
                self.action = SELL#BUY
                #self.exit_price = self.closingPrice
                #self.reward += ((self.entry_price - self.exit_price) * 10000) * 10
                #self.profit = ((self.entry_price - self.exit_price)* 1000)#10000) #* 10
                #print('profit close position: ',self.profit)
                #self.krw_balance = self.krw_balance + self.reward - self.fee
                self.balance += profit 
                #self.clone_balance += self.position_balance - self.fee
                if self.closingPrice > self.highest_price:
                    self.highest_price = self.closingPrice
                self.holding_time += 1
                if profit > 0:
                    self.short_profit += 1
                    self.sum_profit += profit
                    self.arr_profit.append(profit)
                    self.sum_holding_time_pos += self.holding_time
                    self.arr_short_close_prices_profit.append(self.closingPrice)
                    self.arr_short_close_index_profit.append(self.current_tick-1)
                else:
                    self.sum_loss += profit
                    self.sum_holding_time_neg += self.holding_time
                    self.arr_short_close_prices_loss.append(self.closingPrice)
                    self.arr_short_close_index_loss.append(self.current_tick-1)
                #self.entry_price = 0
                self.n_position += 1
                self.n_short += 1
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                if self.holding_time > self.highest_holding_time:
                    self.highest_holding_time = self.holding_time
                if self.holding_time < self.lowest_holding_time:
                    self.lowest_holding_time = self.holding_time
                self.sum_holding_time += self.holding_time
                self.arr_holding_time.append(self.holding_time)
                #self.n_no_position += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
                self.holding_time = 0
                self.delta_time = 0
                self.position = FLAT
                self.n_no_position += 1
            elif self.position == SHORT:
                print('short')
                profit = self.get_profit()
                self.position = SHORT
                pos = self.position
                self.action = SELL
                if self.closingPrice > self.highest_price:
                    self.highest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            
        elif action == SELL:
            print('sell')
            #open position short
            if self.position == FLAT and self.delta_time >= 48-2:
                print('open short at ',self.closingPrice)
                self.arr_short_open_prices.append(self.closingPrice)
                self.arr_short_open_index.append(self.current_tick-1)
                profit = 0 #initial reward
                self.position = SHORT
                pos = self.position
                self.holding_time += 1
                self.action = SELL
                self.entry_price = self.closingPrice
                #self.exit_price = self.closingPrice
                #self.highest_price = self.entry_price
                #self.open_position_balance = 0.1*self.balance
                #self.position_balance = 0.1*self.balance
                self.highest_price = self.closingPrice
                #self.clone_balance -= self.open_position_balance
                #self.time_end = self.current_tick
                self.sum_delta_time += self.delta_time
                #self.arr_delta_time.append(self.delta_time)
                if self.delta_time > self.highest_delta_time:
                    self.highest_delta_time = self.delta_time
                if self.delta_time < self.lowest_delta_time:
                    self.lowest_delta_time = self.delta_time
                self.delta_time = 0
                self.holding_time += 1
                drawdown =0 # self.g_drawdown()
                self.reward = 0
            elif self.position == FLAT:
                self.position = FLAT
                pos = self.position
                print('no position')
                profit = 0# self.get_profit()
                self.action = HOLD
                self.delta_time += 1
                drawdown = 0
                self.reward = 0
            #hold position short
            elif self.position == SHORT:
                print('short')
                profit = self.get_profit()
                self.position = SHORT
                pos = self.position
                self.action = SELL
                if self.closingPrice > self.highest_price:
                    self.highest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            elif self.position == LONG and self.holding_time >= 48-1:
                pos = self.position
                print('close long at ',self.closingPrice)
                profit = self.get_profit()
                self.action = BUY#SELL
                #self.exit_price = self.closingPrice
                #self.reward += ((self.exit_price - self.entry_price) * 10000) * 10
                #self.profit = ((self.exit_price - self.entry_price)* 1000)#10000) #* 10
                #print('profit close position: ',self.profit)
                #self.krw_balance = self.krw_balance + self.reward - self.fee
                self.balance += profit
                #self.clone_balance += self.position_balance - self.fee
                if self.closingPrice < self.lowest_price:
                    self.lowest_price = self.closingPrice
                self.holding_time += 1
                if (profit > 0):
                    self.long_profit += 1
                    self.sum_profit += profit
                    self.arr_profit.append(profit)
                    self.sum_holding_time_pos += self.holding_time
                    self.arr_long_close_prices_profit.append(self.closingPrice)
                    self.arr_long_close_index_profit.append(self.current_tick-1)
                else:
                    self.sum_loss += profit
                    self.sum_holding_time_neg += self.holding_time
                    self.arr_long_close_prices_loss.append(self.closingPrice)
                    self.arr_long_close_index_loss.append(self.current_tick-1)
                #self.entry_price = 0
                self.n_position += 1
                self.n_long += 1
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                if self.holding_time > self.highest_holding_time:
                    self.highest_holding_time = self.holding_time
                if self.holding_time < self.lowest_holding_time:
                    self.lowest_holding_time = self.holding_time
                self.sum_holding_time += self.holding_time
                self.arr_holding_time.append(self.holding_time)
                #self.n_no_position += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
                self.holding_time = 0
                self.delta_time = 0
                self.position = FLAT
                self.n_no_position +=1
            elif self.position == LONG:
                print('long')
                self.position = LONG
                pos = self.position
                profit = self.get_profit()
                self.action = BUY
                if self.closingPrice < self.lowest_price:
                    self.lowest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            
        elif action == HOLD:
            if self.position == SHORT:
                print('hold short position')
                self.position = SHORT
                pos = self.position
                profit = self.get_profit()
                self.action = HOLD
                if self.closingPrice > self.highest_price:
                    self.highest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            elif self.position == LONG:
                print('hold long position')
                pos = self.position
                self.position = LONG
                profit = self.get_profit()
                self.action = HOLD
                if self.closingPrice < self.lowest_price:
                    self.lowest_price = self.closingPrice
                #self.time_start = self.current_tick
                #self.time_end = self.current_tick
                self.holding_time += 1
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
            elif self.position == FLAT:
                pos = self.position
                print('no position')
                profit = 0# self.get_profit()
                self.action = HOLD
                self.delta_time += 1
                drawdown = 0
                self.reward = 0 #+ (0.1*self.delta_time)
            else:
                self.position = FLAT
                pos = self.position
                print('no position')
                profit = 0# self.get_profit()
                self.action = HOLD
                self.delta_time += 1
                drawdown = 0
                self.reward = 0
        #profit = self.get_profit()
        #profit = profit - 1
        #self.profit += profit

        if self.current_tick-1 >= self.df.shape[0]-1 or self.holding_time >= 48-1: #no position selama satu minggu (288*7)
            if self.position == SHORT:
                pos = self.position
                print('force close short at ',self.closingPrice)
                profit = self.get_profit()
                self.action = SELL
                self.balance += profit 
                if self.closingPrice > self.highest_price:
                    self.highest_price = self.closingPrice
                self.holding_time += 1
                if profit > 0:
                    self.short_profit += 1
                    self.sum_profit += profit
                    self.arr_profit.append(profit)
                    self.sum_holding_time_pos += self.holding_time
                    self.arr_short_close_prices_profit.append(self.closingPrice)
                    self.arr_short_close_index_profit.append(self.current_tick-1)
                else:
                    self.sum_loss += profit
                    self.sum_holding_time_neg += self.holding_time
                    self.arr_short_close_prices_loss.append(self.closingPrice)
                    self.arr_short_close_index_loss.append(self.current_tick-1)
                self.arr_holding_time.append(self.holding_time)
                self.n_position += 1
                self.n_short += 1
                if self.holding_time > self.highest_holding_time:
                    self.highest_holding_time = self.holding_time
                if self.holding_time < self.lowest_holding_time:
                    self.lowest_holding_time = self.holding_time
                self.sum_holding_time += self.holding_time
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
                self.holding_time = 0
                self.delta_time = 0
                self.position = FLAT
                self.n_no_position += 1
            if self.position == LONG:
                pos = self.position
                print('force close long at ',self.closingPrice)
                profit = self.get_profit()
                self.action = BUY#SELL
                self.balance += profit
                if self.closingPrice < self.lowest_price:
                    self.lowest_price = self.closingPrice
                self.holding_time += 1
                if (profit > 0):
                    self.long_profit += 1
                    self.sum_profit += profit
                    self.arr_profit.append(profit)
                    self.sum_holding_time_pos += self.holding_time
                    self.arr_long_close_prices_profit.append(self.closingPrice)
                    self.arr_long_close_index_profit.append(self.current_tick-1)
                else:
                    self.sum_loss += profit
                    self.sum_holding_time_neg += self.holding_time
                    self.arr_long_close_prices_loss.append(self.closingPrice)
                    self.arr_long_close_index_loss.append(self.current_tick-1)
                self.arr_holding_time.append(self.holding_time)
                self.n_position += 1
                self.n_long += 1
                if self.holding_time > self.highest_holding_time:
                    self.highest_holding_time = self.holding_time
                if self.holding_time < self.lowest_holding_time:
                    self.lowest_holding_time = self.holding_time
                self.sum_holding_time += self.holding_time
                drawdown = self.g_drawdown()
                self.reward = self.calReward(profit, drawdown)
                self.holding_time = 0
                self.delta_time = 0
                self.position = FLAT
                self.n_no_position +=1

        if profit > self.highest_profit:
            self.highest_profit = profit
        if profit < self.lowest_profit:
            self.lowest_profit = profit

        #self.drawdown = self.g_drawdown()
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        profit_abs = abs(profit)
       
        self.index.append(self.current_tick-1)
        self.arr_position = np.append(self.arr_position,pos)
        self.arr_reward = np.append(self.arr_reward,self.reward)

        print("current closing price", self.closingPrice)
        print("Tick: {0}/ Portfolio (USD): {1}".format(self.current_tick-1, self.balance))
        print("current delta time: ",self.delta_time)
        print("lowest delta time: ",self.lowest_delta_time)
        print("highest delta time: ",self.highest_delta_time)
        print("current hold time: ",self.holding_time)
        print("lowest hold time: ",self.lowest_holding_time)
        print("highest hold time: ",self.highest_holding_time)
        #print("force close position", self.force_close)
        print("Total position: ",self.n_position)
        print("Total no position: ",self.n_no_position)
        if self.n_no_position > 0:
            print("avg delta time :", self.sum_delta_time/self.n_no_position) #n_no_position
        if self.n_position > 0:
            print("avg holding time :", self.sum_holding_time/self.n_position)
            print("std holding time: ",np.std(self.arr_holding_time))
        n_profit = self.long_profit+self.short_profit
        if n_profit > 0:
            print("avg profit :",self.sum_profit/n_profit)
            print("avg profit arr :",np.mean(self.arr_profit))
            print("std profit: ",np.std(self.arr_profit))
            print("Sharpe ratio: ",np.mean(self.arr_profit)/np.std(self.arr_profit))
            print("avg holding time profit: ", self.sum_holding_time_pos/n_profit)
        if self.n_position-n_profit > 0:
            print("avg loss: ", self.sum_loss/(self.n_position-n_profit))
            print("avg holding time loss: ", self.sum_holding_time_neg/(self.n_position-n_profit))
        print("lowest profit: ",self.lowest_profit)
        print("highest profit: ",self.highest_profit)
        print("max drawdown: ", self.max_drawdown)
        print("Total position: ",self.n_position)
        print("Total profit position: ",n_profit)
        print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        print("Long Profit: {0}/ Short Profit: {1}".format(self.long_profit, self.short_profit))
        self.history.append((pos, self.current_tick-1, self.closingPrice, self.balance, self.reward))
       
        color = self.colors[self.current_tick-1]
        color1h = self.colors1h[self.current_tick-1]
        color4h = self.colors4h[self.current_tick-1] 
        colorD = self.colorsD[self.current_tick-1] 
        #self.current_tick_arr += 1
        cek = self.balance - drawdown
        if ((cek == 0)or(self.balance < 0)or(self.current_tick-1 >= self.df.shape[0]-1)):
            print('current tick done: ', self.current_tick)
            self.done = True
        self.current_tick += 1
        self.updateState() 
        return self.state, self.position,self.reward,drawdown,profit_abs,self.holding_time, self.delta_time, color,color1h,color4h,self.done, {'portfolio':np.array([self.balance]),#([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}

    def ssigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
        else:
            z = math.exp(x)
            sig = z / (1 + z)
        return sig

    def scaling(self, reward):
        #if reward > 0:
        #    if reward >=27: rew = 1
        #    else: rew = reward/27
        #else:
        #    if reward <= -7: rew = -1
        #    else: rew = reward/7
        if reward >= 27:
            rew = 27
        elif reward <= -7:
            rew = -7
        else: rew = reward
        #rew = -1 + 2 * (reward - (-7)) / (27 - (-7))
        return rew

    def calReward(self, profit, drawdown):
        rew = (profit/self.holding_time) - (drawdown*self.holding_time)#profit_sig - (0.5*dd_sig) - (0.5*self.holding_time)
        return rew/(self.entry_price*10000)
    
    def get_profit(self):
        if self.position == LONG:
            pro = ((self.closingPrice - self.entry_price)* 10000) #* 10 
            sigg = pro #- 1#self.open_position_balance + pro #- self.fee
        elif self.position == SHORT:
            pro = ((self.entry_price - self.closingPrice)* 10000) #* 10
            sigg = pro #- 1#self.open_position_balance + pro #- self.fee
        else:
            sigg = 0
            #self.open_position_balance = self.open_position_balance + pro
        #print('profit',pro)
        #sigg = self.ssigmoid(pro)
        return sigg    


    def g_drawdown(self):
        if self.position == LONG:
            dd = ((self.entry_price - self.lowest_price)*10000)
            #if self.entry_price == self.lowest_price:
            #    print('SAMA')
            #drawdown = self.open_position_balance + dd
        elif self.position == SHORT:
            dd = ((self.highest_price - self.entry_price)*10000)
            #if self.highest_price == self.entry_price:
            #    print('SAMA SHORT')
            #drawdown = self.open_position_balance + dd
        else: 
            dd = 0
        #dd_sig = self.ssigmoid(drawdown)
        #if(self.position == LONG):
        #    drawdown = ((self.open_position_balance-self.lowest_balance))/(self.open_position_balance)
        #elif(self.position == SHORT):
        #    drawdown = ((self.peak_balance-self.open_position_balance))/(self.open_position_balance)
        #else: drawdown = 0     
        return dd

    def init_profit(self, action, i):
        #membuat action, profit, dan loss saat ini
        #arrlabel = []
        PL = 0
        max_high = self.df.loc[i-48,'high']
        min_low = self.df.loc[i-48,'low']
        for j in range(i-48,i):
            if self.df.loc[j,'high'] > max_high:
                max_high = self.df.loc[j,'high']
            if self.df.loc[j,'low'] < min_low:
                min_low = self.df.loc[j,'low']
        cek = (self.df.loc[j,'close'] - self.df.loc[i-48,'close'])*10000
        if action == 1:
            profit = cek
            loss = -((min_low - self.df.loc[i-48,'close'])*10000)
            PL = profit + loss
        elif action == -1:
            profit = -cek
            loss = -((self.df.loc[i-48,'close'] - max_high)*10000)
            PL = profit + loss
        return PL

    def reset(self):
        self.current_tick = 48

        #position
        self.n_short = 0
        self.n_long = 0
        self.long_profit = 0
        self.short_profit = 0
        self.n_position = 0
        self.n_no_position = 0
        #self.n_profit = 0

        #clear internal variables
        self.arr_long_open_prices = []
        self.arr_long_close_prices_profit = []
        self.arr_long_close_prices_loss = []
        self.arr_short_open_prices = []
        self.arr_short_close_prices_profit = []
        self.arr_short_close_prices_loss = []
        self.arr_long_open_index = []
        self.arr_long_close_index_profit = []
        self.arr_long_close_index_loss = []
        self.arr_short_open_index = []
        self.arr_short_close_index_profit = []
        self.arr_short_close_index_loss = []
        self.index = []
        self.arr_position = self.df.loc[self.current_tick-48:self.current_tick-1,['action']].values
        print(len(self.arr_position))
        self.arr_reward = self.df.loc[self.current_tick-48:self.current_tick-1,['PL']].values
        print(len(self.arr_reward))
        self.arr_profit = []
        #self.arr_loss = []
        self.arr_holding_time = []
        #self.arr_holding_time_pos = []
        #self.arr_holding_time_neg = []
        self.arr_delta_time = []
        self.history = [] # keep buy, sell, hold action history
        self.balance = 1000 # initial balance, u can change it to whatever u like
        #self.clone_balance = self.balance
        #self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.lowest_portfolio = 0
        #self.profit = 0
        #self.peak_balance = 0.1*self.balance
        #self.lowest_balance = 0.1*self.balance
        self.drawdown = 0
        self.max_drawdown = 0
        self.entry_price = 0
        self.highest_profit = 0
        self.lowest_profit = 0
        self.pos_close_profit = 0
        self.neg_close_profit = 0
        self.lowest_price = 0
        self.highest_price = 0
        #self.pos_close_profit2 = 0
        #self.neg_close_profit2 = 0
        #self.open_position_balance = 0.1*self.balance
        #self.position_balance = 0.1*self.balance
        self.loss = 0
        self.reward = 0

        #self.time_end = self.current_tick
        #self.time_start = self.current_tick
        self.delta_time = 0
        self.highest_delta_time = 0
        self.lowest_delta_time = 0
        self.sum_delta_time = 0
        self.holding_time = 0
        self.highest_holding_time = 0
        self.lowest_holding_time = 0
        self.sum_holding_time = 0
        self.container_holding_time = 0
        self.sum_profit = 0
        self.sum_loss = 0
        self.sum_holding_time_pos = 0
        self.sum_holding_time_neg = 0
        
        self.action = HOLD
        self.position = FLAT
        self.done = False

        #for i in range(48):
        #    act = random.choice([1, 0, -1])
        #    reward = self.init_profit(act)
        #    self.arr_position.append(act)
        #    self.arr_reward.append(reward)
        #print(self.arr_position)
        #print(len(self.arr_position))
        #print(self.arr_reward)
        #print(len(self.arr_reward))
        #self.current_tick_arr = 0
        self.updateState()
        return self.state

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick-1])
        #arr = self.closingPrices[self.current_tick-48:self.current_tick]
        prev_position = np.array(self.arr_position[self.current_tick-48:self.current_tick]).flatten()  
        reward = np.array(self.arr_reward[self.current_tick-48:self.current_tick]).flatten()
        arr_colorD = self.colorsD[self.current_tick-48:self.current_tick]
        arr_color4h = self.colors4h[self.current_tick-48:self.current_tick]
        arr_color1h = self.colors1h[self.current_tick-48:self.current_tick]
        #one_hot_position = one_hot_encode(prev_position, 3)
        #profit = self.get_profit()
        #profit = self.get_portfolio()
        #drawdown = self.g_drawdown()
        ##append two
        #self.state = np.array(np.concatenate((self.df[self.current_tick], one_hot_position, [self.reward])), dtype=np.float32)
        #isi obs
        arr = self.df.loc[self.current_tick-48:self.current_tick-1,['color']].values #['color'].to_list()
        arr_reshape = np.array(arr, dtype=np.float32).reshape(48,1)
        #for i in arr_reshape:
        #    a = np.insert(a,1,prev_position, axis=0)
        #    a = np.insert(a,2,self.reward, axis=0)
        #    print(a)
        arr_reshape2 = np.insert(arr_reshape,1,prev_position, axis=1)
        arr_reshape3 = np.insert(arr_reshape2,2,reward, axis=1)
        #arr_reshape4 = np.insert(arr_reshape3,0,arr_colorD, axis=1)
        #arr_reshape5 = np.insert(arr_reshape4,1,arr_color4h, axis=1)
        #arr_reshape6 = np.insert(arr_reshape5,2,arr_color1h, axis=1)
        #print(len(arr_reshape3))
        self.state = np.array(arr_reshape3).reshape((1, arr_reshape3.shape[0], 3))#[dtime])),dtype=np.float32)
        #self.state = np.array(np.concatenate((self.df.loc[self.current_tick].values, one_hot_position, [self.reward])),dtype=np.float32)#,[profit], [-drawdown])),dtype=np.float32)
        return self.state    

    def save_array(self):
        pickle.dump(self.arr_long_close_prices_profit, open('./arr_long_close_prices_profit.pkl', 'wb'))
        pickle.dump(self.arr_long_close_prices_loss, open('./arr_long_close_prices_loss.pkl', 'wb'))
        pickle.dump(self.arr_long_close_index_profit, open('./arr_long_close_index_profit.pkl', 'wb'))
        pickle.dump(self.arr_long_close_index_loss, open('./arr_long_close_index_loss.pkl', 'wb'))
        pickle.dump(self.arr_long_open_prices, open('./arr_long_open_prices.pkl', 'wb'))
        pickle.dump(self.arr_long_open_index, open('./arr_long_open_index.pkl', 'wb'))
        pickle.dump(self.arr_short_close_prices_profit, open('./arr_short_close_prices_profit.pkl', 'wb'))
        pickle.dump(self.arr_short_close_prices_loss, open('./arr_short_close_prices_loss.pkl', 'wb'))
        pickle.dump(self.arr_short_close_index_profit, open('./arr_short_close_index_profit.pkl', 'wb'))
        pickle.dump(self.arr_short_close_index_loss, open('./arr_short_close_index_loss.pkl', 'wb'))  
        pickle.dump(self.arr_short_open_prices, open('./arr_short_open_prices.pkl', 'wb'))
        pickle.dump(self.arr_short_open_index, open('./arr_short_open_index.pkl', 'wb'))   
        pickle.dump(self.closingPrices, open('./arr_closing_prices.pkl', 'wb'))
        pickle.dump(self.index, open('./arr_closing_index.pkl', 'wb'))   