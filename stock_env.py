from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.trajectories import time_step as ts
from tf_agents.specs.array_spec import BoundedArraySpec,ArraySpec
from tf_agents.environments.py_environment import PyEnvironment

from helpers.ohlc import load_ohlc,load_pieces,get_primemap_16

import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_finance import candlestick_ohlc



class StockEnvBasic(PyEnvironment):
    """
    Stock environment class
    Provides a step interface for testing reinforcement learning algos
        date_str: day to trade (Ex: "2018-02-22")
        contract_str: Designates which folder to look in for data (Ex: "CL 04-18")
        time_start: time in minutes to start data from (120 = 2:00am)
        time_start_trade: time in minutes to start trading day 
            (Ex: if trading starts at 720, and need prior 60 minutes of data. startTime should be 660 or earlier)
        time_end: time in minutes to end day
        commission: reward penalty for entering/exiting position
        leverage: dollar reward/price change, varies for different contracts
        pm_frames:number of frames for primemap observation
        the_std: scaling factor for rewards. could be the standard deviation of some sample of rewards
    """
    def __init__(self,date_str,contract_str, time_start=120,time_start_trade=450, time_end=720,
                 commission=-2.5,leverage=1000,pm_frames=30,the_std = 0.045):
        #May want to test variations on observation
        #Observation spec can be a tuple. 
        #if Obs spec is altered, must also alter format_obs function. likely also alter model layers based types of obs
        self._observation_spec = BoundedArraySpec(shape=(pm_frames,16), dtype=np.float32,name='pm',minimum=np.zeros((pm_frames,16)),maximum=np.ones((pm_frames,16)))
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')#-1 is go short, 1 go is long, 0 is flatten
        
        times,closes,primemap = load_pieces(date_str, contract_str,time_start=time_start,time_end=time_end)

        #self.isBull = isBull
        self.date_str = date_str
        self.contract_str = contract_str
 
        self.times = times
        self.closes = closes
        self.primemap = primemap

        self.commission=np.array(commission,dtype=np.float32)
        self.leverage = np.array(leverage,dtype=np.float32)

        self.time_start = np.array(time_start,dtype=np.int32)
        self.time_end = np.array(time_end,dtype=np.int32)
        self.time_start_trade = np.array(time_start_trade,dtype=np.int32)

        self.the_std = np.array(the_std,dtype=np.float32)
        self.pm_frames = np.array(pm_frames,dtype=np.int32)
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.num_entries=np.array(0,dtype=np.int32)
        self.position = np.array(0,dtype=np.int32)
        self.current_step = self.time_start_trade - self.time_start
        ob = self.format_obs()
        return ts.restart(ob)
    
    def _step(self,action):
        """ Method:
            action 1: long
            action -1: short
            action 0: flat
        """
        prev_close = self.closes[self.current_step]
        self.current_step+=1
        current_close = self.closes[self.current_step]
        
        action = np.array(action,dtype=np.int32)
        rew = np.array(0,dtype=np.float32)
        
        #enter trade and calc commission
        if action!=0:
            if self.position ==0:
                self.num_entries+=1
                rew+=self.commission
            elif action != self.position: #-1 to 1 means double transaction
                self.num_entries+=1
                rew+=(self.commission * 2.0)
        else:
            if self.position!=0:
                rew+=self.commission
        self.position = action
        #calc current trade going into next close
        if self.position!=0:
            multiplier = self.position.astype(np.float32)
            rew+= (current_close - prev_close)  * self.leverage * multiplier
        ob = self.format_obs()
        #if at end of closes, day has complete
        if self.current_step>=len(self.closes)-1:
            return ts.termination(ob,self.scale_rewards(rew))
        return ts.transition(ob,self.scale_rewards(rew))
    
    def format_obs(self):
        temp_pm = self.primemap[self.current_step+1-self.pm_frames:self.current_step+1]
        obs = temp_pm

        return obs    
    def scale_rewards(self,rew):
        return rew/(self.the_std*self.leverage)

    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec



class StockEnvRender(StockEnvBasic):
    """
    Visualization
        OHLC
        Buys and Sells
        PrimeMap
    """
    def __init__(self,date_str,contract_str, time_start=120,time_start_trade=450, time_end=720,
                 commission=-2.5,leverage=1000,pm_frames=30,the_std = 0.045):
        StockEnvBasic.__init__(self,date_str,contract_str, time_start=time_start,time_start_trade=time_start_trade, time_end=time_end,
                 commission=commission,leverage=leverage,pm_frames=pm_frames,the_std = the_std)
        self.ohlc=load_ohlc(date_str, contract_str,time_start=time_start,time_end=time_end)
        # For Rendering
        plt.close()
        self.ax1 = plt.subplot(2,1,1)
        self.ax2 = plt.subplot(2, 1, 2)
        self.buys = np.ones((0),dtype=np.int32)
        self.sells =np.ones((0),dtype=np.int32)
        self.flats = np.ones((0),dtype=np.int32)
        self.last_position = None
    def eval_step(self,action):
        the_ts = self._step(action)
        self.add_entries()
        return the_ts
    def add_entries(self):
        #We only want to visualize changes in position
        new_pos = False
        if self.last_position is None:
            if self.position!=0:
                new_pos=True
        else:
            if self.last_position!=self.position:
                new_pos = True
        if new_pos:
            if self.position==1:
                self.buys=np.hstack((self.buys,self.current_step-1))
                self.num_entries+=1
            elif self.position==-1:
                self.sells=np.hstack((self.sells,self.current_step-1))
                self.num_entries+=1
            else:
                self.flats=np.hstack((self.flats,self.current_step-1))
            self.last_position = self.position
    def draw_primemap(self):
        cmap=plt.get_cmap('plasma')
        X =s.times
        Y = range(-8,8)
        the_min=-1.5
        the_max=1.5

        ax = plt.gca()
        difx = X[1]-X[0]
        dify = Y[1]-Y[0]
        the_dif = the_max-the_min
        for i in range(len(X)):
            for j in range(len(Y)):
                xy = X[i],Y[j]
                temp_val = (s.primemap[i,j] - the_min)/the_dif
                rect = Rectangle(xy, difx, dify,facecolor=cmap(temp_val))#cmap(temp_val))
                self.ax2.add_patch(rect)
        self.ax2.set(ylim=(Y[0], Y[-1]+1))
    def render(self,use_show=False):
        #Draw PrimeMap
        self.draw_primemap()
        #Draw OHLC
        temp_ohlc = self.ohlc[:self.current_step+1]
        candlestick_ohlc(self.ax1,temp_ohlc,0.5,'g','r',1)

        #Draw Trades
        if self.buys.shape[0]>0:
            self.ax1.plot(temp_ohlc[self.buys,0],temp_ohlc[self.buys,-1],'bo')
        if self.sells.shape[0]>0:
            self.ax1.plot(temp_ohlc[self.sells,0],temp_ohlc[self.sells,-1],'mo')
        if self.flats.shape[0]>0:
            self.ax1.plot(temp_ohlc[self.flats,0],temp_ohlc[self.flats,-1],'ko')
        self.ax1.set(xlim=(s.times[0], s.times[-1]+1))
        self.ax2.set(xlim=(s.times[0], s.times[-1]+1))
        if use_show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001) 
if __name__=="__main__":
    s = StockEnvRender('2018-02-22','CL 04-18',time_start_trade =450,time_end=720 ,the_std = 1.0)
    the_ts = s.reset()
    rews=0
    rews = []
    while not the_ts.is_last():
        action = 0
        the_ts = s.eval_step(action)
        #print(s.times[s.current_step],the_ts.reward)
        rews.append(the_ts.reward)
    #print(np.std(rews))
    
    s.render(True)