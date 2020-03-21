from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers.tf_helper import *

from helpers.ohlc import get_ohlc
from stock_env import StockEnvRender
from cparse import config
from tf_agents.utils import nest_utils

from helpers.catdqn import scalar_returns

import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc

import sys
import numpy as np
#=========================================================================

def load_ohlc_c(date_str,contract_str,c):
    return load_ohlc(date_str=date_str,contract_str=contract_str,time_start=c.stock_env['time_start_trade'],time_end=c.stock_env['time_end'])
def load_ohlc(date_str, contract_str,time_start=120,time_end=900):
    temp_ohlc=get_ohlc(date_str,contract_str,'1 min') #TODO: raise error if holes in data
    if len(temp_ohlc) < int(time_end-time_start+1):
            raise ValueError('Unexpected Num of Bars')
    myinds = (temp_ohlc[:,0]>time_start-0.5)&(temp_ohlc[:,0]<time_end+0.5)
    return temp_ohlc[myinds]
def day_preds(date_str,contract_str,c,model):
    s = StockEnvBasic(date_str=date_str,contract_str=contract_str,**c.stock_env)
    
    obs = []
    times = range(c.stock_env['time_start_trade'],c.stock_env['time_end'])
    the_ts = s.reset()
    while not the_ts.is_last():
        obs.append(the_ts.observation)
        action = 1
        the_ts = s.step(action)
        
    obs = nest_utils.stack_nested_tensors(obs)
    preds = model(obs,is_training=False)
    return times,preds
def plotter(times,ohlc,preds):
    preds = scalar_returns(preds)
    plt.close()
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    candlestick_ohlc(ax1,ohlc,0.5,'g','r',1)
    ax2.plot(times,preds)
    ax1.set(xlim=(times[0], times[-1]+1))
    ax2.set(xlim=(times[0], times[-1]+1))
    plt.show()
def plotter_at_ind(ind,times,ohlc,preds):
    support = np.linspace(-10.0,10.0,51)
    plt.close()
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    candlestick_ohlc(ax1,ohlc[:ind+1],0.5,'g','r',1)
    ax2.bar(support,preds[ind])
    ax1.set(xlim=(times[0], times[-1]+1))
    ax2.set(xlim=(support[0], support[-1]))
    plt.show()
if __name__=="__main__":
    c = config()
    if c.date_str is None:
        date_str = c.default_env['date_str']
        contract_str = c.default_env['contract_str']
    else:
        date_str = c.date_str
        contract_str = c.contract_str
    value_net,_=load_model_checkpoint(c)
    ohlc = load_ohlc_c(date_str,contract_str,c)
    times,preds=day_preds(date_str,contract_str,c,value_net)
    if c.ind_p is not None:
        plotter_at_ind(int(c.ind_p),times,ohlc,preds)
    else:
    	plotter(times,ohlc,preds)
