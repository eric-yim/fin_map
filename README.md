# fin_map

In this repository, I use the distributional output described in https://arxiv.org/pdf/1707.06887.pdf, and apply it to financial movement for short-term value predictions. I follow general Reinforcement Learning frameworks: I have an environment to "interact" with and a replay buffer for collection and training. I can therefore use much of existing structure in Tensorflow Agents. However, here, I cut out the action taking asepct of Reinforcement Learning.
<img src="https://user-images.githubusercontent.com/48815706/77218908-724b5b00-6afe-11ea-85c8-dc91f554981c.gif" />
<h3>Why no action?</h2>
<p>
In financial markets (except in the case where the actor has loads of capital), an action does not change the environment. Hence, the price movement is assumed to unfold the way it did regardless of actions taken. Therefore, I do need to value a location differently based on action taken, only based on the expected price movement. </p>
<p>In a prior experiment, I ran the traditional DQN (with actions) on a small set of data. Note: the DQN agent learns the values across actions at each observation (i.e. the value of action 0 at observation X and the value of action 1 at observation X). The network heavily weighted the observation and could not create a meaningful value differential between actions (not good when the difference between action 0 and action 1 is buy and sell).</p>
<p>Therefore, it is better to remove unnecessary and perhaps harmful complexity.
</p>
<h3>Why Cat DQN?</h2>
<p>
The categorical dqn outputs a value distribution rather than a single value. This distribution gives the network some flexibility with matching the valuation, which is important for an inherently random environment as is a financial market. A prior train with similar inputs and the standard dqn output, yielded a network that could only output a constant number (meaning, it could not find a pattern and defaulted to some mean value).
</p>
<h2>Setup</h2>
<p>The environment loads with some known data. (Data is assumed to be stored in OHLC format.) The environment can be used with standard reinforcement learning methods. But here, I am interested only in valuing the observation, so the collection and training files are modified accordingly. The observation at each step is a manipulation of the recent X minutes (30 minutes), which is explained below. The reward is a scaled profit or loss going into the next minute.</p>
<h3>PrimeMap</h3>
<p>
The observation, here, is what I call the PrimeMap. To represent a single minute, I've created a histogram of the 60 second-to-second moves. For example, six second prices may be: [11, 13, 11, 12, 14, 11]. The this shows 5 price moves [+2, -2, +1, +2, -3]. This data can be represented as a histogram {-3: 1, -2: 1, -1: 0, 0: 0, +1: 1, +2: 2}. In this example, the net change is 0, but viewing the histogram, I can see movement was very wild. Through the PrimeMap, the network should be able to pick up information such as how rapidly the price is moving as well as generally whether the price is moving up, down, or sideways. Further, because 60 seconds is merged into 1-minute this allows minutes with similar distributions to be viewed similarly while putting less emphasis on exact order of second-to-second moves which contain a fair bit of randomness. The hope is that while seconds can be random, a minute may have meaning.
  </p>
  <p>
I show the overnight and first half of a trading day here. From the PrimeMap, the human eye can see periods of higher volatility and lower volatility. To a trader, this could show increased and decreased interest in the trading product. Perhaps the trader would have a separate strategy for each state.</p>
<img src="https://user-images.githubusercontent.com/48815706/77219367-2cdd5c80-6b03-11ea-9c8f-d966e2b6c29d.png">  
<p>The PrimeMap for the model trained contains buckets from -0.08 to +0.07 at 0.01 increments (16 buckets). Any second moves outside of -0.08 and +0.07 were placed in the -0.08 and +0.07 buckets appropriately. </p>
<p>The network input is 30 x 16 PrimeMap. The network output is a 51-atom distribution. Testing networks with mainly LSTM layers, mainly Conv1D layers, and exclusively Dense layers yielded similar results</p>
 
<h2>Results</h2>
<p>The network can easily fit to the training data. This is good news considering previous networks with a scalar output ended in "analysis paralysis." Below, I show price movement with along with the predictions reduced to a mean.
  </p>
  <img src="https://user-images.githubusercontent.com/48815706/77219814-4e404780-6b07-11ea-9eab-f06bcf14e6be.png">
<p>
 Interestingly, the trained model does seem to make better-than-random predictions on out-of-sample data. Note how after each of the 2 highest peaks, the price goes up. However, the model failed to predict the extended rally. Perhaps short-term rebounds are more common in the training data, and extended rallies are less common.
</p>
<img src="https://user-images.githubusercontent.com/48815706/77219817-4ed8de00-6b07-11ea-8877-3a94e2009266.png">
<p>I've discussed two main concepts: using the PrimeMap as a method of transforming price movement and using the distribution output to make an easier target for randomness. While the output still contains too much noise to directly trade off of, there is some potential. Future studies could include combining the inputs with other observations, such as economic data, time of day data, other financial indicators, or some trend following system. Another route could be to train on a subset of days/times (for example days that meet some criteria) and see if out-of-sample results become more exact.</p>
