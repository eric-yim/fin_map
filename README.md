# fin_map

In this repository, I use the distributional output described in https://arxiv.org/pdf/1707.06887.pdf, and apply it to financial movement for short-term value predictions. I follow general Reinforcement Learning frameworks: I have an environment to "interact" with and a replay buffer for collection and training. I can therefore use much of existing structure in Tensorflow Agents. However, here, I cut out the action taking asepct of Reinforcement Learning.
<img src="https://user-images.githubusercontent.com/48815706/77218908-724b5b00-6afe-11ea-85c8-dc91f554981c.gif" />
<h2>Why no action?</h2>
<p>
In financial markets (except in the case where the actor has loads of capital), an action does not change the environment. Hence, the price movement is assumed to unfold the way it did regardless of actions taken. Therefore, I do need to value a location differently based on action taken, only based on the expected price movement. Indeed, when training an agent based on a traditional DQN approach, the network learns roughly the valuation as seen here, while the difference in value based on action does not seem to provide meaningful insight. It is better to remove unnecessary complexity.
</p>
<h2>Why Cat DQN</h2>
<p>
The categorical dqn outputs a distribution rather than a single value. This distribution gives the network some flexibility with matching the valuation, which is important for an inherently random environment like in financial markets. A prior train with similar inputs and the standard dqn output, yielded a network that would output a constant number (meaning, it could not find a pattern and defaulted to some mean value).
</p>
<h2>Setup</h2>
<p>The environment loads with some known data. (Data is assumed to be stored in OHLC format.) The environment can be used with standard reinforcement learning methods. But here, I am interested only in valuation. The observation is a manipulation of the recent X minutes (30 minutes). The reward is a scaled profit or loss going into the next minute.</p>
<h3>PrimeMap</h3>
<p>
The observation, here, is what I call the PrimeMap. To represent a single minute, I've cut created a histogram of the 60 second-to-second moves. For example, the six second prices may be: [11, 13, 11, 12, 14, 11]. The this shows 5 price moves [+2, -2, +1, +2, -3]. This data can be represented as a histogram {-3: 1, -2: 1, -1: 0, 0: 0, +1: 1, +2: 2}. In this example, the net change is 0, but viewing the histogram, I can see movement was very wild. Through the PrimeMap, the network should be able to pick up information such as how rapidly the price is moving as well as generally whether the price is moving up, down, or sideways. Further, because 60 seconds is merged into 1-minute this allows minutes with similar distributions to be viewed similarly while putting less emphasis on exact order of second-to-second moves which contain a fair bit of randomness. The hope is that while seconds can be random, a minute may have meaning.
  </p>
  <p>
I show the overnight and first half of a trading day here. From the PrimeMap, the human eye can see periods of higher volatility and lower volatility. </p>
<img src="https://user-images.githubusercontent.com/48815706/77219367-2cdd5c80-6b03-11ea-9c8f-d966e2b6c29d.png">  
 
