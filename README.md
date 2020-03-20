# fin_map

In this repository, I use the distributional output described in https://arxiv.org/pdf/1707.06887.pdf, and apply it to financial movement for short-term value predictions. I follow general Reinforcement Learning frameworks: I have an environment to "interact" with and a replay buffer for collection and training. I can therefore use much of existing structure in Tensorflow Agents. However, here, I cut out the action taking asepct of Reinforcement Learning.

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
The observation, here, is what I call the PrimeMap. It i
