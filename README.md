# fin_map

In this repository, I use the distributional output described in https://arxiv.org/pdf/1707.06887.pdf, and apply it to financial movement for short-term value predictions. I follow general Reinforcement Learning frameworks: I have an environment to "interact" with and a replay buffer for collection and training. I can therefore use much of existing structure in Tensorflow Agents. However, here, I cut out the action taking asepct of Reinforcement Learning.

<h1><b>Why no action?</b></h1>
<p>
In financial markets (except in the case where the actor has loads of capital), an action does not change the environment. Hence, the price movement is assumed to unfold the way it did regardless of actions taken. Therefore, I do need to value a location differently based on action taken, only based on the expected price movement. Indeed, on when training an agent based on a traditional DQN approach, the network learns roughly the valuation as seen here, while the difference in value based on action does not seem to provide meaningful insight. It is better to remove unnecessary complexity.
</p>

