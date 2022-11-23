# File structure
```
|-- envs
    |-- 2Dpusher_env.py # Implementation of the pusher environment. You do not have to do anything to this file.
    |-- __init__.py 	# Necessary file for registration of the two pusher environment. You do not have to do anything to this file.
|-- utils
	|-- config.py 		# Some configuration functions for plotting & logging. You do not have to do anything to this file.
	|-- opencv_draw.py 	# Contains features for rendering the Box2D environments. You do not have to do anything to this file.
	|-- util.py 		# Some utility functions we kindly provide, which you may find useful. You do not have to do anything to this file.	
|-- agent.py 	# An agent that can be used to interact with the environment. You do not have to do anything to this file.
|-- cem.py 		# A Cross Entropy Method optimizer class to optimise an arbitrary objective. You do not have to do anything to this file.
|-- model.py 	# Create and train the ensemble dynamics model. *** You need to add code to this file. ***
|-- mpc.py 		# Model Predictive Control policy to optimize model; call CEM from this file. *** You need to add code to this file. ***
|-- randopt.py 	# A Random Search optimizer class to optimise an arbitrary objective. You do not have to do anything to this file.
|-- README.md 	# This README. You do not have to do anything to this file.
|-- run.py 		# Entry of all the experiments. You do not have to do anything to this file.
```