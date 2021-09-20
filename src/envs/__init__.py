from gym.envs.registration import register

#  MineCraft Envs
'''
register(
	id='Safe-MineCraft-v0',
	entry_point='envs.minecraft.minecraft_env:MineCraftEnvRM',
	max_episode_steps=1000)
'''

#  Water World Envs

register(
	id='Safe-Water-World-NoMachine-v0',
	entry_point='envs.water.water_env:WaterEnvNoM')

register(
	id='Safe-Water-World-RewardMachine-v0',
	entry_point='envs.water.water_env:WaterEnvRM')

register(
	id='Safe-Water-World-SafetyMachine-v0',
	entry_point='envs.water.water_env:WaterEnvSM')

''' to be able to run code on Google Colab
#  Half Cheetah Envs

register(
	id='Safe-Half-Cheetah-v0',
	entry_point='envs.halfcheetah.hc_env:HalfCheetahSafeEnvRM')

### Safety Gym Envs

register(
	id='Safety-Gym-v0',
	entry_point='envs.safety.safety_env:SafeEnvRM')
'''