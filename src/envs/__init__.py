from gym.envs.registration import register

from envs.safety.safety_env import SafeEnv, SafeEnvRM
from envs.halfcheetah.hc_env import HalfCheetahSafeEnv, HalfCheetahSafeEnvRM
from envs.water.water_env import WaterEnv, WaterEnvRM
# from envs.minecraft.minecraft_env import MineCraftEnv, MineCraftEnvRM

__all__ = ["SafeEnvRM","HalfCheetahSafeEnvRM","WaterEnvRM"]

### MineCraft Envs
'''
register(
	id='Safe-MineCraft-v0',
	entry_point='envs.minecraft.minecraft_env:MineCraftEnvRM',
	max_episode_steps=1000)
'''
### Water World Envs

register(
	id='Safe-Water-World-v0',
	entry_point='envs.water.water_env:WaterEnvRM')
	# max_episode_steps=600)

### Half Cheetah Envs

register(
	id='Safe-Half-Cheetah-v0',
	entry_point='envs.halfcheetah.hc_env:HalfCheetahSafeEnvRM')

### Safety Gym Envs

register(
	id='Safety-Gym-v0',
	entry_point='envs.safety.safety_env:SafeEnvRM')