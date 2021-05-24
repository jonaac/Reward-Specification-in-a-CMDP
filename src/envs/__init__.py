from gym.envs.registration import register

from envs.safety.safety_env import SafeEnv, SafeEnvRM
from envs.halfcheetah.hc_env import HalfCheetahSafeEnv, HalfCheetahSafeEnvRM

__all__ = ["SafeEnvRM"]

### Half Cheetah Envs

register(
	id='Safe-Half-Cheetah-v0',
	entry_point='envs.halfcheetah.hc_env:HalfCheetahSafeEnvRM')

### Safety Gym Envs

register(
	id='Safety-v0',
	entry_point='envs.safety.safety_env:SafeEnvRM')