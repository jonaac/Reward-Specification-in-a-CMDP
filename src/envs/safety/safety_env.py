import numpy as np
import enum
import gym

from safety_gym.envs.engine import Engine
from rm.rm_env import RewardMachineEnv

class zone(enum.Enum):
	JetBlack = 0
	White	= 1
	Blue	= 2
	Green	= 3
	Red		= 4
	Yellow	= 5
	Cyan	= 6
	Magenta	= 7

	def __lt__(self, sth):
		return self.value < sth.value

	def __str__(self):
		return self.name[0]

	def __repr__(self):
		return self.name

GROUP_ZONE = 7

COLORS = {
	str(np.array([0, 0, 0, 1]) * np.array([1, 1, 1, 0.25]))	:	'j',
	str(np.array([0, 0, 1, 1]) * np.array([1, 1, 1, 0.25]))	:	'b',
	str(np.array([0, 1, 0, 1]) * np.array([1, 1, 1, 0.25]))	:	'g',
	str(np.array([0, 1, 1, 1]) * np.array([1, 1, 1, 0.25]))	:	'c',
	str(np.array([1, 0, 0, 1]) * np.array([1, 1, 1, 0.25]))	:	'r',
	str(np.array([1, 0, 1, 1]) * np.array([1, 1, 1, 0.25]))	:	'm',
	str(np.array([1, 1, 0, 1]) * np.array([1, 1, 1, 0.25]))	:	'y',
	str(np.array([1, 1, 1, 1]) * np.array([1, 1, 1, 0.25]))	:	'w'
}

class SafeEnv(Engine):
	"""
	This environment is a modification of the Safety-Gym's environment.
	There is no "goal circle" but rather a collection of zones that the
	agent has to visit or to avoid in order to finish the task.
	For now we only support the 'point' robot.
	"""
	def __init__(
			self, zones:list, 
			use_fixed_map:float,
			timeout:int, config=dict):

		self.DEFAULT.update({
			'observe_zones': False,
			'zones_num': 0,  # Number of zones in an environment
			'zones_placements': None,  # Placements list for hazards
			'zones_locations': [],  # Fixed locations to override placements
			'zones_keepout': 0.55,  # Radius of hazard keepout for placement
			'zones_size': 0.25,  # Radius of hazards
		})

		self.zones = zones
		self.zone_types = list(set(zones))
		self.zone_types.sort()
		self.use_fixed_map = use_fixed_map
		self._rgb = {
			zone.JetBlack	: [0, 0, 0, 1],
			zone.Blue		: [0, 0, 1, 1],
			zone.Green		: [0, 1, 0, 1],
			zone.Cyan		: [0, 1, 1, 1],
			zone.Red		: [1, 0, 0, 1],
			zone.Magenta	: [1, 0, 1, 1],
			zone.Yellow		: [1, 1, 0, 1],
			zone.White		: [1, 1, 1, 1]
		}
		self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])

		parent_config = {
			'robot_base': 'xmls/point.xml',
			'task': 'none',
			'lidar_num_bins': 16,
			'observe_zones': True,
			'zones_num': len(zones),
			'num_steps': timeout
		}
		parent_config.update(config)

		super().__init__(parent_config)

	@property
	def zones_pos(self):
		''' Helper to get the zones positions from layout '''
		n = self.zones_num
		return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(n)]

	def build_observation_space(self):
		''' Construct observtion space.  Happens only once at during __init__'''
		super().build_observation_space()

		if self.observe_zones:
			for zone_type in self.zone_types:
				self.obs_space_dict.update({f'zones_lidar_{zone_type}': \
												gym.spaces.Box(
													0.0, 1.0, 
													(self.lidar_num_bins,), 
													dtype=np.float32
												)
											})
 
		if self.observation_flatten:
			temp = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
			self.obs_flat_size = temp
			self.observation_space = gym.spaces.Box(-np.inf, np.inf,
													(self.obs_flat_size,), 
													dtype=np.float32)
		else:
			self.observation_space = gym.spaces.Dict(self.obs_space_dict)

	def build_placements_dict(self):
		''' Build a dict of placements.  Happens once during __init__'''
		super().build_placements_dict()

		if self.zones_num: #self.constrain_hazards:
			self.placements.update(self.placements_dict_from_object('zone'))

	def build_world_config(self):
		''' Create a world_config from our own config '''
		world_config = super().build_world_config()

		for i in range(self.zones_num):
			rbga = self.zone_rgbs[i] * [1, 1, 1, 0.25]
			name = f'zone{i}'
			geom = {'name': name,
					'size': [self.zones_size, 1e-2],#self.zones_size / 2],
					'pos': np.r_[self.layout[name], 2e-2],#self.zones_size / 2 + 1e-2],
					'rot': self.random_rot(),
					'type': 'cylinder',
					'contype': 0,
					'conaffinity': 0,
					'group': GROUP_ZONE,
					'rgba': rbga,
					'color': COLORS[str(rbga)]} #0.1]}
			world_config['geoms'][name] = geom

		return world_config

	def build_obs(self):

		obs = super().build_obs()

		if self.observe_zones:
			for zone_type in self.zone_types:
				ind = []
				for i, z in enumerate(self.zones):
					if (self.zones[i] == zone_type):
						ind.append(i)
				pos_in_type = list(np.array(self.zones_pos)[ind])

				obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(
													pos_in_type,
													GROUP_ZONE)

		return obs


	def render_lidars(self):
		offset = super().render_lidars()

		if self.render_lidar_markers:
			for zone_type in self.zone_types:
				if f'zones_lidar_{zone_type}' in self.obs_space_dict:
					ind = []
					for i, z in enumerate(self.zones):
						if (self.zones[i] == zone_type):
							ind.append(i)
					pos_in_type = list(np.array(self.zones_pos)[ind])

					self.render_lidar(
						pos_in_type, np.array([self._rgb[zone_type]]),
						offset, GROUP_ZONE)
					offset += self.render_lidar_offset_delta

		return offset

	def seed(self, seed=None):
		''' Set internal random state seeds '''
		if (self.use_fixed_map): 
			self._seed = np.random.randint(2**32) if seed is None else seed

	'''
	TO BE COMPLETED
	'''
	def get_events(self):

		events = ''
		for i in range(self.zones_num):
			name = f'zone{i}'
			color = self.world_config_dict['geoms'][name]['color']

			z_pos = self.world_config_dict['geoms'][name]['pos']
			z_dist = self.dist_xy(z_pos)

			if z_dist <= self.zones_size:
				events += color

		return events


class SafeEnvRM(RewardMachineEnv):
	def __init__(self):

		zones =	[	
			zone.JetBlack,
			zone.JetBlack,
			zone.JetBlack,
			zone.JetBlack, 
			zone.Red,
			zone.White,
			zone.Magenta
		]

		env = SafeEnv(
			zones=zones, 
			use_fixed_map=True, 
			timeout=1000,
			config={}
		)
		rm_files = ["./envs/safety/rm/rm.txt"]
		super().__init__(env, rm_files)
