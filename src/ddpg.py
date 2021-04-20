import tensorflow as tf 


class CriticNetwork():

	self.model = None

	def __init__(self, states, actions, action_max):
		
		last_init = tf.random_normal_initializer(stddev=0.0005)

		inputs = tf.keras.layers.Input(shape=(states,), dtype=tf.float32)
		out = tf.keras.layers.Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(inputs)
		out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(out)
		outputs = tf.keras.layers.Dense(actions, activation="tanh", kernel_initializer=last_init)(out) * action_max

		model = tf.keras.Model(inputs, outputs)
		self.model  = model

class ActorNetwork():

	self.model = None

	def __init__(self, states, actions, action_max):
		
		last_init = tf.random_normal_initializer(stddev=0.00005)

		# State input
		state_input = tf.keras.layers.Input(shape=(states), dtype=tf.float32)
		state_out = tf.keras.layers.Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(state_input)
		state_out = tf.keras.layers.BatchNormalization()(state_out)
		state_out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(state_out)

		# Action input
		action_input = tf.keras.layers.Input(shape=(actions), dtype=tf.float32)
		action_out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(action_input/action_max)

		# Concatenate Layers
		added = tf.keras.layers.Add()([state_out, action_out])

		added = tf.keras.layers.BatchNormalization()(added)
		outs = tf.keras.layers.Dense(150, activation=tf.nn.leaky_relu, kernel_initializer=KERNEL_INITIALIZER)(added)
		outs = tf.keras.layers.BatchNormalization()(outs)
		# outs = tf.keras.layers.Dropout(DROUPUT_N)(outs)
		outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)

		# Outputs single value for give state-action
		model = tf.keras.Model([state_input, action_input], outputs)
		self.model  = model

