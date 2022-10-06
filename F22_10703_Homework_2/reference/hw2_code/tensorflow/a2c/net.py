import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, output_size, activation, layers=[32,32,16]):
        super().__init__()
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="normal", seed=None)

        self.dense1 = tf.keras.layers.Dense(layers[0], activation=tf.nn.relu, kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(layers[1], activation=tf.nn.relu, kernel_initializer=initializer)
        self.dense3 = tf.keras.layers.Dense(layers[2], activation=tf.nn.relu, kernel_initializer=initializer)

        self.output_layer = tf.keras.layers.Dense(output_size, activation=activation, kernel_initializer=initializer)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.output_layer(x)
        return x
