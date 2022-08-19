import os
import sys
import flwr as fl
import tensorflow as tf
import utils

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":

    #get training dataset filename from command line arguments
    filename = sys.argv[1]

    #set random seed, so all radom numbers genation process can be reproceduced between executions
    tf.random.set_seed(42)

    #create the model
    model=tf.keras.Sequential([
      tf.keras.Input(shape=(24,)),
      tf.keras.layers.Dense(48,activation="relu"),
      tf.keras.layers.Dense(4,activation="softmax")
    ])

    #compile the model
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics="accuracy"
    )

    # Load train and test datasets
    x_train, y_train = utils.load_data(filename, 39)
    x_test, y_test = utils.load_data('test.csv', 15)

    # Define Flower client
    class TransportClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            #create a learning rate callback
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch : 1e-3 *10**(epoch/20) )
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=27, callbacks=[lr_scheduler])
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=TransportClient())