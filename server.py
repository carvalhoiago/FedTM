import flwr as fl
import utils
import tensorflow as tf

def fit_round(rnd):
  """Send round number to client."""
  return {"rnd": rnd}


def get_eval_fn(model):
  """Return an evaluation function for server-side evaluation."""

  # Load test data here to avoid the overhead of doing it in `evaluate` itself
  x_test, y_test = utils.load_data('test.csv', 15)

  # The `evaluate` function will be called after every round
  def evaluate(parameters):
    # Update model with the latest parameters
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, {"accuracy": accuracy}

  return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":

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

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
  
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 2})