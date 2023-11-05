import json

from flask import request

from rl.api import logger, app, algorithm_manager
from logger.Logger import LogType

@app.route("/logs")
def get_logs():
    """
    Endpoint return logs, registered in RL module. It will be possible
    to filter thelogs with query params.
    """
    logs = logger.get_messages()
    some_filter = request.args.get("filter")
    if some_filter:
        print(f"filtering with {some_filter}")
    return {"logs": logs}


@app.route("/model")
def get_model():
    """
    Enpoint allows downloading model from RL module.
    """
    return {"model": "..."}


@app.route("/config", methods=["GET", "PUT"])
def config():
    """
    Endpoint allows for GETtin curent configuration and PUTting
    new configuration. Acceptable keys in request:
        key1 - bla bla
        key2 - bla bla
    """
    if request.method == "PUT":
        data = json.loads(request.data)
        algorithm_name = data.pop("algorithm")
        algorithm_manager.set_algorithm(algorithm_name)
        algorithm_manager.configure_algorithm(data)
        
        logger.info(
            f"Configured algorithm {algorithm_name}",
            LogType.CONFIG,
        )
        logger.info(
            f"New config: {algorithm_manager.algorithm.config.as_dict()}",
            LogType.CONFIG,
        )
        
        return json.dumps(algorithm_manager.algorithm.config.as_dict())
    else:
        data = algorithm_manager.algorithm.config.as_dict()
        data["algorithm"] = algorithm_manager.algorithm_name
        return json.dumps(data)


@app.route("/config-params")
def get_configurable_parameters():
    """
    Enpoint returns tree of configurable parameter.
    Paramters are going to depend on mode (train/test) and
    algoritm.
    """
    return {
        "train": {"Random anlgorithm": {}, "DQN": {"num_layers": "int", "lr": "float"}},
        "test": {"Random anlgorithm": {}, "DQN": {"num_layers": "int", "lr": "float"}},
    }


@app.route("/action", methods=["PUT"])
def action():
    """
    Endpoint allows for getting an action from model, based
    on game state. According to mode, there may run some training
    process. Required keys in request (types are probably going to
    change):
        state: List[float] - game state
        moves: List[List[Int]] - allowed moves
        reward: float - reward for previous action
    """
    data = json.loads(request.data)
    state = data["state"]
    moves = data["moves"]
    reward = data["reward"]

    chosen_action = algorithm_manager.algorithm.make_action(state, moves)
    return {"action": chosen_action}
