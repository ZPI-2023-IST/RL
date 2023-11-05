import json

from flask import request

from rl.api import logger, app
from rl.algorithms.Algorithm import Algorithm

@app.route("/logs")
def get_logs():
    """
    Endpoint return logs, registered in RL module. It will be possible
    to filter thelogs with query params.
    """
    logs = logger.get_messages()
    some_filter = request.args.get('filter')
    if some_filter:
        print(f"filtering with {some_filter}")
    return {"logs": logs}


@app.route("/model")
def get_model():
    """
    Enpoint allows downloading model from RL module.
    """
    return {"model": "..."}


@app.route("/config", methods=['GET', 'PUT'])
def config():
    """
    Endpoint allows for GETtin curent configuration and PUTting
    new configuration. Acceptable keys in request:
        key1 - bla bla
        key2 - bla bla
    """
    if request.method == 'PUT':
        print(request.data)
        return {}
    else:
        return {
            "mode": "train",
            "algorithm": "DQN",
            "num_layers": 4,
            "lr": 1e-3
            }


@app.route("/config-params")
def get_configurable_parameters():
    """
    Enpoint returns tree of configurable parameter.
    Paramters are going to depend on mode (train/test) and
    algoritm.
    """
    return {
        "train": {
            "Random anlgorithm": {},
            "DQN": {
                "num_layers": "int",
                "lr": "float"
            }
        },
        "test": {
            "Random anlgorithm": {},
            "DQN": {
                "num_layers": "int",
                "lr": "float"
            }
        }
    }
 
    
@app.route("/action", methods=['PUT'])
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
    
    chosen_action = [1, 0, 0]
    return {"action": chosen_action}
