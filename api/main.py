import json

from flask import Flask, request

app = Flask(__name__)


@app.route("/logs")
def get_logs():
    some_filter = request.args.get('filter')
    if some_filter:
        print(f"filtering with {some_filter}")
    return {"logs": "..."}


@app.route("/model")
def get_model():
    return {"model": "..."}


@app.route("/config", methods=['GET', 'PUT'])
def config():
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
    data = json.loads(request.data)
    state = data["state"]
    moves = data["moves"]
    reward = data["reward"]
    
    chosen_action = [1, 0, 0]
    return {"action": chosen_action}
