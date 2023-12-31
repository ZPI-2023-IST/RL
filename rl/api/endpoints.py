import io
import json
import os
import pathlib
import shutil
import zipfile

from flask import request
import flask
import torch

from rl.api import logger, app, algorithm_manager, runner, name_to_time
from rl.logger.Logger import LogType


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
    response = flask.jsonify({"logs": logs})
    return response


@app.route("/run", methods=["GET", "PUT"])
def run():
    """
    Endpoint allows for starting and stopping training/testing process.
    """
    if request.method == "PUT":
        data = json.loads(request.data)
        run = data["run"]
        if run:
            algorithm_manager.algorithm.config.mode = data["mode"]
            runner.start()
        else:
            runner.stop()
        return flask.jsonify({"run": run})
    else:
        return flask.jsonify(
            {"run": runner.running, "time": runner.time, "steps": runner.steps}
        )


@app.route("/model", methods=["GET", "PUT"])
def model():
    """
    Endpoint allows downloading model from RL module.
    GET method returns model in zip format.
    PUT method imports model from zip file.
    """
    data_dir = pathlib.Path("data")
    model_dir = data_dir / "model"
    config_name = "config.json"
    params_name = "params.pt"
    zip_name = "params"
    os.makedirs(model_dir, exist_ok=True)

    if request.method == "GET":
        config = algorithm_manager.algorithm.config.as_dict()
        config["algorithm"] = algorithm_manager.algorithm_name

        with open(model_dir / config_name, "w") as f:
            json.dump(config, f)

        model = algorithm_manager.algorithm.get_model()
        if model:
            torch.save(model.state_dict(), model_dir / params_name)

        shutil.make_archive(data_dir / zip_name, "zip", model_dir)
        abs_path = pathlib.Path(data_dir / zip_name).resolve()
        response = flask.send_file(f"{abs_path}.zip", as_attachment=True)
        return response
    else:
        if runner.running:
            response = flask.jsonify(
                {"error": "Stop training/testing before importing model"}
            )
            response.status_code = 400
            return response

        data = request.data
        z = zipfile.ZipFile(io.BytesIO(data))
        z.extractall(data_dir)
        with open(data_dir / config_name, "r") as f:
            config = json.load(f)
            algorithm_manager.set_algorithm(config.pop("algorithm"))
            algorithm_manager.configure_algorithm(config)

        if os.path.isfile(data_dir / params_name):
            device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
            params = torch.load(data_dir / params_name, map_location=device)
            algorithm_manager.algorithm.set_params(params)

        logger.info(
            f"Imported model",
            LogType.CONFIG,
        )
        return flask.jsonify({"success": "success"})


@app.route("/config", methods=["GET", "PUT", "POST"])
def config():
    """
    Endpoint allows for GETtin curent configuration, POSTting
    new configuration and PUTting updated configuration.
    """
    if request.method == "PUT":
        if runner.running:
            response = flask.jsonify(
                {"error": "Stop training/testing before changing configuration"}
            )
            response.status_code = 400
            return response

        data = json.loads(request.data)
        if "algorithm" in data.keys():
            data.pop("algorithm")

        # check if any key are marked as unmodifiable
        for k, _ in data.items():
            if not algorithm_manager.algorithm.get_configurable_parameters()[
                k
            ].modifiable:
                response = flask.jsonify({"error": f"Parameter {k} is not modifiable"})
                response.status_code = 400
                return response
        
        algorithm = algorithm_manager.algorithm.__class__
        val, msg = algorithm.validate(data)
        if not val:
            response = flask.jsonify({"error": msg})
            response.status_code = 400
            return response
    
        algorithm_manager.update_config(data)
        response_data = algorithm_manager.algorithm.config.as_dict()
        response = flask.jsonify(response_data)
        return response
    elif request.method == "POST":
        if runner.running:
            response = flask.jsonify(
                {"error": "Stop training/testing before changing configuration"}
            )
            response.status_code = 400
            return response

        data = json.loads(request.data)

        algorithm_name = (
            data.pop("algorithm")
            if "algorithm" in data.keys()
            else algorithm_manager.algorithm_name
        )
        algorithm = algorithm_manager.registered_algorithms[algorithm_name]
        val, msg = algorithm.validate(data)
        if not val:
            response = flask.jsonify({"error": msg})
            response.status_code = 400
            return response
        
        algorithm_manager.set_algorithm(algorithm_name)
        algorithm_manager.configure_algorithm(data)
        response_data = algorithm_manager.algorithm.config.as_dict()
        response = flask.jsonify(response_data)
        return response
    else:
        data = algorithm_manager.algorithm.config.as_dict()

        if request.args.get("modifiable"):
            data = {
                k: v
                for k, v in data.items()
                if k != "algorithm"
                and k != "mode"
                and algorithm_manager.algorithm.get_configurable_parameters()[
                    k
                ].modifiable
            }
        data["algorithm"] = algorithm_manager.algorithm_name
        response = flask.jsonify(data)
        return response


@app.route("/config-params")
def get_configurable_parameters():
    """
    Enpoint returns tree of configurable parameter.
    Each algorithm has its own set of parameters.
    """
    params = {}

    if request.args.get("modifiable"):
        for (
            algorithm_name,
            algorithm,
        ) in algorithm_manager.registered_algorithms.items():
            params[algorithm_name] = {
                k: v
                for k, v in algorithm.get_configurable_parameters().items()
                if v.modifiable
            }
    else:
        for (
            algorithm_name,
            algorithm,
        ) in algorithm_manager.registered_algorithms.items():
            params[algorithm_name] = algorithm.get_configurable_parameters()

    response = flask.jsonify(params)
    return response


@app.route("/game-history")
def get_game_history():
    """
    Endpoint returns game history.
    """
    MAX_HISTORY = 100
    data = runner.game_history
    if len(data) > MAX_HISTORY:
        data = data[-MAX_HISTORY:]
    response = flask.jsonify({"history": data})
    return response


@app.route("/stats")
def stats():
    """
    Endpoint returns statistics about training/testing process.
    """
    MAX_STATS = 25
    data = []
    for file in runner.stats_dir.iterdir():
        with open(file) as f:
            data.append(json.load(f))
    data = sorted(data, key=lambda x: name_to_time(x["Name"]))
    if runner.running:
        current = runner.game_results.get_results()
        current["Name"] = "Current"
        data.append(current)
    if len(data) > MAX_STATS:
        data = data[-MAX_STATS:]
    return flask.jsonify(data)
