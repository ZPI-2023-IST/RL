import io
import json
import os
import pathlib
import shutil
import zipfile

from flask import request
import flask
import torch

from rl.api import logger, app, algorithm_manager, runner
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
        print(runner.time)
        return flask.jsonify({"run": runner.running, "time": runner.time})


@app.route("/model", methods=["GET", "PUT"])
def model():
    """
    Enpoint allows downloading model from RL module.
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
        
        print(config)

        with open(model_dir / config_name, "w") as f:
            json.dump(config, f)

        model = algorithm_manager.algorithm.get_model()
        if model:
            torch.save(model.state_dict(), model_dir / params_name)

        shutil.make_archive(data_dir / zip_name, "zip", model_dir)
        abs_path = pathlib.Path(data_dir / zip_name).resolve()
        response = flask.send_file(
            f"{abs_path}.zip", as_attachment=True
        )
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
            params = torch.load(data_dir / params_name)
            algorithm_manager.algorithm.set_params(params)

        logger.info(
            f"Imported model",
            LogType.CONFIG,
        )
        return flask.jsonify({"success": "success"})


@app.route("/config", methods=["GET", "PUT", "POST"])
def config():
    """
    Endpoint allows for GETtin curent configuration and PUTting
    new configuration. Acceptable keys in request:
        key1 - bla bla
        key2 - bla bla
    """
    if request.method == "PUT":
        if runner.running:
            response = flask.jsonify(
                {"error": "Stop training/testing before changing configuration"}
            )
            response.status_code = 400
            return response

        data = json.loads(request.data)
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
        algorithm_manager.set_algorithm(algorithm_name)
        algorithm_manager.configure_algorithm(data)
        response_data = algorithm_manager.algorithm.config.as_dict()
        response = flask.jsonify(response_data)
        return response
    else:
        data = algorithm_manager.algorithm.config.as_dict()
        if "mode" in data.keys():
            data.pop("mode")
        
        if request.args.get("modifiable"):
            data = {
                k: v
                for k, v in data.items()
                if k != "algorithm" and algorithm_manager.algorithm.get_configurable_parameters()[k].modifiable
            }
        data["algorithm"] = algorithm_manager.algorithm_name
        response = flask.jsonify(data)
        return response


@app.route("/config-params")
def get_configurable_parameters():
    """
    Enpoint returns tree of configurable parameter.
    Paramters are going to depend on mode (train/test) and
    algoritm.
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
