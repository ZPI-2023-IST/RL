import io
import json
import os
import pathlib
import shutil
import zipfile

from flask import request
import flask

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
        return flask.jsonify({"run": runner.running})


@app.route("/model", methods=["GET", "PUT"])
def model():
    """
    Enpoint allows downloading model from RL module.
    """
    data_dir = pathlib.Path("data")
    model_dir = data_dir / "model"
    config_name = "config.json"
    zip_name = "params"
    os.makedirs(model_dir, exist_ok=True)

    if request.method == "GET":
        config = algorithm_manager.algorithm.config.as_dict()
        config["algorithm"] = algorithm_manager.algorithm_name

        with open(model_dir / config_name, "w") as f:
            json.dump(config, f)

        shutil.make_archive(data_dir / zip_name, "zip", model_dir)
        response = flask.send_file(
            pathlib.Path(f"../{data_dir / zip_name}.zip"), as_attachment=True
        )
        return response
    else:
        data = request.data
        z = zipfile.ZipFile(io.BytesIO(data))
        z.extractall(data_dir)
        with open(data_dir / config_name, "r") as f:
            config = json.load(f)
            algorithm_manager.set_algorithm(config.pop("algorithm"))
            algorithm_manager.configure_algorithm(config)
            logger.info(
                f"Imported model",
                LogType.CONFIG,
            )
        return flask.jsonify({"success": "success"})


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
    for algorithm_name, algorithm in algorithm_manager.registered_algorithms.items():
        params[algorithm_name] = algorithm.get_configurable_parameters()
    response = flask.jsonify(params)
    return response
