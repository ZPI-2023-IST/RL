import unittest
import json

from rl.api.main import app
from rl.api import algorithm_manager
from rl.algorithms import DQN


class TestAPI(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_config_endpoint(self):
        client = self.client
        response = client.get("/config")
        self.assertEqual(response.status_code, 200)

        algorithm_manager.set_default_algorithm()
        algorithm_manager.configure_algorithm({"seed": 1})
        response = client.get("/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)["seed"], 1)

        response = client.put(
            "/config", data=json.dumps({"seed": 2}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)["seed"], 2)

        response = client.get("/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)["seed"], 2)

    def test_algorithm_update(self):
        config = {
            k: v[1] for k, v in DQN.get_configurable_parameters()["train"].items()
        }
        config["algorithm"] = "dqn"
        config["n_observations"] = 1
        config["n_actions"] = 1
        config["mode"] = "train"

        client = self.client
        response = client.put(
            "/config",
            data=json.dumps(config),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(algorithm_manager.algorithm_name, "dqn")
        self.assertTrue(isinstance(algorithm_manager.algorithm, DQN))

    def test_configurable_params(self):
        client = self.client
        response = client.get("/config-params")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("random" in json.loads(response.data))
        self.assertTrue("train" in json.loads(response.data)["random"])
        self.assertTrue("test" in json.loads(response.data)["random"])

    def test_logs_endpoint(self):
        client = self.client
        response = client.get("/logs")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("logs" in json.loads(response.data))
        num_logs = len(json.loads(response.data)["logs"])

        from rl.api import logger
        from rl.logger.Logger import LogType

        logger.info("test", LogType.CONFIG)
        response = client.get("/logs")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("logs" in json.loads(response.data))

        self.assertEqual(len(json.loads(response.data)["logs"]), num_logs + 1)
        self.assertEqual(json.loads(response.data)["logs"][num_logs]["message"], "test")
        self.assertEqual(json.loads(response.data)["logs"][num_logs]["type"], "CONFIG")

    def test_model_endpoint(self):
        client = self.client
        response = client.get("/model")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("model" in json.loads(response.data))
        model = json.loads(response.data)["model"]

        response = client.put(
            "/model", data=json.dumps({"model": model}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
