from flask import Flask

from rl.logger.Logger import Logger
from rl.algorithms import algorithm_manager
from flask_cors import CORS, cross_origin
from rl.api.Runner import Runner

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

logger = Logger()
algorithm_manager.mount(logger)

runner = Runner(logger, algorithm_manager)
