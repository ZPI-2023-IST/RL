from flask import Flask

from rl.logger.Logger import Logger
from rl.algorithms import algorithm_manager

app = Flask(__name__)
logger = Logger()
algorithm_manager.mount(logger)
