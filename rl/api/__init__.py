from flask import Flask

from rl.logger.Logger import Logger

app = Flask(__name__)
logger = Logger()
