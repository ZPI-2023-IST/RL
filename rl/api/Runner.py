import threading
import time
import socketio
import json

from rl.algorithms.AlgorithmManager import AlgorithmManager
from rl.logger.Logger import LogType, Logger
from rl.algorithms.Config import States


class Runner:
    def __init__(
        self, logger: Logger, algorithm_manager: AlgorithmManager, max_game_len=100, config="config.json"
    ) -> None:
        self.logger = logger
        self.algorithm_manager = algorithm_manager
        self.running = False
        self.max_game_len = max_game_len
        self.start_time = 0

        self.run_process = threading.Thread(target=self.run)
        self.sio = None
        self.data = None
        
        self.current_game = []
        self.game_history = []
       
        with open(config) as f:
            self.config = json.load(f)

        self._mount_socketio()

    def _mount_socketio(self) -> None:
        self.sio = socketio.Client()
        
        @self.sio.event
        def connect():
            mode = self.algorithm_manager.algorithm.config.mode
            self.logger.info(
                "I'm connected!", LogType.TRAIN if mode == "train" else LogType.TEST
            )

        @self.sio.event
        def disconnect():
            mode = self.algorithm_manager.algorithm.config.mode
            self.logger.info(
                "I'm disconnected!", LogType.TRAIN if mode == "train" else LogType.TEST
            )

        @self.sio.event
        def get_response(message):
            self.data = message

    @property
    def time(self) -> float:
        return time.time() - self.start_time if self.running else 0

    def run(self) -> None:
        self.start_time = time.time()
        port = self.config["game_port"]
        self.sio.connect(f"http://localhost:{port}", wait_timeout=10, namespaces=["/"])
        self.sio.emit("make_move", json.dumps({"move": None}), namespace="/")

        move = None
        game_step = 0
        while self.running:
            if not self.data:
                continue

            self.data = json.loads(self.data)
            reward = self.data["reward"]
            board = self.data["game_board"]
            actions = self.data["moves_vector"]
            board_raw = self.data["board_raw"]
            state = self.data["state"]
                        
            if self.algorithm_manager.algorithm.config.mode == "test":
                board_raw_dict = {
                    "Board": board_raw[0],
                    "FreeCells": board_raw[1],
                    "Stack": board_raw[2]
                }
                self.current_game.append(board_raw_dict)
                if state.__str__() != "ONGOING" or game_step >= self.max_game_len or len(actions) == 0:
                    self.game_history.append(self.current_game)
                    self.current_game = []

            self.data = None
            game_step += 1
            
            if len(actions) == 0 or game_step > self.max_game_len:
                self.sio.emit("make_move", json.dumps({"move": None}), namespace="/")
                game_step = 0
            else:
                move = self.algorithm_manager.algorithm.forward(board, actions, reward)
                self.sio.emit("make_move", json.dumps({"move": move}), namespace="/")

        self.sio.disconnect()

    def start(self) -> None:
        if self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Starting {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == States.TRAIN else LogType.TEST,
        )
        self.running = True
        self.run_process.start()

    def stop(self) -> None:
        if not self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Stopping {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == States.TRAIN else LogType.TEST,
        )
        self.running = False
        self.data = None
        self.run_process.join()

        self.run_process = threading.Thread(target=self.run)
