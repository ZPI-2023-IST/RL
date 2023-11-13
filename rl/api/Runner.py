import threading
import time
import socketio
import json

from rl.algorithms.AlgorithmManager import AlgorithmManager
from rl.logger.Logger import LogType, Logger, LogLevel


class Runner:
    def __init__(
        self, logger: Logger, algorithm_manager: AlgorithmManager, max_game_len=100
    ) -> None:
        self.logger = logger
        self.algorithm_manager = algorithm_manager
        self.running = False
        self.max_game_len = max_game_len
        self.start_time = 0

        self.run_process = threading.Thread(target=self.run)
        self.sio = None
        self.data = {}

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
        self.sio.connect("http://localhost:5002", wait_timeout=10)
        self.sio.emit("make_move", json.dumps({"move": None}))

        control = 1
        move = None
        game_step = 0
        while self.running:
            if not self.data:
                continue

            self.data = json.loads(self.data)
            reward = self.data["reward"]
            state = self.data["game_board"]
            actions = self.data["moves_vector"]
            api_control = self.data["control"]

            self.data = None

            if api_control != control and move is not None:
                mode = self.algorithm_manager.algorithm.config.mode
                self.logger.log(
                    "Synchronization error",
                    LogLevel.ERROR,
                    LogType.TRAIN if mode == "train" else LogType.TEST,
                )
                self.running = False
                break
            else:
                control = api_control

            control += 1
            game_step += 1
            if len(actions) == 0 or game_step > self.max_game_len:
                self.sio.emit("make_move", json.dumps({"move": None}))
                game_step = 0
            else:
                move = self.algorithm_manager.algorithm.forward(state, actions, reward)
                self.sio.emit("make_move", json.dumps({"move": move}))

    def start(self) -> None:
        if self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Starting {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == "train" else LogType.TEST,
        )
        self.running = True
        self.run_process.start()

    def stop(self) -> None:
        if not self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Stopping {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == "train" else LogType.TEST,
        )
        self.running = False
        self.run_process.join()
        self.sio.disconnect()
        self.run_process = threading.Thread(target=self.run)
