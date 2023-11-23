import threading
import time
import socketio
import json
from enum import Enum

from rl.algorithms.AlgorithmManager import AlgorithmManager
from rl.logger.Logger import LogType, Logger
from rl.algorithms.Config import States


class State(Enum):
    ONGOING = 0
    WON = 1
    LOST = 2


class GameResults:
    def __init__(self) -> None:
        self.cur_game_rewards = []
        self.all_game_rewards_sum = {}
        self.no_games_played = 0
        self.no_won_games = 0
        self.no_lost_games = 0
        self.no_timeouts = 0

    def store_game_results(self, reward, game_status, is_end_game):
        self.cur_game_rewards.append(reward)
        if is_end_game:
            self.no_games_played += 1
            if game_status == State.WON.__str__():
                self.no_won_games += 1
            elif game_status == State.LOST.__str__():
                self.no_lost_games += 1
            elif game_status == State.ONGOING.__str__():
                self.no_timeouts += 1
            else:
                raise Exception("Unknown status")

            self.all_game_rewards_sum[self.no_games_played] = sum(self.cur_game_rewards)
            self.cur_game_rewards = []
            
    def __str__(self) -> str:
        text = "Game Results\n"
        text += f"Current game rewards: {self.cur_game_rewards}\n"
        text += f"All game rewards sum: {self.all_game_rewards_sum}\n"
        text += f"No games played: {self.no_games_played}\n"
        text += f"No won games: {self.no_won_games}\n"
        text += f"No lost games: {self.no_lost_games}\n"
        text += f"No timeouts: {self.no_timeouts}\n"
        return text

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
<<<<<<< HEAD
        self.game_results = GameResults()
=======
        
        self.current_game = []
        self.game_history = []
       
        with open(config) as f:
            self.config = json.load(f)
>>>>>>> integration

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
<<<<<<< HEAD
            game_status = self.data["state"]
=======
            board_raw = self.data["board_raw"]
            state = self.data["state"]
                        
            if self.algorithm_manager.algorithm.config.mode == "test":
                board_raw_dict = {
                    "Board": board_raw[0],
                    "FreeCells": board_raw[1],
                    "Stack": board_raw[2]
                }
                self.current_game.append(board_raw_dict)
                if state.__str__() != "ONGOING" or game_step > self.max_game_len or len(actions) == 0:
                    self.game_history.append(self.current_game)
                    self.current_game = []
>>>>>>> integration

            self.data = None
            game_step += 1
            
            if len(actions) == 0 or game_step > self.max_game_len:
                print(self.game_results)
                if game_status == State.ONGOING.__str__():
                    self.algorithm_manager.algorithm.forward(state, actions, reward)
                else:
                    self.algorithm_manager.algorithm.forward(None, None, reward)
                self.game_results.store_game_results(reward, game_status, True)

                self.sio.emit("make_move", json.dumps({"move": None}), namespace="/")
                game_step = 0
            else:
<<<<<<< HEAD
                move = self.algorithm_manager.algorithm.forward(state, actions, reward)
                self.game_results.store_game_results(reward, game_status, False)
                
=======
                move = self.algorithm_manager.algorithm.forward(board, actions, reward)
>>>>>>> integration
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
