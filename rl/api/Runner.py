import os
import threading
import time
import socketio
import json
import pathlib
from enum import auto, Enum

from rl.algorithms.AlgorithmManager import AlgorithmManager
from rl.logger.Logger import LogLevel, LogType, Logger
from rl.algorithms.Config import States


class GameResults:
    def __init__(self, algorithm_manager) -> None:
        self.algorithm_manager = algorithm_manager
        self.cur_game_rewards = []
        self.all_game_rewards_sum = []
        self.no_games_played = 0
        self.no_won_games = 0
        self.no_lost_games = 0
        self.no_timeouts = 0

    def store_game_results(self, reward, game_status, is_end_game):
        self.cur_game_rewards.append(reward)
        if is_end_game:
            self.no_games_played += 1
            if game_status == GameStates.WON.name:
                self.no_won_games += 1
            elif game_status == GameStates.LOST.name:
                self.no_lost_games += 1
            elif game_status == GameStates.ONGOING.name:
                self.no_timeouts += 1
            else:
                raise Exception("Unknown status")

            self.all_game_rewards_sum.append(sum(self.cur_game_rewards))
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

    def get_results(self):
        name = f"{self.algorithm_manager.algorithm_name}_{self.algorithm_manager.algorithm.config.mode}_{time.strftime('%Y/%m/%d-%H:%M:%S')}"
        return {
            "Name": name,
            "CurrentGameRewards": self.cur_game_rewards,
            "AllGameRewardsSummed": self.all_game_rewards_sum,
            "NoGamesPlayer": self.no_games_played,
            "NoWonGames": self.no_won_games,
            "NoLostGames": self.no_lost_games,
            "NoTimeouts": self.no_timeouts,
        }

    def save_results(self, path):
        with open(path, "w") as f:
            json.dump(self.get_results(), f)

    def reset(self):
        self.cur_game_rewards = []
        self.all_game_rewards_sum = []
        self.no_games_played = 0
        self.no_won_games = 0
        self.no_lost_games = 0
        self.no_timeouts = 0


class GameStates(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()


class Runner:
    def __init__(
        self,
        logger: Logger,
        algorithm_manager: AlgorithmManager,
        max_game_len=100,
        config="config.json",
        stats_dir="stats",
    ) -> None:
        self.logger = logger
        self.algorithm_manager = algorithm_manager
        self.running = False
        self.max_game_len = max_game_len
        self.start_time = 0

        self.run_process = threading.Thread(target=self.run)
        self.sio = None
        self.data = None
        self.steps = 0
        self.game_results = GameResults(algorithm_manager)

        self.current_game = []
        self.game_history = []

        self.stats_dir = pathlib.Path(stats_dir)
        os.makedirs(self.stats_dir, exist_ok=True)

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
        try:
            self.start_time = time.time()
            port = self.config["game_port"]
            self.sio.connect(
                f"http://localhost:{port}", wait_timeout=10, namespaces=["/"]
            )
            self.sio.emit("make_move", json.dumps({"move": None}), namespace="/")

            move = None
            game_step = 0
            while self.running:
                if not self.data:
                    continue

                self.steps += 1
                self.data = json.loads(self.data)
                reward = self.data["reward"]
                actions = self.data["moves_vector"]
                game_board = self.data["game_board"]
                game_status = self.data["state"]
                board_raw = self.data["board_raw"]

                if self.algorithm_manager.algorithm.config.mode == States.TEST.value:
                    self.current_game.append(board_raw)
                    if (
                        game_status != GameStates.ONGOING.name
                        or game_step >= self.max_game_len
                        or len(actions) == 0
                    ):
                        state_info = (
                            game_status
                            if game_status != GameStates.ONGOING.name
                            else "TIMEOUT"
                        )
                        game_info = {
                            "game": self.current_game,
                            "state": state_info,
                        }
                        self.game_history.append(game_info)
                        self.current_game = []

                self.data = None
                game_step += 1

                if len(actions) == 0 or game_step > self.max_game_len:
                    if game_status == GameStates.ONGOING.name:
                        self.algorithm_manager.algorithm.forward(
                            game_board, actions, reward
                        )
                    else:
                        self.algorithm_manager.algorithm.forward(None, None, reward)
                    self.game_results.store_game_results(reward, game_status, True)

                    self.sio.emit(
                        "make_move", json.dumps({"move": None}), namespace="/"
                    )
                    game_step = 0
                else:
                    move = self.algorithm_manager.algorithm.forward(
                        game_board, actions, reward
                    )
                    self.game_results.store_game_results(reward, game_status, False)

                    self.sio.emit(
                        "make_move", json.dumps({"move": move}), namespace="/"
                    )
        except Exception as e:
            self.logger.log(
                f"Error while running {self.algorithm_manager.algorithm_name}: {e}",
                LogLevel.ERROR,
                LogType.TEST
                if self.algorithm_manager.algorithm.config.mode == States.TEST.value
                else LogType.TRAIN,
            )
            self.running = False
            self.data = None
        finally:
            self.sio.disconnect()
            return

    def start(self) -> None:
        self.steps = 0
        self.run_process = threading.Thread(target=self.run)
        if self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Starting {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == States.TRAIN.value else LogType.TEST,
        )
        self.running = True
        self.game_results.reset()
        self.run_process.start()

    def stop(self) -> None:
        if not self.running:
            return
        mode = self.algorithm_manager.algorithm.config.mode
        self.logger.info(
            f"Stopping {self.algorithm_manager.algorithm_name} in {mode} mode",
            LogType.TRAIN if mode == States.TRAIN.value else LogType.TEST,
        )
        self.running = False
        self.data = None
        timestamp_str = time.strftime("%Y%m%d%H%M%S")
        self.game_results.save_results(self.stats_dir / f"{timestamp_str}.json")
        self.game_results.reset()
        self.run_process.join()
