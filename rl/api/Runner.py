import threading

from rl.algorithms.AlgorithmManager import AlgorithmManager
from rl.logger.Logger import LogType, Logger


class Runner:
    def __init__(self, logger: Logger, algorithm_manager: AlgorithmManager) -> None:
        self.logger = logger
        self.algorithm_manager = algorithm_manager
        self.running = False
        
        self.run_process = threading.Thread(target=self.run)
        
    def run(self) -> None:
        while self.running:
            print("Running")
    
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
        self.run_process = threading.Thread(target=self.run)
