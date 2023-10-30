from fastapi import FastAPI

from . import schemas


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/logs")
async def get_logs():
    return {"logs": "..."}


@app.get("/config")
async def get_configurable_parameters():
    return {
        "train": {
            "Random anlgorithm": {},
            "DQN": {
                "num_layers": "int",
                "lr": "float"
            }
        },
        "test": {
            "Random anlgorithm": {},
            "DQN": {
                "num_layers": "int",
                "lr": "float"
            }
        }
        }


@app.get("/current_config")
async def get_current_config():
    return {
        "mode": "train",
        "algorithm": "DQN",
        "num_layers": 4,
        "lr": 1e-3
        }


@app.put("/current_config")
async def update_config(config_id: int, config: schemas.ConfigUpdate):
    pass


@app.put("/start")
async def start_process(process_id: int, process: schemas.Process):
    pass
