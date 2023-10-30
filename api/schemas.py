from pydantic import BaseModel, Field, validator

class ConfigBase(BaseModel):
    algorithm: str
    mode: str
    lr: float
    num_layers: int


class ConfigUpdate(ConfigBase):
    pass
    
    
class Config(ConfigBase):
    pass


class Process(BaseModel):
    run: bool
    