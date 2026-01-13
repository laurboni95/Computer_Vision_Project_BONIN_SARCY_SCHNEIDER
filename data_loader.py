from pydantic import BaseModel
from typing import Optional
import numpy as np
from pickle import load

class MyConfig(BaseModel):
    dataset: str
    OBS_LEN: int
    OVERLAP: float
    HORIZON_FRAMES: int
    JAAD_PATH: Optional[str] = None
    split_strategy: Optional[str] = None

class Sample(BaseModel):
    dataset: str
    split: str # Partition ("train", "val" or "test")
    vid: str
    pid: str
    video_path: str
    frame_ids: list[int]
    bboxes: list[list[float]] # list of 16 list of 4 float that are [x1, y1, x2, y2] boxes -> input
    occlusion: list[int]
    is_crossing_track: int
    event_frame: int
    horizon_frames: int
    time_to_event_idx: int # May be useful for evaluation (predicting 29 images before is harder than 1 images before and less important)

class FullDataset(BaseModel):
    config: MyConfig
    samples: list[Sample]
    labels: np.ndarray

    class Config:
        arbitrary_types_allowed = True # To accept the np.ndarray type

def load_data(path_to_pkl: str) -> FullDataset:
    with open(path_to_pkl, mode="rb") as file:
        dico: dict = load(file)
    return FullDataset(**dico)