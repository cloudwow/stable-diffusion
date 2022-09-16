

from dataclasses import dataclass
from image_data import ImageData
from enum import IntEnum
class JobType(IntEnum):
  TEXT_2_IMAGE = 1,
  IMAGE_2_IMAGE = 2
  UPSCALE = 3
  
@dataclass
class Text2ImageParams:
  seed: int

@dataclass
class Image2ImageParams:
  strength: float
  source_image_id: str
  seeds: list[int]


@dataclass
class Job:
  id: str
  job_type: JobType
  prompt: str
  batch_size: int
  creation_time_millis: int
  image2ImageParams: Image2ImageParams
  text2ImageParams: Text2ImageParams


@dataclass
class JobResult:
  job_id: str
  progress_pct: float
  results: list[ImageData]
  

    