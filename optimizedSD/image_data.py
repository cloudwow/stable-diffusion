

from dataclasses import dataclass

from PIL import Image
@dataclass
class ImageData:
  id: str
  prompt: str
  seed: int
  size_width: int
  size_height: int
  creation_time_millis: int
  source_image_id: str
  pil_image: Image
  job_id: str

  
