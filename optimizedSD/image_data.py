

from dataclasses import dataclass

from PIL import Image
@dataclass
class ImageData:
  id: str
  parent_image_id: str
  prompt: str
  seed: int
  size_width: int
  size_height: int
  creation_time_millis: int
  pil_image: Image

  
