import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
from artist import Artist
import options
import models
# from samplers import CompVisDenoiser
logging.set_verbosity_error()




opt = options.get_options()

tic = time.time()
if opt.seed == None:
    opt.seed = randint(0, 1000000)
if not opt.prompt:
    opt.prompt = "A picture of a cat"
seed_everything(opt.seed)
model, modelCS, modelFS = models.load(opt)
artist = Artist(model, modelCS,modelFS,"/home/david/output", opt)
seed = opt.seed

result = artist.from_prompts([opt.prompt],seed, opt.n_samples)

for index,image in enumerate(result):
  image.save(
      os.path.join("/home/david/output", f"seed_{seed+index}.png")
  )
  # sub_result = image
  # for depth  in range(17):
  #   sub_results = artist.from_image(sub_result,[opt.prompt],[3553+depth], 0.65)
  #   sub_result = sub_results[0]
    
  #   sub_result.save(
  #       os.path.join("/home/david/output", f"seed_{seed+index}_{depth}.png")
  #   )



toc = time.time()

time_taken = (toc - tic) / 60.0

print(
    (
        "Samples finished in {0:.2f} minutes"
    ).format(time_taken)
)