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

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())



def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load(opt) :
  ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
  seed_everything(opt.seed)

  # Logging
  logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

  sd = load_model_from_config(f"{ckpt}")
  li, lo = [], []
  for key, value in sd.items():
      sp = key.split(".")
      if (sp[0]) == "model":
          if "input_blocks" in sp:
              li.append(key)
          elif "middle_block" in sp:
              li.append(key)
          elif "time_embed" in sp:
              li.append(key)
          else:
              lo.append(key)
  for key in li:
      sd["model1." + key[6:]] = sd.pop(key)
  for key in lo:
      sd["model2." + key[6:]] = sd.pop(key)
  config = "optimizedSD/v1-inference.yaml"
  
  config = OmegaConf.load(f"{config}")

  model = instantiate_from_config(config.modelUNet)
  _, _ = model.load_state_dict(sd, strict=False)
  model.eval()
  model.unet_bs = opt.unet_bs
  model.cdevice = opt.device
  model.turbo = opt.turbo

  modelCS = instantiate_from_config(config.modelCondStage)
  _, _ = modelCS.load_state_dict(sd, strict=False)
  modelCS.eval()
  modelCS.cond_stage_model.device = opt.device

  modelFS = instantiate_from_config(config.modelFirstStage)
  _, _ = modelFS.load_state_dict(sd, strict=False)
  modelFS.eval()
  del sd

  if opt.precision == "autocast":
    model.half()
    modelCS.half()
    modelFS.half()

  return model, modelCS, modelFS
