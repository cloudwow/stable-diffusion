import argparse, os, re
from optimizedSD.image_data import ImageData
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
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
import uuid

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def convert_img(image, h0, w0):

    image = image.convert("RGB")
    w, h = image.size

    
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return (2.0 * image - 1.0).to("cuda")

class Artist:
  def __init__(self, model, modelCS, modelFS,output_path:str,opt):
    self.model = model
    self.modelCS = modelCS
    self.modelFS = modelFS
    self.output_path = output_path
    self.opt = opt
    logger(vars(opt), log_csv = "logs/txt2img_logs.csv")
    if opt.precision == "autocast":
        self.precision_scope = autocast
    else:
        self.precision_scope = nullcontext
        
    self.start_code = None
    if opt.fixed_code:
        self.start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device="cuda")




  def from_prompts(self, prompts: list[str], seed, batch_size)->list[ImageData ]:
    data = batch_size * prompts
    data = list(chunk(sorted(data), batch_size))
    with torch.no_grad() :

          for prompts in tqdm(data, desc="data"):

            with self.precision_scope("cuda"):
              self.modelCS.to("cuda")
              uc = None
              if self.opt.scale != 1.0:
                  uc = self.modelCS.get_learned_conditioning(batch_size * [""])
              if isinstance(prompts, tuple):
                  prompts = list(prompts)

              subprompts, weights = split_weighted_subprompts(prompts[0])
              if len(subprompts) > 1:
                  c = torch.zeros_like(uc)
                  totalWeight = sum(weights)
                  # normalize each "sub prompt" and add it
                  for i in range(len(subprompts)):
                      weight = weights[i]
                      # if not skip_normalize:
                      weight = weight / totalWeight
                      c = torch.add(c, self.modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
              else:
                  c = self.modelCS.get_learned_conditioning(prompts)

              shape = [batch_size, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]

          
              mem = torch.cuda.memory_allocated() / 1e6
              self.modelCS.to("cpu")
              while torch.cuda.memory_allocated() / 1e6 >= mem:
                  time.sleep(1)

              samples_ddim = self.model.sample(
                  S=self.opt.ddim_steps,
                  conditioning=c,
                  seed=seed,
                  shape=shape,
                  verbose=False,
                  unconditional_guidance_scale=self.opt.scale,
                  unconditional_conditioning=uc,
                  eta=self.opt.ddim_eta,
                  x_T=self.start_code,
                  sampler = self.opt.sampler,
              )

              self.modelFS.to("cuda")

              print(samples_ddim.shape)
              print("saving images")
              result = []
              for i in range(batch_size):

                  x_samples_ddim = self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                  x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                  x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                  image = Image.fromarray(x_sample.astype(np.uint8))
                  image_data = ImageData(uuid.uuid4().hex, None, prompts[i], seed+i, image.width, image.height, 
                      int(time.time_ns()/100000),image)
                  result.append(image_data)
                  

          
              mem = torch.cuda.memory_allocated() / 1e6
              self.modelFS.to("cpu")
              while torch.cuda.memory_allocated() / 1e6 >= mem:
                  time.sleep(1)
              del samples_ddim
              print("memory_final = ", torch.cuda.memory_allocated() / 1e6)
              return result

  def from_image(self, src_image, prompts: list[str], seeds, strength):
    batch_size = len(seeds)
    data = batch_size * prompts
    data = list(chunk(sorted(data), batch_size))
    self.modelFS.to("cuda")
    init_image = convert_img(src_image, self.opt.H, self.opt.W)
    if self.opt.precision == "autocast":
      init_image = init_image.half()
    
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)

   
    init_latent = self.modelFS.get_first_stage_encoding(self.modelFS.encode_first_stage(init_image))  # move to latent space

    mem = torch.cuda.memory_allocated() / 1e6
    self.modelFS.to("cpu")
    while torch.cuda.memory_allocated() / 1e6 >= mem:
      time.sleep(1)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * self.opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    result = []
    with torch.no_grad():

        
        
          for prompts in tqdm(data, desc="data"):

              
              with self.precision_scope("cuda"):
                  self.modelCS.to("cuda")
                  uc = None
                  if self.opt.scale != 1.0:
                      uc = self.modelCS.get_learned_conditioning(batch_size * [""])
                  if isinstance(prompts, tuple):
                      prompts = list(prompts)

                  subprompts, weights = split_weighted_subprompts(prompts[0])
                  if len(subprompts) > 1:
                      c = torch.zeros_like(uc)
                      totalWeight = sum(weights)
                      # normalize each "sub prompt" and add it
                      for i in range(len(subprompts)):
                          weight = weights[i]
                          # if not skip_normalize:
                          weight = weight / totalWeight
                          c = torch.add(c, self.modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                  else:
                      c = self.modelCS.get_learned_conditioning(prompts)

              
                  mem = torch.cuda.memory_allocated() / 1e6
                  self.modelCS.to("cpu")
                  while torch.cuda.memory_allocated() / 1e6 >= mem:
                      time.sleep(1)

                  # encode (scaled latent)
                  z_enc = self.model.stochastic_encode_my(
                      init_latent,
                      torch.tensor([t_enc] * batch_size).to("cuda"),
                      seeds,
                      self.opt.ddim_eta,
                      self.opt.ddim_steps,
                  )
                  # decode it
                  samples_ddim =  self.model.sample(
                      t_enc,
                      c,
                      z_enc,
                      unconditional_guidance_scale= self.opt.scale,
                      unconditional_conditioning=uc,
                      sampler =  "ddim"
                  )

                  self.modelFS.to("cuda")
                  print("saving images")
                  for i in range(batch_size):

                      x_samples_ddim =  self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                      x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                      x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                      result.append(Image.fromarray(x_sample.astype(np.uint8)))
                      

              
                  mem = torch.cuda.memory_allocated() / 1e6
                  self.modelFS.to("cpu")
                  while torch.cuda.memory_allocated() / 1e6 >= mem:
                      time.sleep(1)

                  del samples_ddim
                  print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    print(f"result as {len(result)} images")
    return result
