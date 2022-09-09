import argparse, os, re
def get_options():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render"
  )
  parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
  parser.add_argument(
      "--skip_grid",
      action="store_true",
      help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
  )
  parser.add_argument(
      "--skip_save",
      action="store_true",
      help="do not save individual samples. For speed measurements.",
  )
  parser.add_argument(
      "--ddim_steps",
      type=int,
      default=50,
      help="number of ddim sampling steps",
  )

  parser.add_argument(
      "--fixed_code",
      action="store_true",
      help="if enabled, uses the same starting code across samples ",
  )
  parser.add_argument(
      "--ddim_eta",
      type=float,
      default=0.0,
      help="ddim eta (eta=0.0 corresponds to deterministic sampling",
  )
  parser.add_argument(
      "--n_iter",
      type=int,
      default=1,
      help="sample this often",
  )
  parser.add_argument(
      "--H",
      type=int,
      default=512,
      help="image height, in pixel space",
  )
  parser.add_argument(
      "--W",
      type=int,
      default=512,
      help="image width, in pixel space",
  )
  parser.add_argument(
      "--C",
      type=int,
      default=4,
      help="latent channels",
  )
  parser.add_argument(
      "--f",
      type=int,
      default=8,
      help="downsampling factor",
  )
  parser.add_argument(
      "--n_samples",
      type=int,
      default=5,
      help="how many samples to produce for each given prompt. A.k.a. batch size",
  )
  parser.add_argument(
      "--n_rows",
      type=int,
      default=0,
      help="rows in the grid (default: n_samples)",
  )
  parser.add_argument(
      "--scale",
      type=float,
      default=7.5,
      help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cuda",
      help="specify GPU (cuda/cuda:0/cuda:1/...)",
  )
  parser.add_argument(
      "--from-file",
      type=str,
      help="if specified, load prompts from this file",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="the seed (for reproducible sampling)",
  )
  parser.add_argument(
      "--unet_bs",
      type=int,
      default=1,
      help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
  )
  parser.add_argument(
      "--turbo",
      action="store_true",
      help="Reduces inference time on the expense of 1GB VRAM",
  )
  parser.add_argument(
      "--precision", 
      type=str,
      help="evaluate at this precision",
      choices=["full", "autocast"],
      default="autocast"
  )
  parser.add_argument(
      "--format",
      type=str,
      help="output image format",
      choices=["jpg", "png"],
      default="png",
  )
  parser.add_argument(
      "--sampler",
      type=str,
      help="sampler",
      choices=["ddim", "plms"],
      default="plms",
  )
  opt = parser.parse_args()
  return opt