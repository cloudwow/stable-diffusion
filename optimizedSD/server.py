import  flask
import options
from artist import Artist
import models
import io
from random import randint
from flask import jsonify
from PIL import Image
from pytorch_lightning import seed_everything
import os
image_count = 0
opt = options.get_options()
if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed = opt.seed

artist = None

def initialize():
  global seed
  global artist
  if not artist:
    seed_everything(seed)
    model, modelCS, modelFS = models.load(opt)
    artist = Artist(model, modelCS,modelFS,"/home/david/output", opt)

app = flask.Flask(__name__)
def run_inference(prompt, batch_size=1):
  global artist
  global seed

  initialize()
  results = artist.from_prompts([prompt],seed, batch_size)
  seed += 1
  return results

def to_derp(image):
  img_data = io.BytesIO()
  image.save(img_data, "PNG")
  img_data.seek(0)
  return img_data

@app.route('/')
def myapp():
  return "hello"

@app.route('/txt2img/<prompt>')
@app.route('/txt2img/<prompt>/<int:batch_size>')
def txt2img(prompt, batch_size=1):
  results = run_inference(prompt, batch_size)
  
  for image_data in results:    
    filename = f"{image_data.id}.png"
    filepath = os.path.join("./scratch", filename)
    image_data.pil_image.save(filepath)
    image_data.pil_image=None
  
  return jsonify(results)

@app.route('/image/<image_id>')
def image(image_id):
  filename = f"{image_id}.png"
  filepath = os.path.join("../scratch", filename)
  return flask.send_file(filepath, mimetype='image/png')


@app.route('/thumb/<image_id>')
@app.route('/thumb/<image_id>/<int:max_pixel_dim>')
def thumb(image_id,max_pixel_dim=100):
  filename = f"{image_id}.png"
  filepath = os.path.join("./scratch", filename)
  image = Image.open(filepath)
  max_size = (max_pixel_dim, max_pixel_dim)
    
  image.thumbnail(max_size)
  return flask.send_file(to_derp(image), mimetype='image/png')


@app.route('/scratch/')
def list_scratch():
  path = "./scratch"
  dir_list =[filename[:-4] for filename in os.listdir(path)]
  return jsonify(dir_list)
  

app.run()