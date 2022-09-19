import flask
import options
from artist import Artist
import models
import io
import uuid
import time
from random import randint
from flask import jsonify
from flask import Flask, flash, request, redirect, url_for
from PIL import Image
from pytorch_lightning import seed_everything
import os
from job_manager import Jobmanager
from job import Image2ImageParams, Job, JobResult, JobType, Text2ImageParams
from optimizedSD.image_data import ImageData

image_count = 0
opt = options.get_options()
if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed = opt.seed

job_manager = Jobmanager(opt)
app = flask.Flask(__name__)
app.config["IMAGE_FOLDER"] = "./scratch"


def to_derp(image):
    img_data = io.BytesIO()
    image.save(img_data, "PNG")
    img_data.seek(0)
    return img_data


@app.route("/")
def myapp():
    return "hello"


@app.route("/txt2img/<prompt>")
@app.route("/txt2img/<prompt>/<int:batch_size>")
def txt2img(prompt, batch_size=1):
  scale = opt.scale
  ddim_steps = opt.ddim_steps
  output_width = opt.W
  output_height = opt.H
  if request.args.get("scale") != None:
    scale = float(request.args.get('scale'))
  if request.args.get("ddim_steps") != None:
    ddim_steps = int(request.args.get('ddim_steps'))
  if request.args.get("output_height") != None:
    output_height = int(request.args.get('output_height'))
  if request.args.get("output_width") != None:
    output_width = int(request.args.get('output_width'))
  job = Job(
      id=str(uuid.uuid4()),
      job_type=JobType.TEXT_2_IMAGE,
      creation_time_millis=int(time.time_ns() / 100000),
      prompt=prompt,
      batch_size=batch_size,
      scale=scale,
    ddim_steps=ddim_steps,
    output_width=output_width,
    output_height=output_height,
      text2ImageParams=Text2ImageParams(seed=None),
      image2ImageParams=None,

  )
  job_manager.enqueue(job)

  return jsonify(job)


@app.route("/job/<job_id>")
def job_results(job_id):
    results = job_manager.get_job_results(job_id)
    return jsonify(results)


def save_image(image_data):
    size = image_data.pil_image.size
    if max(size) > 2048:
        ratio = float(2048) / max(size)
        new_image_size = tuple([int(x * ratio) for x in size])
        image_data.pil_image = image_data.pil_image.resize(
            new_image_size, Image.ANTIALIAS
        )

    filename = f"{image_data.id}.png"
    filepath = os.path.join("./scratch", filename)
    image_data.pil_image.save(filepath)
    image_data.pil_image = None
    return image_data


@app.route("/img2img/<source_image_id>/<float:strength>/<int:batch_size>/<prompt>")
def img2img(source_image_id, prompt, strength=0.5, batch_size=1):
  scale = opt.scale
  ddim_steps = opt.ddim_steps
  output_width = opt.W
  output_height = opt.H
  if request.args.get("scale") != None:
    scale = float(request.args.get('scale'))
  if request.args.get("ddim_steps") != None:
    ddim_steps = int(request.args.get('ddim_steps'))
  if request.args.get("output_height") != None:
    output_height = int(request.args.get('output_height'))
  if request.args.get("output_width") != None:
    output_width = int(request.args.get('output_width'))
  job = Job(
      id=str(uuid.uuid4()),
      job_type=JobType.IMAGE_2_IMAGE,
      creation_time_millis=int(time.time_ns() / 100000),
      prompt=prompt,
      batch_size=batch_size,
      scale=scale,
      ddim_steps=ddim_steps,
      output_width=output_width,
      output_height=output_height,
      image2ImageParams=Image2ImageParams(
          seeds=None,
          strength=strength,
          source_image_id=source_image_id,
      ),
      text2ImageParams=None,
  )
  job_manager.enqueue(job)

  return jsonify(job)


def upload_image_form():
    return """
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
  </form>
  """


@app.route("/image/<image_id>", methods=["GET"])
@app.route("/image", methods=["POST", "GET"])  # for upload
def image(image_id=None):
    if image_id == None and request.method != "POST":
        return upload_image_form()
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        image = Image.open(file)
        job_id = uuid.uuid4().hex
        image_data = ImageData(
            uuid.uuid4().hex,
            "",
            0,
            image.width,
            image.height,
            int(time.time_ns() / 100000),
            None,
            image,
            job_id,
        )
        image_data = save_image(image_data)
        job = Job(
          id=job_id,
          job_type=JobType.UPLOAD,
          creation_time_millis=int(time.time_ns() / 100000),
          prompt="",
          batch_size=0,
          ddim_steps=0,
          scale=1.0,
          output_height=image_data.size_height,
          output_width=image_data.size_width,
          text2ImageParams=None,
          image2ImageParams=None,                      
          
        )
        job_manager.set_result(job.id, JobResult(job.id, 100.0, [image_data]))
        
        
        
        return jsonify(job)

    else:
        filename = f"{image_id}.png"
        filepath = os.path.join("../scratch", filename)
        return flask.send_file(filepath, mimetype="image/png")


@app.route("/thumb/<image_id>")
@app.route("/thumb/<image_id>/<int:max_pixel_dim>")
def thumb(image_id, max_pixel_dim=100):
    filename = f"{image_id}.png"
    filepath = os.path.join("./scratch", filename)
    image = Image.open(filepath)
    max_size = (max_pixel_dim, max_pixel_dim)

    image.thumbnail(max_size)
    return flask.send_file(to_derp(image), mimetype="image/png")


@app.route("/scratch/")
def list_scratch():
    path = "./scratch"
    dir_list = [filename[:-4] for filename in os.listdir(path)]
    return jsonify(dir_list)


app.run()
