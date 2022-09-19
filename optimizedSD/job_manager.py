
from artist import Artist
from job import Job, JobResult, JobType
import multiprocessing as mp
from random import randint
import models
import os
from PIL import Image
from pytorch_lightning import seed_everything
class Jobmanager:
  
  _work_queue: mp.Queue
  _results_queue: mp.Queue
  _worker: mp.Process
  _result_by_id: dict[str, JobResult]

  def __init__(self, opt):
    self._opt = opt
    self._work_queue = mp.Queue()
    self._results_queue = mp.Queue()
    self._result_by_id = {}
    self._worker = mp.Process(target=_worker_loop,
        args=(self._work_queue, self._results_queue,self._opt))

    self._worker.start()
  def set_result(self, job_id, job_result):
    self._result_by_id[job_id] = job_result
  def enqueue(self, job: Job):
    self._result_by_id[job.id] = JobResult(job.id, 0.0, []) 
    self._work_queue.put(job)

  def get_job_results(self, job_id):
    while True:
      try:
          job_result = self._results_queue.get_nowait()
          self._result_by_id[job_result.job_id] = job_result
      except:
          break
    return self._result_by_id[job_id]

def _worker_loop( work_queue: mp.Queue, results_queue: mp.Queue, opt):
  seed = randint(0, 1000000)
  seed_everything(seed)
  model, modelCS, modelFS = models.load(opt)
  artist = Artist(model, modelCS,modelFS,"/home/david/output", opt)

  while True:
    job = work_queue.get()
    if job is None:
      break
    _do_job(artist, job, results_queue)

def _do_job( artist:Artist, job: Job, results_queue: mp.Queue):
  if job.job_type == JobType.TEXT_2_IMAGE:
    results = _do_text_2_image(artist, job)
  elif job.job_type == JobType.IMAGE_2_IMAGE:
    results = _do_image_2_image(artist,job)
  else:
    raise Exception(f"Unknown job type {job.job_type}")
  results_queue.put(results)

def _do_text_2_image(artist:Artist, job: Job):
  if job.text2ImageParams.seed is  None:
    job.text2ImageParams.seed= randint(0, 1000000)
  image_results = artist.from_prompts(job, [job.prompt],job.text2ImageParams.seed, job.batch_size)
  for image_data in image_results:   
    image_data.job_id =  job.id
    filename = f"{image_data.id}.png"
    filepath = os.path.join("./scratch", filename)
    image_data.pil_image.save(filepath)
    image_data.pil_image=None
  return JobResult(job.id, 100.0, image_results)


def _do_image_2_image(artist:Artist, job: Job):
  seeds = job.image2ImageParams.seeds
  if seeds is None:
    seeds = [randint(0, 1000000) for i in range(job.batch_size)]
  
  filename = f"{job.image2ImageParams.source_image_id}.png"
  filepath = os.path.join("./scratch", filename)
  source_image = Image.open(filepath)
  image_results = artist.from_image (job, source_image, job.prompt,seeds, job.image2ImageParams.strength)
  for image_data in image_results:   
    image_data.job_id =  job.id
    image_data.source_image_id =  job.image2ImageParams.source_image_id
    filename = f"{image_data.id}.png"
    filepath = os.path.join("./scratch", filename)
    image_data.pil_image.save(filepath)
    image_data.pil_image=None
  
  return JobResult(job.id, 100.0, image_results)