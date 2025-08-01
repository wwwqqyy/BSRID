import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN


def seed_np_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()

    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrain = CN()
    conf.JointTrain.SampleMaxSteps = 0
    conf.JointTrain.BufferMaxLength = 0
    conf.JointTrain.BufferWarmUp = 0
    conf.JointTrain.NumEnvs = 0
    conf.JointTrain.RID = 0
    conf.JointTrain.ContextLength = 0
    conf.JointTrain.BatchSize = 0
    conf.JointTrain.BatchLength = 0
    conf.JointTrain.ImagineBatchSize = 0
    conf.JointTrain.ImagineContextLength = 0
    conf.JointTrain.ImagineBatchLength = 0
    conf.JointTrain.TrainDynamicsEverySteps = 0
    conf.JointTrain.TrainAgentEverySteps = 0
    conf.JointTrain.Temperature = 0
    conf.JointTrain.SaveEverySteps = 0
    conf.JointTrain.BalancedSample = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
