import math
import os
import shutil


def delete_existing(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def dann_scheduler(x, ramp_gamma=10):
    den = 1.0 + math.exp(-ramp_gamma * x)
    lamb = 2.0 / den - 1.0
    return min(lamb, 1.0)
