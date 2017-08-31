import matplotlib

matplotlib.use('Agg')

import funfolding as ff
from funfolding import discretization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import corner
import evaluate_unfolding
import unfolding

import multiprocessing

from fact.io import read_h5py
from fact.analysis import split_on_off_source_independent
from fact.analysis import split_on_off_source_dependent