# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:31:32 2023

@author: wenao
"""

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import codoc
import numpy as np


"""Experiment Name Choices

Featured datasets:
- "uk_mammo_single", "uk_mammo_arbitration" for UK Mammography Dataset with
      either single reader or arbitration respectively.
- "us_mammo" for US Mammography Dataset.

For own datasets, pass the experiment name you have assigned once you store the
data under `data/[experiment_name]`.
"""
experiment_name = "internal"

tune_path = './data/internal/tune.csv'
val_path = './data/internal/val.csv'
test_path = './data/internal/test.csv'


all_hp = codoc.load_hyperparameters()
df_tune, df_val, df_test = codoc.load_data(experiment_name,tune_path,val_path,test_path)


# Target metric: "sens" for sensitivity, "spec" for specificity.
target_metric = "spec"

try:
  hp = all_hp["uk_mammo_single_" + target_metric]
except KeyError:
  hp = {}
  print(
      "The experiment + target_metric combination"
      f" {experiment_name + '_' + target_metric} you requested cannot be found!"
  )
  
  
model = codoc.estimate_model(
    df_tune=df_tune,  # The tune dataset split.
    df_val=df_val,  # The validation dataset split.
    num_bins=hp["num_bins"],  # $T$ from the main paper, number of bins.
    tau=hp[
        "tau"
    ],  # Index for operating point for predictive AI: $\tau = \theta * T$.
    lam=hp[
        "lam"
    ],  # $\lambda$ from the main paper, sens-spec trade off hyperparameter.
    pseudocounts=hp["pseudocounts"],  # $\kappa$ from the main paper.
    smoothing_bandwidth=hp[
        "smoothing_bandwidth"
    ],  # $\sigma$ from the main paper.
)

test_sens, test_spec = codoc.evaluate_codoc_model(
    df_test, model["operating_point"], model["thresholds"]
)
print("CoDoC sensitivity on test set:", test_sens)
print("CoDoC specificity on test set:", test_spec)
codoc.plot_advantage_z(model["phis"], model["params"]["tau"], model["a_z"])

print("Number of cases in test set is {}".format(len(df_test)))
print("Number of positive cases in test set is {}".format(df_test.y_true.sum()))

test_reader_sens, test_reader_spec = codoc.evaluate_baseline_reader(df_test)
print("Clinical workflow sensitivity on test set: {}".format(test_reader_sens))
print("Clinical workflow specificity on test set: {}".format(test_reader_spec))

test_ai_sens, test_ai_spec = codoc.evaluate_baseline_model(df_test)
print("Predictive AI sensitivity on test set: {}".format(test_ai_sens))
print("Predictive AI specificity on test set: {}".format(test_ai_spec))