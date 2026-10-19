"""Matched-budget extension of the recorded four-frame camera experiment."""

import os

from configs.scenes.scalarsyn import *

expname = "greencube/basis_" + os.environ.get("GC_RUN", "pilot")
kernel_num = 20000
num_epochs = int(os.environ.get("GC_EPOCHS", "200"))
sliding_window_subsequent_epochs = num_epochs
sliding_window_size = int(os.environ.get("GC_FRAMES", "4"))
frame_zero_iterations = 3000
max_windows = 1
start_frame = 0
end_frame = sliding_window_size
eval_frame_limit = sliding_window_size
fix_seed = int(os.environ.get("GC_SEED", "42"))
gc_sensor_file = os.environ.get("GC_SENSOR_FILE", "")
gc_sensor_weight = float(os.environ.get("GC_SENSOR_WEIGHT", "0"))
gc_basis_file = os.environ.get("GC_BASIS_FILE", "")
OptimizationParams.update({"use_test_in_training": False, "coarse_iterations": 3000})
gaussian_ckpt_path = os.environ.get("GC_GAUSSIAN_CKPT") or None
