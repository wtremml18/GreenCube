"""Reduced-budget upstream runtime pilot; test camera is reserved."""

from configs.scenes.scalarsyn import *  # noqa: F403

expname = "greencube/scalarsyn_pilot"
kernel_num = 5000
num_epochs = 30
sliding_window_subsequent_epochs = 30
sliding_window_size = 2
frame_zero_iterations = 300
start_frame = 0
end_frame = 3
max_windows = 1
OptimizationParams.update(  # noqa: F405
    dict(use_test_in_training=False, coarse_iterations=300)
)
