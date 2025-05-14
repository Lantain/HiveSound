import pandas as pd
import os
from src.dataset import create_segmented_dataset_from_dir

create_segmented_dataset_from_dir("dataset/sbcm-depitched-5", "dataset/sbcm-depitched-5_train")