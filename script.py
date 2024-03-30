import pandas as pd
import os
from src.audio import augment_dir
from src.dataset import create_sbcm_original, create_dir_split_from, create_segmented_dataset_from_dir

create_segmented_dataset_from_dir("dataset/sbcm-depitched-5", "dataset/sbcm-depitched-5_train")