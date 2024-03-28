import pandas as pd
import os
from src.audio import augment_dir
from src.dataset import create_sbcm_original, create_dir_split_from, create_segmented_dataset_from_dir

augment_dir("dataset/osbh_pure/queen", "dataset/osbh_augmented/queen")
augment_dir("dataset/osbh_pure/noqueen", "dataset/osbh_augmented/noqueen")