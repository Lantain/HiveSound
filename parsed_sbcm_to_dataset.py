import os
from src.dataset import segments_from_audio_file


DS_DIR = "./dataset/smart-bee-colony-monitor" #"./dataset/ds" #"./dataset/archive" # "./dataset/ds" # "./dataset/archive"
OUT_DIR = "./dataset/sbcm"

def parse_from_to(src_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        segments = segments_from_audio_file(f"{src_dir}/{file}", 4000, 500, None)
        print(f"Segments from {src_dir}/{file}: {len(segments)}")
        for i, seg in enumerate(segments):
            seg.set_sample_width(2)
            seg.export(f"{out_dir}/{file}_{i}.wav", format="wav", bitrate='16k', parameters=["-sample_fmt", "s16"])
            # seg.export(f"{out_dir}/{file}_{i}.wav", format="wav", bitrate='16k')

os.makedirs(OUT_DIR, exist_ok=True)
parse_from_to(f"{DS_DIR}/train/queen not present", f"{OUT_DIR}/noqueen")    
parse_from_to(f"{DS_DIR}/train/queen present or original queen", f"{OUT_DIR}/queen")    
