# find all .lab files in the directory, skip first line and sparate contents by tabs
import os
import csv
import re
import math
import pandas as pd
from pydub import AudioSegment

DS_DIR =  "./ds" #"./archive" # "./ds" # "./archive"

class Section:
    time_from: float
    time_to: float
    src: str
    label: str
    status: str

    def __init__(self, time_from: float, time_to: float, src: str, label: str):
        self.src = src
        self.time_from = time_from
        self.time_to = time_to
        self.label = label
        if src.find("Missing Queen") != -1 or src.find("NO_QueenBee") != -1:
            self.status = "no_queen"
        elif src.find("QueenBee") != -1:
            self.status = "queen"
        elif src.find("Swarming") != -1:
            self.status = "swarming"
        elif src.find("Active") != -1:
            self.status = "active"
        else:
            print(f"Unknown Status in file {src}")
        
    def show(self):
        print(f"Src: {self.src}, Label {self.label}, Status {self.status}, From: {self.time_from}, To: {self.time_to}")

    def slice(self):
        basename = os.path.basename(self.src).replace(".lab", "")
        sound_file = f"{DS_DIR}/{basename}.wav"

        out_file = f"./out/{self.label}/{self.status}_{basename}.wav"

        if (not os.path.exists(sound_file)):
            sound_file = f"{DS_DIR}/{basename}.mp3" 
            if (not os.path.exists(sound_file)):
                return
        audio = AudioSegment.from_file(sound_file)
        try: 
            a = audio[self.time_from * 1000:min(self.time_to * 1000, len(audio))]
            # split audio slice to 4 seconds chunks
            if len(a) > 4000:
                for i in range(math.ceil(len(a) / 4000)):
                    chunk = a[i * 4000:(i + 1) * 4000]
                    if not os.path.exists(f"./queenless/{self.status}"):
                        os.makedirs(f"./queenless/{self.status}")
                    if (len(chunk) == 4000):
                        mono_audios = chunk.split_to_mono()
                        mono_left = mono_audios[0]
                        mono_left.export(f"./queenless/{self.status}/{basename}_{i}.wav", format="wav")
                return
            
            a.export(out_file, format="wav")
        except Exception as e:
            print(f"Failed to export {self.time_from} to {self.time_to} from file {self.src} length {len(audio)}")
        # print(f"Sliced {self.src} from {self.time_from} to {self.time_to}: {out_file}")

sections: list[Section] = list()

# get all .lab files in the directory
def get_lab_files(dir: str):
    lab_files = []
    for file in os.listdir(dir):
        if not file.startswith('.') and file.endswith(".lab"):
            lab_files.append(file)
    return lab_files

# read .lab file and return the contents
def read_lab_file(file: str):
    with open(file, "r") as f:
        contents = f.readlines()
    return contents

# remove first line and split contents by tabs
def process_lab_file(contents: list):
    contents = contents[1:]
    contents = [re.split(r'\t+', line) for line in contents]
    return contents

files = get_lab_files(DS_DIR)

# read and process all .lab files
for file in files:
    contents = read_lab_file(f"{DS_DIR}/" + file)
    processed_contents = process_lab_file(contents)
    for line in processed_contents:
        if len(line) == 3:
            sections.append(
                Section(
                    float(line[0]),
                    float(line[1]),
                    f"{DS_DIR}/{file}",
                    line[2].strip()
                )
            )

try:
    os.makedirs("./out")
    os.makedirs("./queenless")
    os.makedirs("./out/bee")
    os.makedirs("./out/nobee")

except:
    pass

for sec in sections:
    sec.show()
    sec.slice()
