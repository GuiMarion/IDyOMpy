#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:44:49 2021

@author: gui
"""

import pretty_midi
from glob import glob
from tqdm import tqdm
from shutil import copyfile
import random
random.seed(1)

china_files = glob("train_china/*.mid")
china = 0          
for file in tqdm(china_files):
    midi_data = pretty_midi.PrettyMIDI(file)
    china += len(midi_data.instruments[0].notes)
    
europe_files = glob("train_europa/*.mid")
random.shuffle(europe_files)
europe_files= europe_files[:round(len(europe_files)/2.58)]

europe = 0
for file in tqdm(europe_files):
    midi_data = pretty_midi.PrettyMIDI(file)
    europe += len(midi_data.instruments[0].notes)
    
print()
print(china)
print(europe)


for file in china_files:
    copyfile(file, file.replace("china", "china_eq"))

for file in europe_files:
    copyfile(file, file.replace("europa", "europe_eq"))
