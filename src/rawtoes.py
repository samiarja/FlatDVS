import dvs_warping_package
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import subprocess
import sys
import loris
sys.path.append("src")
from dat_files import load_dat_event

'''
Read data from gen4 .raw
'''

hot_pixel_filter     = 1
recordingname        = 'recording_2024-11-11_21-05-14_single_dot_highsensitivity'
input_directory      = '/media/samiarja/VERBATIM HD/Dataset/flatdvs/lensless_ev'
raw_files = [file for file in os.listdir(input_directory) if file.endswith('.raw')]

output_directory_dat = f'{input_directory}/dat'
output_directory_es  = f'{input_directory}/es'
out_image            = f'{input_directory}/img'

file_path = f'{output_directory_dat}/{recordingname}_cd.dat'
file_path_es = f'{output_directory_es}/{recordingname}.es'

dvs_warping_package.process_raw_to_dat(input_directory, output_directory_dat, recordingname)

width, height, events = dvs_warping_package.read_dat_file(file_path)
output_events, detected_noise = dvs_warping_package.CrossConv_HotPixelFilter(events,ratio_par=2)

nEvents = len(output_events["x"])
final_events = np.zeros((nEvents,4))
final_events = np.concatenate((output_events["t"].reshape((nEvents, 1)),
                            output_events["x"].reshape((nEvents, 1)), 
                            output_events["y"].reshape((nEvents, 1)), 
                            output_events["on"].reshape((nEvents, 1))),axis=1).reshape((nEvents,4))
finalArray = np.asarray(final_events)

loris.write_events_to_file(finalArray, file_path_es,"txyp")
cumulative_map = dvs_warping_package.accumulate((width, height), output_events, (0 / 1e6, 0 / 1e6))
warped_image = dvs_warping_package.render(cumulative_map, colormap_name="magma", gamma=lambda image: image ** (1 / 5))
warped_image.show()
warped_image.save(f"{out_image}/{recordingname}.png")
