import dvs_warping_package
import numpy as np
import csv

recordingname   = "recording_2024-11-08_19-36-30_two_dot_measurements"
input_directory = "/media/samiarja/VERBATIM HD/Dataset/flatdvs/lensless_ev"
es_path         = f"{input_directory}/es/{recordingname}.es"
label_path      = f"{input_directory}/es/gt/{recordingname}_labels_satellite.txt"

width, height, events = dvs_warping_package.read_es_file(es_path)


matrix = []
with open(label_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        matrix.append(row[-4:])


all_events = []
vx_values = []
vy_values = []
for vx, vy, t1, t2 in matrix:
    # Filter events between t1 and t2
    idx = np.where(np.logical_and(events["t"] >= float(t1), events["t"] <= float(t2)))
    selected_events = events[idx]
    
    # Accumulate the events and velocities
    vx_values.extend([float(vx)] * len(selected_events))
    vy_values.extend([float(vy)] * len(selected_events))

idx = np.where(np.logical_and(events["t"] >= float(matrix[0][-2]), events["t"] <= float(matrix[-1][-1])))
input_events = events[idx]


output_events, detected_noise = dvs_warping_package.CrossConv_HotPixelFilter(input_events,ratio_par=2)

out_idx = np.where(detected_noise==1)[0]
output_events = input_events[out_idx]

cumulative_map = dvs_warping_package.accumulate_cnt((width, height), output_events, (np.array(vx_values)[out_idx] / 1e6, np.array(vy_values)[out_idx] / 1e6))
warped_image = dvs_warping_package.render(cumulative_map, colormap_name="magma", gamma=lambda image: image ** (1 / 5))
warped_image.show()
warped_image.save(f"{input_directory}/img/{recordingname}_motion_compensated.png")
