import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import numpy as np
import mne
from tqdm import tqdm
from pathlib import Path
import os
from cwt_spectrogram_Mads import cwt_spectrogram, spectrogram_plot
from collections import Counter

mne.set_log_level('ERROR')

################
# Define data paths
input_data_dir = Path("data/raw/")

output_data_dir = Path("data/spectrograms/")
spectrogram_folder = Path(output_data_dir / "images")
label_file_path = f'{output_data_dir}/labels.txt'

# Create the output directories if they don't exist
os.makedirs(spectrogram_folder, exist_ok=True)

if not os.path.exists(label_file_path):
    with open(label_file_path, 'w') as label_file:
        label_file

################
subject_ids = [f"SC4{i:02d}1" for i in range(0, 11)]
all_psg_files = []
all_hypnogram_files = []

for subject_id in subject_ids:
    psg_files = list(input_data_dir.rglob(f"{subject_id}/PSG/*.edf"))
    hypnogram_files = list(input_data_dir.rglob(f"{subject_id}/Hypnogram/*.edf"))
    all_psg_files.extend(psg_files)
    all_hypnogram_files.extend(hypnogram_files)

assert len(all_psg_files) == len(all_hypnogram_files), "Number of PSG files and hypnogram files are not equal"


annotation_desc_2_event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

for i in range(len(all_psg_files)):
    print(f"Processing {all_psg_files[i]}")
    subject_id = all_psg_files[i].parts[-3]
    raw_train = mne.io.read_raw_edf(all_psg_files[i], stim_channel="Event Marker", infer_types=True, preload=True) # without stim_channel="Event Marker", there will be 3 eeg channels instead of 2 + a stimulus channel.
    freq = raw_train.info['sfreq']
    assert freq == 100, f"Sampling frequency for file {all_psg_files[i]} is not 100"
    raw_train.filter(l_freq = 1, h_freq = None)
    
    annot_train = mne.read_annotations(all_hypnogram_files[i])
    annot_train = annot_train.crop(0, raw_train.times[-1]) # looks like all annotations have '?' at the end, so we can remove it
    raw_train.set_annotations(annot_train, emit_warning=True)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0)
    tmax = 30.0 - 1.0 / freq
    epochs_train = mne.Epochs(
        raw=raw_train,
        events=events_train,
        event_id=annotation_desc_2_event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,)

    #####################################  
    # Step 1: Get the labels and their counts
    labels = epochs_train.events[:, -1]
    label_counts = dict(Counter(labels))
    # Step 2: Determine how many samples to keep per class (max 250 per class)
    max_samples_per_class = 250
    indices_to_keep = []
    for label, count in label_counts.items():
        # Get indices of all samples of this class
        class_indices = np.where(labels == label)[0]
        if count > max_samples_per_class:
            # Randomly select `max_samples_per_class` indices
            selected_indices = np.random.choice(class_indices, max_samples_per_class, replace=False)
        else:
            # Keep all indices if count is less than or equal to `max_samples_per_class`
            selected_indices = class_indices
        # Add the selected indices to the list of indices to keep
        indices_to_keep.extend(selected_indices)
    # Step 3: Filter `epochs_train` to keep only the selected indices
    epochs_train = epochs_train[indices_to_keep]    

    # print number of samples per class:
    print(f"Number of samples per class after filtering: {dict(Counter(epochs_train.events[:, -1]))}")
    #####################################

    epochs_train_formatted = epochs_train.get_data().squeeze()[:,0,  :]

    label_entries = []
    for i in tqdm(range(len(epochs_train_formatted))): 
        one_epoch = epochs_train_formatted[i]

        # Create the spectrogram
        plt.rcParams['figure.figsize'] = (16, 6)
        power, times, frequencies, coif = cwt_spectrogram(one_epoch, freq, nNotes=4)
        fig = spectrogram_plot(power, times, frequencies, coif, cmap = 'jet', norm = LogNorm(), colorbar = None)

        # Define the filename and save the figure
        image_name = f'{subject_id}_{i:03d}.png'
        spectro_filename = f'{spectrogram_folder}/{image_name}'
        fig.savefig(spectro_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        fig.clf()

        # Get the label
        label = epochs_train.events[i][2]
        label_entries.append(f'{image_name}, {label}')

    # write labels to file
    with open(label_file_path, 'a') as label_file:
        label_file.write('\n'.join(label_entries) + '\n')

print("Done!")