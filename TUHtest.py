#import David_LoadTUHData_Andreas as loadTUH
import DavidLoadData as David
#import DavidPreprocessing
import os
import os.path
import mne

from prepareData import TUH_data

path = "TUH_data_sample"
TUH = TUH_data()
TUH.findEdf(path=path)
TUH.loadAllRaw()

print("Data loaded in dictionary: ", TUH.EEG_dict)

TUH_raw = TUH.EEG_raw_dict[0]

# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(TUH_raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
#ica.plot_properties(raw, picks=ica.exclude)

orig_raw = TUH_raw.copy()
TUH_raw.load_data()
ica.apply(TUH_raw)

#events = mne.find_events(TUH_raw, stim_channel='STI 014')
#print(events[:5])  # show the first 5

# show some frontal channels to clearly illustrate the artifact removal
#chs = ['MEG 0111', 'MEG 0121', 'MEG 0131', 'MEG 0211', 'MEG 0221', 'MEG 0231',
#       'MEG 0311', 'MEG 0321', 'MEG 0331', 'MEG 1511', 'MEG 1521', 'MEG 1531',
#       'EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
#       'EEG 007', 'EEG 008']
chs = ['EEG FP1-REF',
 'EEG FP2-REF',
 'EEG F3-REF',
 'EEG F4-REF',
 'EEG C3-REF',
 'EEG C4-REF',
 'EEG P3-REF',
 'EEG P4-REF',
 'EEG O1-REF',
 'EEG O2-REF',
 'EEG F7-REF',
 'EEG F8-REF',
 'EEG T3-REF',
 'EEG T4-REF',
 'EEG T5-REF',
 'EEG T6-REF',
 'EEG T1-REF',
 'EEG T2-REF',
 'EEG FZ-REF',
 'EEG CZ-REF',
 'EEG PZ-REF',
 'EEG EKG1-REF',
 'EEG LOC-REF',
 'EEG ROC-REF',
 'EEG SP1-REF',
 'EEG SP2-REF',
 'EMG-REF',
 'EEG 29-REF',
 'EEG 30-REF',
 'EEG 31-REF',
 'EEG 32-REF',
 'IBI',
 'BURSTS',
 'SUPPR']
chan_idxs = [TUH_raw.ch_names.index(ch) for ch in chs]
#orig_raw.plot(order=chan_idxs, start=12, duration=4)
orig_raw.plot(order=chan_idxs, duration=4)
TUH_raw.plot(order=chan_idxs, duration=4)