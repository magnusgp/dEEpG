from loadData import TUH_data
import preprocessing
from collections import defaultdict
from eegProcess import TUH_rename_ch, readRawEdf, pipeline



# Load sample EDF file
TUH = TUH_data()
TUAR_data = TUH.findEdfDavid("TUH_data_sample")
file_selected_TUAR = TUAR_data.copy()

# Create empty dictionary for subjects
subjects_TUAR19 = defaultdict(dict)

counter = 0
save_dir = r"EEG2/scripts"
for edf in file_selected_TUAR:
    subject_ID = edf.split('_')[0]
    if subject_ID in subjects_TUAR19.keys():
        subjects_TUAR19[subject_ID][edf] = TUAR_data[edf].copy()
    else:
        subjects_TUAR19[subject_ID] = {edf: TUAR_data[edf].copy()}

    # debug counter for subject error
    counter += 1
    print("\n%s is patient: %i\n" % (edf, counter))

    # initialize hierarchical dict
    proc_subject = subjects_TUAR19[subject_ID][edf]
    proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1*.25,
                              read_raw_edf_param={'preload': True})

    # prepare raw
    proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
    TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Cz']

    proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
    proc_subject["rawData"].reorder_channels(TUH_pick)
    pipeline(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60, downSam=250) # "standard_1005" "easycap-M1"

    # Generate output windows for (X,y) as (tensor, label)
    #proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times[-1],

