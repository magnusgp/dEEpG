import mne
from torch.utils.data import DataLoader

from braindecode.datasets import TUH
from braindecode.preprocessing import create_fixed_length_windows


def findEdf(path):
    ### Makes dictionary of all edf files
    EEG_count = 0
    EEG_dict = {}

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".edf")]:
            """For every edf file found somewhere in the directory, it is assumed the folders hold the structure: 
            ".../id/patientId/sessionId/edfFile".
            Therefore the path is split backwards and the EEG_dict updated with the found ids/paths.
            Furthermore it is expected that a csv file will always be found in the directory."""
            session_path_split = os.path.split(dirpath)
            patient_path_split = os.path.split(session_path_split[0])
            id_path_split = os.path.split(patient_path_split[0])
            EEG_dict.update({EEG_count: {"id": id_path_split[1], "patient_id": patient_path_split[1],
                                         "session": session_path_split[1],
                                         "path": os.path.join(dirpath, filename),
                                         "csvpath": os.path.join(dirpath, os.path.splitext(filename)[0] + '.csv')}})
            EEG_count += 1
    self.EEG_dict = EEG_dict
    self.EEG_count = EEG_count

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

TUH_PATH = 'D:fagprojekt/fagprojekt_data'
tuh = TUH(path=TUH_PATH,recording_ids=None,target_name=(),preload=False,add_physician_reports=False)
tuh.description