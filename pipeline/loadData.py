import os
import os.path
import mne

class TUH_data:
    def __init__(self):
        pass

    def findEdf(self,path):
        ### Makes dictionary of all edf files
        EEG_count = 0
        EEG_dict = {}

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".edf")]:
                """For every edf file found somewhere in the directory, it is assumed the folders hold the structure: 
                ".../id/patientId/sessionId/edfFile".
                Therefore the path is split backwards and the EEG_dict updated with the found ids/paths.
                Furthermore it is expected that a csv file will always be found in the directory."""
                session_path_split=os.path.split(dirpath)
                patient_path_split = os.path.split(session_path_split[0])
                id_path_split=os.path.split(patient_path_split[0])
                EEG_dict.update({EEG_count: {"id": id_path_split[1], "patient_id": patient_path_split[1], "session":  session_path_split[1],
                                          "path": os.path.join(dirpath, filename),"csvpath": os.path.join(dirpath, os.path.splitext(filename)[0]+'.csv')}})
                EEG_count+=1
        self.EEG_dict = EEG_dict
        self.EEG_count=EEG_count

    def loadOneRaw(self,id):
        return mne.io.read_raw_edf(self.EEG_dict[id]["path"], preload=True)

    def loadAllRaw(self):
        EEG_raw_dict={}
        for id in range(self.EEG_count):
            EEG_raw_dict[id] = self.loadOneRaw(id)
        self.EEG_raw_dict=EEG_raw_dict