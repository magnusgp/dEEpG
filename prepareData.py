
#import David_LoadTUHData_Andreas as loadTUH
import DavidLoadData as David
#import DavidPreprocessing
import os
import os.path
import mne
from mne.io import read_raw_edf
from collections import defaultdict
from datetime import datetime, timezone
import torch, re, warnings
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from eegProcess import TUH_rename_ch, nonPipeline, spectrogramMake, slidingWindow, pipeline

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
        #return self.readRawEdf().read_raw_edf(self.EEG_dict[id]["path"], preload=True)

    def loadAllRaw(self):
        EEG_raw_dict={}
        for id in range(self.EEG_count):
            EEG_raw_dict[id] = self.loadOneRaw(id)
        self.EEG_raw_dict=EEG_raw_dict

    def readRawEdf(self, edfDict=None, tWindow=120, tStep=30,
                   read_raw_edf_param={'preload': True, "stim_channel": "auto"}):
        #### This function i copied from https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG/blob/master/Transfer%20learning%20project/eegProcess.py##
        try:
            edfDict["rawData"] = read_raw_edf(edfDict["path"], **read_raw_edf_param)
            edfDict["fS"] = edfDict["rawData"].info["sfreq"]
            t_start = edfDict["rawData"].annotations.orig_time
            if t_start.timestamp() <= 0:
                edfDict["t0"] = datetime.fromtimestamp(0, tz=timezone.utc)
                t_last = edfDict["t0"].timestamp() + edfDict["rawData"]._last_time + 1 / edfDict["fS"]
                edfDict["tN"] = datetime.fromtimestamp(t_last, tz=timezone.utc)
            else:
                t_last = t_start.timestamp() + edfDict["rawData"]._last_time + 1 / edfDict["fS"]
                edfDict["t0"] = t_start  # datetime.fromtimestamp(t_start.timestamp(), tz=timezone.utc)
                edfDict["tN"] = datetime.fromtimestamp(t_last, tz=timezone.utc)

            edfDict["tWindow"] = float(tWindow)  # width of EEG sample window, given in (sec)
            edfDict["tStep"] = float(tStep)  # step/overlap between EEG sample windows, given in (sec)

        except:
            print("error break please inspect:\n %s\n~~~~~~~~~~~~" % edfDict["rawData"].filenames[0])

        return edfDict

    def label_TUH(annoPath=False, window=[0, 0], saveDir=os.getcwd(), header=None):
        ## From https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG/blob/0bfd1a9349f60f44e6f7df5aa6820434e44263a2/Transfer%20learning%20project/eegLoader.py#L52 ##
        if type(saveDir) is str:
            df = pd.read_csv(saveDir + annoPath, sep=" ", skiprows=1, header=header)
            df.fillna('null', inplace=True)
            within_con0 = (df[0] <= window[0]) & (window[0] <= df[1])
            within_con1 = (df[0] <= window[1]) & (window[1] <= df[1])
            label_TUH = df[df[0].between(window[0], window[1]) |
                           df[1].between(window[0], window[1]) |
                           (within_con0 & within_con1)]
            label_df = label_TUH.rename(columns={0: 't_start', 1: 't_end', 2: 'label', 3: 'confidence'})["label"]
            return_list = label_df.to_numpy().tolist()
        else:
            return_list = saveDir
        return return_list

    def prep(self):
        subjects_TUAR19 = defaultdict(dict)
        for k in range(len(self.EEG_dict)):
            subjects_TUAR19[k] = {'path':self.EEG_dict[k]['path']}

            proc_subject = subjects_TUAR19[k]
            proc_subject = self.readRawEdf(proc_subject, tWindow=1, tStep=1*.25,read_raw_edf_param={'preload': True})

            proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
            TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz'] #A1, A2 removed
            proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
            proc_subject["rawData"].reorder_channels(TUH_pick)

            pipeline(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60,
                     downSam=250)  # "standard_1005" "easycap-M1"

            print('done')

path="D:/fagprojekt/fagprojekt_data"
save_dir="D:/fagprojekt"
TUH=TUH_data()
TUH.findEdf(path=path)
#David.label_TUH(annoPath='\\'+os.path.split(TUH.EEG_dict[0]['csvpath'])[1],saveDir=os.path.split(TUH.EEG_dict[0]['csvpath'])[0])
print(TUH.EEG_dict)
#TUH.loadAllRaw()
TUH.prep()


#print(TUH.EEG_raw_dict)
#TUH.EEG_raw_dict[0].plot(duration=4)

