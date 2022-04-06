

import os, mne, time
import os.path
from mne.io import read_raw_edf
from collections import defaultdict
from datetime import datetime, timezone
import torch, re, warnings
import pandas as pd
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from eegProcess import TUH_rename_ch, nonPipeline, spectrogramMake, slidingWindow, pipeline
from braindecode.datasets import create_from_X_y
from pipeline.raw_utils import labelInt

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

    def prep(self, saveDir):
        tic = time.time()
        subjects_TUAR19 = defaultdict(dict)
        Xwindows=[]
        Ywindows=[]
        for k in range(len(self.EEG_dict)):
            subjects_TUAR19[k] = {'path':self.EEG_dict[k]['path']}

            proc_subject = subjects_TUAR19[k]
            proc_subject = self.readRawEdf(proc_subject, tWindow=100, tStep=100*.25,read_raw_edf_param={'preload': True})

            proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
            TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz'] #A1, A2 removed
            proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
            proc_subject["rawData"].reorder_channels(TUH_pick)

            if k == 0:
                self.sfreq = proc_subject["rawData"].info["sfreq"]
                self.ch_names = proc_subject["rawData"].info["ch_names"]


            pipeline(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60,
                     downSam=250)

            # Generate output windows for (X,y) as (tensor, label)
            proc_subject["preprocessing_output"] = slidingRawWindow(proc_subject, t_max=proc_subject["rawData"].times[-1],
                                                                 tStep=proc_subject["tStep"], FFToverlap=0.75,
                                                                 crop_fq=24,
                                                                 annoDir=self.EEG_dict[k]['csvpath'],
                                                                 localSave={"sliceSave": True,
                                                                            "saveDir": saveDir + r'/tensor',
                                                                            "local_return": False})

            for window in proc_subject["preprocessing_output"].values():
                Xwindows.append(window[0])
                Ywindows.append(window[1])
                #Xraw=np.concatenate((Xraw,np.array([window[0]])))
                #Y=np.concatenate((Y,np.array([window[1]])))

        toc = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~\n"
              "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
              "\n~~~~~~~~~~~~~~~~~~~~\n"% (int((toc-tic)/60), int((toc-tic) % 60), len(subjects_TUAR19),
                                           subjects_TUAR19[k]["tWindow"], subjects_TUAR19[k]["tStep"]))

        self.Xwindows=Xwindows
        self.Ywindows=Ywindows

def label_TUH(annoPath=False, window=[0,0], header=None): #saveDir=os.getcwd(),
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)
    df.fillna('null', inplace=True)
    within_con0 = (df[2] <= window[0]) & (window[0] <= df[3])
    within_con1 = (df[2] <= window[1]) & (window[1] <= df[3])
    label_TUH = df[df[2].between(window[0], window[1]) |
                   df[3].between(window[0], window[1]) |
                   (within_con0 & within_con1)]
    label_df = label_TUH.rename(columns={2: 't_start', 3: 't_end', 4: 'label', 5: 'confidence'})["label"] #Renamer headers i pandas dataen
    return_list = label_df.to_numpy().tolist() #Outputter kun listen af label navne i vinduet, fx ["eyem", "null"]
    return return_list

def makeRawWindow(MNE_raw=None, t0=0, tWindow=120):
    #take a raw signal and make a window given time specifications.
    chWindows = MNE_raw.get_data(start=int(t0), stop=int(t0+tWindow), reject_by_annotation="omit", picks=['eeg'])
    return chWindows

def slidingRawWindow(EEG_series=None, t_max=0, tStep=1, FFToverlap=None, crop_fq=45, annoDir=None,
                  localSave={"sliceSave":False, "saveDir":os.getcwd(), "local_return":False}):
    # catch correct sample frequency and end sample
    edf_fS = EEG_series["rawData"].info["sfreq"]
    t_N = int(t_max*edf_fS)

    # ensure window-overlaps progress in sample interger
    if float(tStep*edf_fS) == float(int(tStep*edf_fS)):
        t_overlap = int(tStep*edf_fS)
    else:
        t_overlap = int(tStep*edf_fS)
        overlap_change = 100-(t_overlap/edf_fS)*100
        print("\n  tStep [%.3f], overlap does not equal an interger [%f] and have been rounded to %i"
              "\n  equaling to %.1f%% overlap or %.3fs time steps\n\n"
              % (tStep, tStep*edf_fS, t_overlap, overlap_change, t_overlap/edf_fS))

    # initialize variables for segments
    window_EEG = defaultdict(tuple)
    window_width = int(EEG_series["tWindow"]*edf_fS)
    label_path = EEG_series['path'].split(".edf")[0] + ".csv"

    # segment all N-1 windows (by positive lookahead)
    for i in range(0, t_N-window_width, t_overlap):
        t_start = i/edf_fS
        t_end = (i+window_width)/edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = makeRawWindow(EEG_series["rawData"], t0=i, tWindow=window_width) # , show_chan_num=0) #)
        window_label = label_TUH(annoPath=label_path, window=[t_start, t_end])#, saveDir=annoDir)
        window_EEG[window_key] = (window_data, window_label)
    # window_N segments (by negative lookahead)
    if t_N % t_overlap != 0:
        t_start = (t_N - window_width)/edf_fS
        t_end = t_N/edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = makeRawWindow(EEG_series["rawData"], t0=i, tWindow=window_width)
        window_label = label_TUH(annoPath=label_path, window=[t_start, t_end])#, saveDir=annoDir)
        window_EEG[window_key] = (window_data, window_label)

    return window_EEG
    # save in RAM, disk or not
    """    if localSave["sliceSave"]:
        idDir = EEG_series["rawData"].filenames[0].split('\\')[-1].split('.')[0]
        if not os.path.exists(localSave["saveDir"] + "tempData\\"):
            os.mkdir(localSave["saveDir"] + "tempData\\")
        if not os.path.exists(localSave["saveDir"] + "tempData\\" + idDir):
            os.mkdir(localSave["saveDir"] + "tempData\\" + idDir)
        for k, v in window_EEG.items():
            torch.save(v, localSave["saveDir"] + "tempData\\%s\\%s.pt" % (idDir, k)) # for np del torch.save
    if not localSave["sliceSave"] or localSave["local_return"] is True:
        windowOut = window_EEG.copy()
    else:
        windowOut = None
    
    return windowOut"""

def annotate_TUH(raw,annoPath=False, header=None):
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)
    t_start=df[2].to_numpy()
    dura=df[3].to_numpy()-t_start
    labels=df[4].to_numpy().tolist()

    anno=mne.Annotations(onset=t_start.tolist(),
                            duration=dura.tolist(),
                              description=labels)

    raw_anno=raw.set_annotations(anno)
    return raw_anno

path="TUH_data_sample"
save_dir=os.getcwd()
TUH=TUH_data()
TUH.findEdf(path=path)
print(TUH.EEG_dict)
TUH.loadAllRaw()

raw_anno=annotate_TUH(TUH.EEG_raw_dict[0],annoPath=TUH.EEG_dict[0]['csvpath'])
raw_anno.plot(start=689,duration=30)
plt.show()

#windows_dataset = create_from_X_y(
 #   TUH.Xwindows, labelInt(TUH.Ywindows), drop_last_window=True, sfreq=TUH.sfreq, ch_names=TUH.ch_names,
  #  window_stride_samples=len(TUH.Xwindows[0][0]),
   # window_size_samples=len(TUH.Xwindows[0][0]),
#)

#windows_dataset.description


#TUH.EEG_raw_dict[0].plot(start=689,duration=10)
#plt.show()

#print(TUH.EEG_raw_dict)
#TUH.EEG_raw_dict[0].plot_psd()
#TUH.EEG_raw_dict[0].plot(duration=4)
#plt.show()

