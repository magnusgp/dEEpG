import os, mne, time, re
from mne.io import read_raw_edf
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import torch
from preprocessFunctions import simplePreprocess, rereference, preprocessRaw
import matplotlib.pyplot as plt
from scipy import signal, stats
from raw_utils import oneHotEncoder
from tqdm import *
from labelFunctions import label_TUH, annotate_TUH, solveLabelChannelRelation

plt.rcParams["font.family"] = "Times New Roman"

##These functions are either inspired from or modified copies of code written by David Nyrnberg:
# https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG/tree/0bfd1a9349f60f44e6f7df5aa6820434e44263a2/Transfer%20learning%20project


class TUH_data:
    def __init__(self, path):
        ### Makes dictionary of all edf files
        EEG_count = 0
        EEG_dict = {}
        index_patient_df = pd.DataFrame(columns=['index', 'patient_id'])
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".edf")]:
                """For every edf file found somewhere in the directory, it is assumed the folders hold the structure: 
                ".../id/patientId/sessionId/edfFile".
                Therefore the path is split backwards and the EEG_dict updated with the found ids/paths.
                Furthermore it is expected that a csv file will always be found in the directory."""
                session_path_split = os.path.split(dirpath)
                patient_path_split = os.path.split(session_path_split[0])
                id_path_split = os.path.split(patient_path_split[0])
                EEG_dict.update({EEG_count: {"id": id_path_split[1],
                                             "patient_id": patient_path_split[1],
                                             "session": session_path_split[1],
                                             "path": os.path.join(dirpath, filename),
                                             "csvpath": os.path.join(dirpath, os.path.splitext(filename)[0]+'.csv')}})
                new_index_patient = pd.DataFrame({'index': EEG_count,'patient_id': EEG_dict[EEG_count]["patient_id"]}, index = [EEG_count])
                index_patient_df=pd.concat([index_patient_df, new_index_patient])
                EEG_count += 1
        self.index_patient_df = index_patient_df
        self.EEG_dict = EEG_dict
        self.EEG_count = EEG_count

    def readRawEdf(self, edfDict=None, tWindow=120, tStep=30,
                   read_raw_edf_param={'preload': True, "stim_channel": "auto"}):
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

    def electrodeCLFPrep(self, tWindow=100, tStep=100 *.25,plot=False):
        tic = time.time()
        subjects_TUAR19 = defaultdict(dict)
        for k in tqdm(range(len(self.EEG_dict))):

            annotations=solveLabelChannelRelation(self.EEG_dict[k]['csvpath'])

            #subjects_TUAR19[k] = {'path': self.EEG_dict[k]['path']}

            #self.EEG_dict[k] = subjects_TUAR19[k]
            self.EEG_dict[k] = self.readRawEdf(self.EEG_dict[k], tWindow=tWindow, tStep=tStep,
                                           read_raw_edf_param={'preload': True})

            self.EEG_dict[k]["rawData"] = TUH_rename_ch(self.EEG_dict[k]["rawData"])
            TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
            self.EEG_dict[k]["rawData"].pick_channels(ch_names=TUH_pick)
            self.EEG_dict[k]["rawData"].reorder_channels(TUH_pick)

            if k == 0 and plot:
                #Plot the energy voltage potential against frequency.
                self.EEG_dict[k]["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

                raw_anno = annotate_TUH(self.EEG_dict[k]["rawData"],df=annotations)
                raw_anno.plot()
                plt.title("Untouched raw signal")
                plt.show()

            simplePreprocess(self.EEG_dict[k]["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=100, notchfq=60,
                     downSam=250)

            if k == 0:
                self.sfreq = self.EEG_dict[k]["rawData"].info["sfreq"]
                self.ch_names = self.EEG_dict[k]["rawData"].info["ch_names"]
                if plot:
                    self.EEG_dict[k]["rawData"].plot_psd(tmax=np.inf, fmax=125, average=True)

                    raw_anno = annotate_TUH(self.EEG_dict[k]["rawData"], df=annotations)
                    raw_anno.plot()
                    plt.title("Raw signal after simple preprocessing")
                    plt.show()


            # Generate output windows for (X,y) as (array, label)
            self.EEG_dict[k]["labeled_windows"] = slidingRawWindow(self.EEG_dict[k],
                                                                    t_max=self.EEG_dict[k]["rawData"].times[-1],
                                                                    tStep=self.EEG_dict[k]["tStep"],
                                                                    electrodeCLF=True,df=annotations)
        toc = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~\n"
              "it took %imin:%is to run electrode classifier preprocess-pipeline for %i file(s)\nwith window length [%.2fs] and t_step [%.2fs]"
              "\n~~~~~~~~~~~~~~~~~~~~\n" % (int((toc - tic) / 60), int((toc - tic) % 60), len(self.EEG_dict),
                                            tWindow, tStep))


    def collectWindows(self,id=None):
        # Helper funtion to makeDatasetFromIds
        # Collects all windows from one session into list
        Xwindows = []
        Ywindows = []
        windowInfo = []
        for window in self.EEG_dict[id]["labeled_windows"].values():
            Xwindows=Xwindows+[window[0]]
            if window[1] == ['elec']:
                Ywindows.append([1])
            else:
                Ywindows.append([0])
            #Ywindows.append(1 if window[1]==['elec'] else 0)
            # save info about which raw file and start time and end time this window is.
            windowInfo.append([{'patient_id':self.EEG_dict[id]['patient_id'], 't_start':window[2], 't_end':window[3]}])

        return Xwindows,Ywindows,windowInfo


    def makeDatasetFromIds(self,ids=None):
        # Needs list of Ids/indexes in EEG_dict. Function electrodeCLFPrep should be called beforehand.
        # Collects all windows of all given ids into one list of X (window data) and Y corresponding labels
        Xwindows = []
        Ywindows = []
        windowInfo = []
        for id in ids:
            Xwind,Ywind,windowIn=self.collectWindows(id=id)
            Xwindows.append(Xwind)
            Ywindows.append(Ywind)
            windowInfo.append(windowIn)

        return Xwindows,Ywindows,windowInfo

    def specMaker(self):
        Xwindows=self.Xwindows
        Freq = self.sfreq
        tWindow=self.tWindow
        tStep=self.tStep
        overlap=(tWindow-tStep)/tWindow #The amount of the window that overlaps with the next window.

        for k in range(len(Xwindows)):
            spectrogramMake(Xwindows[k], Freq,FFToverlap=overlap,tWindow=tWindow, show_chan_num=1,chan_names=self.ch_names)

# renames TUH channels to conventional 10-20 system
def TUH_rename_ch(MNE_raw=False):
    # MNE_raw
    # mne.channels.rename_channels(MNE_raw.info, {"PHOTIC-REF": "PROTIC"})
    for i in MNE_raw.info["ch_names"]:
        reSTR = r"(?<=EEG )(\S*)(?=-REF)"  # working reSTR = r"(?<=EEG )(.*)(?=-REF)"
        reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ']

        if re.search(reSTR, i) and re.search(reSTR, i).group() in reLowC:
            lowC = i[0:5]+i[5].lower()+i[6:]
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, lowC)[0]})
        elif i == "PHOTIC-REF":
            mne.channels.rename_channels(MNE_raw.info, {i: "PHOTIC"})
        elif re.search(reSTR, i):
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, i)[0]})
        else:
            continue
            # print(i)
    print(MNE_raw.info["ch_names"])
    return MNE_raw


def makeArrayWindow(MNE_raw=None, t0=0, tWindow=120):
    # take a raw signal and make a window given time specifications. Outputs an array, because of raw.get_data().
    chWindows = MNE_raw.get_data(start=int(t0), stop=int(t0 + tWindow), reject_by_annotation=None, picks=['eeg'])
    return chWindows


def slidingRawWindow(EEG_series=None, t_max=0, tStep=1,electrodeCLF=False, df=False):
    #If electrodeCLF is set to true, the function outputs a window per channel
    # with labels assigned only for this channel.

    # catch correct sample frequency and end sample
    edf_fS = EEG_series["rawData"].info["sfreq"]
    t_N = int(t_max * edf_fS)

    # ensure window-overlaps progress in sample interger
    if float(tStep * edf_fS) == float(int(tStep * edf_fS)):
        t_overlap = int(tStep * edf_fS)
    else:
        t_overlap = int(tStep * edf_fS)
        overlap_change = 100 - (t_overlap / edf_fS) * 100
        print("\n  tStep [%.3f], overlap does not equal an interger [%f] and have been rounded to %i"
              "\n  equaling to %.1f%% overlap or %.3fs time steps\n\n"
              % (tStep, tStep * edf_fS, t_overlap, overlap_change, t_overlap / edf_fS))

    # initialize variables for segments
    window_EEG = defaultdict(tuple)
    window_width = int(EEG_series["tWindow"] * edf_fS)
    label_path = EEG_series['path'].split(".edf")[0] + ".csv"

    # segment all N-1 windows (by positive lookahead)
    for i in range(0, t_N - window_width, t_overlap):
        t_start = i / edf_fS
        t_end = (i + window_width) / edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = makeArrayWindow(EEG_series["rawData"], t0=i, tWindow=window_width)  # , show_chan_num=0) #)
        if electrodeCLF:
            for i in range(len(window_data)):
                chan=EEG_series['rawData'].info['ch_names'][i]
                channel_label=label_TUH(dataFrame=df, window=[t_start, t_end],channel=chan)
                oneHotChan=(np.asarray(EEG_series['rawData'].info['ch_names'])==chan)*1
                window_EEG[window_key+f"{i}"] = (np.concatenate((oneHotChan,window_data[i])), channel_label,t_start,t_end)
        else:
            window_label = label_TUH(dataFrame=df, window=[t_start, t_end],channel=None)  # , saveDir=annoDir)
            window_EEG[window_key] = (window_data, window_label)
    # window_N segments (by negative lookahead)
    if t_N % t_overlap != 0:
        t_start = (t_N - window_width) / edf_fS
        t_end = t_N / edf_fS
        window_key = "window_%.3fs_%.3fs" % (t_start, t_end)
        window_data = makeArrayWindow(EEG_series["rawData"], t0=i, tWindow=window_width)
        if electrodeCLF:
            for i in range(len(window_data)):
                chan=EEG_series['rawData'].info['ch_names'][i]
                channel_label=label_TUH(dataFrame=df, window=[t_start, t_end],channel=chan)
                oneHotChan=(np.asarray(EEG_series['rawData'].info['ch_names'])==chan)*1
                window_EEG[window_key+f"{i}"] = (np.concatenate((oneHotChan,window_data[i])), channel_label,t_start,t_end)
        else:
            window_label = label_TUH(dataFrame=df, window=[t_start, t_end])  # , saveDir=annoDir)
            window_EEG[window_key] = (window_data, window_label)
    return window_EEG

def plotWindow(EEG_series,label="null", t_max=0, t_step=1):
    edf_fS = EEG_series["rawData"].info["sfreq"]
    t_N = int(t_max * edf_fS)
    window_width = int(EEG_series["tWindow"] * edf_fS)
    label_path = EEG_series['path'].split(".edf")[0] + ".csv"

    for i in range(0, t_N - window_width, t_step):
        t_start = i / edf_fS
        t_end = (i + window_width) / edf_fS
        window_label = label_TUH(dataFrame=df, window=[t_start, t_end])
        if len(window_label)==1 & window_label[0]==label:
            return EEG_series["rawData"].plot(t_start=t_start, t_end=t_end)
    return None

def spectrogramMake(MNE_window=None, freq = None, tWindow=100, crop_fq=45, FFToverlap=None, show_chan_num=None,chan_names=None):
    try:
        edfFs = freq
        chWindows = MNE_window

        if FFToverlap is None:
            specOption = {"x": chWindows, "fs": edfFs, "mode": "psd"}
        else:
            window = signal.get_window(window=('tukey', 0.25), Nx=int(tWindow))  # TODO: error in 'Nx' & 'noverlap' proportions
            specOption = {"x": chWindows, "fs": edfFs, "window": window, "noverlap": int(tWindow*FFToverlap), "mode": "psd"}

        fAx, tAx, Sxx = signal.spectrogram(**specOption)
        normSxx = stats.zscore(np.log(Sxx[:, fAx <= crop_fq, :] + 2**-52)) #np.finfo(float).eps))
        if isinstance(show_chan_num, int):
            plot_spec = plotSpec(ch_names=chan_names, chan=show_chan_num,
                                 fAx=fAx[fAx <= crop_fq], tAx=tAx, Sxx=normSxx)
            plot_spec.show()
    except:
        print("pause here")
        # fTemp, tTemp, SxxTemp = signal.spectrogram(chWindows[0], fs=edfFs)
        # plt.pcolormesh(tTemp, fTemp, np.log(SxxTemp))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.title("channel spectrogram: "+MNE_raw.ch_names[0])
        # plt.ylim(0,45)
        # plt.show()

    return torch.tensor(normSxx.astype(np.float16)) # for np delete torch.tensor

def plotSpec(ch_names=False, chan=False, fAx=False, tAx=False, Sxx=False):
    # fTemp, tTemp, SxxTemp = signal.spectrogram(chWindows[0], fs=edfFs)
    # normSxx = stats.zscore(np.log(Sxx[:, fAx <= cropFq, :] + np.finfo(float).eps))
    plt.pcolormesh(tAx, fAx, Sxx[chan, :, :])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("channel spectrogram: " + ch_names[chan])

    return plt

