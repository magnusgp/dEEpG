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

plt.rcParams["font.family"] = "Times New Roman"

##These functions are either inspired from or modified copies of code written by David Nyrnberg:
# https://github.com/DavidEnslevNyrnberg/DTU_DL_EEG/tree/0bfd1a9349f60f44e6f7df5aa6820434e44263a2/Transfer%20learning%20project


class TUH_data:
    def __init__(self, path):
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
                EEG_dict.update({EEG_count: {"id": id_path_split[1],
                                             "patient_id": patient_path_split[1],
                                             "session": session_path_split[1],
                                             "path": os.path.join(dirpath, filename),
                                             "csvpath": os.path.join(dirpath, os.path.splitext(filename)[0]+'.csv')}})
                EEG_count += 1
        self.EEG_dict = EEG_dict
        self.EEG_count = EEG_count

    """ These functions could probably be deleted, but are nice in case we want a quick plot of a raw file.
    def loadOneRaw(self, id):
        return mne.io.read_raw_edf(self.EEG_dict[id]["path"], preload=True)

    def loadAllRaw(self):
        EEG_raw_dict = {}
        for id in range(self.EEG_count):
            EEG_raw_dict[id] = self.loadOneRaw(id)
        self.EEG_raw_dict = EEG_raw_dict
        """

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
        Xwindows = []
        Ywindows = []
        windowInfo = []
        for k in tqdm(range(len(self.EEG_dict))):
            subjects_TUAR19[k] = {'path': self.EEG_dict[k]['path']}

            proc_subject = subjects_TUAR19[k]
            proc_subject = self.readRawEdf(proc_subject, tWindow=tWindow, tStep=tStep,
                                           read_raw_edf_param={'preload': True})
            if k == 0 and plot:
                #Plot the energy voltage potential against frequency.
                #proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

                raw_anno = annotate_TUH(proc_subject["rawData"],annoPath=self.EEG_dict[k]["csvpath"])
                raw_anno.plot()
                plt.show()

            proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
            TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
            proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
            proc_subject["rawData"].reorder_channels(TUH_pick)

            if k == 0 and plot:
                #Plot the energy voltage potential against frequency.
                proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

                raw_anno = annotate_TUH(proc_subject["rawData"],annoPath=self.EEG_dict[k]["csvpath"])
                raw_anno.plot()
                plt.title("Untouched raw signal")
                plt.show()

            simplePreprocess(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60,
                     downSam=250)

            if k == 0:

                self.sfreq = proc_subject["rawData"].info["sfreq"]
                self.ch_names = proc_subject["rawData"].info["ch_names"]
                if plot:
                    proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=125, average=True)

                    raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
                    raw_anno.plot()
                    plt.title("Raw signal after simple preprocessing")
                    plt.show()


            # Generate output windows for (X,y) as (array, label)
            proc_subject["preprocessing_output"] = slidingRawWindow(proc_subject,
                                                                    t_max=proc_subject["rawData"].times[-1],
                                                                    tStep=proc_subject["tStep"],
                                                                    electrodeCLF=True)

            for window in proc_subject["preprocessing_output"].values():
                Xwindows.append(window[0])
                Ywindows.append(window[1])
                #save info about which raw file and start time and end time this window is.
                windowInfo.append(self.EEG_dict[k]['path'],window[2],window[3])

        toc = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~\n"
              "it took %imin:%is to run electrode classifier preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
              "\n~~~~~~~~~~~~~~~~~~~~\n" % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects_TUAR19),
                                            tWindow, tStep))

        Ywindows = oneHotEncoder(Ywindows, enumerate_labels=True, clfbin=True)

        return Xwindows, Ywindows,windowInfo

    def prep(self, tWindow=100, tStep=100 *.25,plot=False):
        self.tWindow=tWindow
        self.tStep=tStep
        tic = time.time()
        subjects_TUAR19 = defaultdict(dict)
        Xwindows = []
        Ywindows = []
        for k in range(len(self.EEG_dict)):
            subjects_TUAR19[k] = {'path': self.EEG_dict[k]['path']}

            proc_subject = subjects_TUAR19[k]
            proc_subject = self.readRawEdf(proc_subject, tWindow=tWindow, tStep=tStep,
                                           read_raw_edf_param={'preload': True})
            if k == 0 and plot:
                #Plot the energy voltage potential against frequency.
                #proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

                raw_anno = annotate_TUH(proc_subject["rawData"],annoPath=self.EEG_dict[k]["csvpath"])
                raw_anno.plot()
                plt.show()

            proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
            TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
            proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
            proc_subject["rawData"].reorder_channels(TUH_pick)

            if k == 0 and plot:
                #Plot the energy voltage potential against frequency.
                proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

                raw_anno = annotate_TUH(proc_subject["rawData"],annoPath=self.EEG_dict[k]["csvpath"])
                raw_anno.plot()
                plt.show()

            preprocessRaw(proc_subject["rawData"], cap_setup="standard_1005", lpfq=1, hpfq=40, notchfq=60,
                     downSam=250)

            if k == 0:

                self.sfreq = proc_subject["rawData"].info["sfreq"]
                self.ch_names = proc_subject["rawData"].info["ch_names"]
                if plot:
                    proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=125, average=True)

                    raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
                    raw_anno.plot()
                    plt.show()

            # Generate output windows for (X,y) as (array, label)
            proc_subject["preprocessing_output"] = slidingRawWindow(proc_subject,
                                                                    t_max=proc_subject["rawData"].times[-1],
                                                                    tStep=proc_subject["tStep"])

            for window in proc_subject["preprocessing_output"].values():
                Xwindows.append(window[0])
                Ywindows.append(window[1])

        toc = time.time()
        print("\n~~~~~~~~~~~~~~~~~~~~\n"
              "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
              "\n~~~~~~~~~~~~~~~~~~~~\n" % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects_TUAR19),
                                            tWindow, tStep))

        self.Xwindows = Xwindows
        self.Ywindows = Ywindows

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

def label_TUH(annoPath=False, window=[0, 0], header=None,channel=None):  # saveDir=os.getcwd(),
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)
    df.fillna('null', inplace=True)
    within_con0 = (df[2] <= window[0]) & (window[0] <= df[3])
    within_con1 = (df[2] <= window[1]) & (window[1] <= df[3])
    if channel:
        chan_names = df[1].to_numpy().tolist()
        low_char = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}
        for i in range(len(chan_names)):
            # remove numbers behind channel names:
            chan_names[i] = [chan_names[i][:-3], chan_names[i][-2:]]

            # Loop through all channel names in reverse order, so if something is removed it does not affect other index.
            # Change certain channels to have smaller letters:
            for k in range(len(chan_names[i]) - 1, -1, -1):
                if chan_names[i][k] in low_char:
                    chan_names[i][k] = low_char[chan_names[i][k]]
        label_TUH = df[(df[2].between(window[0], window[1]) |
                       df[3].between(window[0], window[1]) |
                       (within_con0 & within_con1))
                       & (np.sum(np.asarray(chan_names)==np.asarray(channel),axis=1).tolist())
                        & ((df[4].to_numpy()=='elec')|
                           (df[4].to_numpy()=='musc_elec')|
                           (df[4].to_numpy()=='eyem_elec')|
                           (df[4].to_numpy()=='shiv_elec')|
                           (df[4].to_numpy()=='chew_elec'))]
    else:
        label_TUH = df[df[2].between(window[0], window[1]) |
                   df[3].between(window[0], window[1]) |
                   (within_con0 & within_con1)]
    label_df = label_TUH.rename(columns={2: 't_start', 3: 't_end', 4: 'label', 5: 'confidence'})["label"]  # Renamer headers i pandas dataen
    return_list = label_df.to_numpy().tolist()  # Outputter kun listen af label navne i vinduet, fx ["eyem", "null"]
    if return_list==[]:
        return_list=['null']
    elif channel:
        return_list=['elec']
    return return_list


def makeArrayWindow(MNE_raw=None, t0=0, tWindow=120):
    # take a raw signal and make a window given time specifications. Outputs an array, because of raw.get_data().
    chWindows = MNE_raw.get_data(start=int(t0), stop=int(t0 + tWindow), reject_by_annotation=None, picks=['eeg'])
    return chWindows


def slidingRawWindow(EEG_series=None, t_max=0, tStep=1,electrodeCLF=False):
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
                channel_label=label_TUH(annoPath=label_path, window=[t_start, t_end],channel=chan)
                oneHotChan=(np.asarray(EEG_series['rawData'].info['ch_names'])==chan)*1
                window_EEG[window_key+f"{i}"] = (np.concatenate((oneHotChan,window_data[i])), channel_label,t_start,t_end)
        else:
            window_label = label_TUH(annoPath=label_path, window=[t_start, t_end],channel=None)  # , saveDir=annoDir)
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
                channel_label=label_TUH(annoPath=label_path, window=[t_start, t_end],channel=chan)
                oneHotChan=(np.asarray(EEG_series['rawData'].info['ch_names'])==chan)*1
                window_EEG[window_key+f"{i}"] = (np.concatenate((oneHotChan,window_data[i])), channel_label,t_start,t_end)
        else:
            window_label = label_TUH(annoPath=label_path, window=[t_start, t_end])  # , saveDir=annoDir)
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
        window_label = label_TUH(annoPath=label_path, window=[t_start, t_end])
        if len(window_label)==1 & window_label[0]==label:
            return EEG_series["rawData"].plot(t_start=t_start, t_end=t_end)
    return None

# The function "annotate_TUH()" takes a raw signal and a path for a csv file with annotations/labels in it.
# The annotations are read and added to the raw signal. The function is mainly made for the purpose of making
# plots with the artifacts showing.
def annotate_TUH(raw,annoPath=False, header=None):
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)
    t_start=df[2].to_numpy()
    dura=df[3].to_numpy()-t_start
    labels=df[4].to_numpy().tolist()
    chan_names=df[1].to_numpy().tolist()
    t_start=t_start.tolist()
    dura=dura.tolist()

    delete=[]
    low_char={'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz'}
    for i in range(len(chan_names)):
        #remove numbers behind channel names:
        chan_names[i]=[chan_names[i][:-3],chan_names[i][-2:]]

        # Loop through all channel names in reverse order, so if something is removed it does not affect other index.
        # Change certain channels to have smaller letters:
        for k in range(len(chan_names[i])-1,-1,-1):
            if chan_names[i][k] in low_char:
                chan_names[i][k]=low_char[chan_names[i][k]]

            # If channel names are not in the raw info their are removed from an annotation:
            if chan_names[i][k] not in raw.ch_names:
                chan_names[i].remove(chan_names[i][k])

        # If no channel names are left for an annotation its index is saved for later removal entirely:
        # (It could potentially just be annotated for the whole signal)
        if not chan_names[i]:
            delete.append(i)


    #removes every annotation that cannot be handled backwards:
    for ele in sorted(delete,reverse=True):
        print(f"Annotation {labels[ele]} on non-existing channel {chan_names[ele]} removed from annotations.")
        del t_start[ele], dura[ele],labels[ele],chan_names[ele]

    anno=mne.Annotations(onset=t_start,
                            duration=dura,
                              description=labels,
                                ch_names=chan_names)

    raw_anno=raw.set_annotations(anno)
    return raw_anno

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

def solveLabelChannelRelation(annoPath, header = None):
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)

    # Split pairs into single channels
    #channel_pairs=df[1].to_numpy().tolist()
    #channel_pairs=[n.split('-') for n in channel_pairs]
    #channel_unique=list(set([n for n in channel_pairs]))
    #Creating data frame:
    anno_df=pd.DataFrame(columns=['channel','t_start','t_end','label'])

    #checking every entry in label data:
    for i in tqdm(range(len(df))):
        chan1, chan2=df[1][i].split('-')
        # Only check row against rows further down:
        temp = df[i+1:]
        # Only rows with same label:
        temp = temp[temp[4] == df[4][i]]

        # Only overlap in time:
        temp_time = temp[((df[2][i]<=temp[2]) & (temp[2]<=df[3][i])) |
                         ((df[2][i]<=temp[3]) & (temp[3]<=df[3][i])) |
                         ((temp[2]<df[2][i]) & (df[3][i]<temp[3]))]

        for k in temp_time.index:
            #check if first channel is a match with one in the new channel pair:
            channel = None
            if chan1 in temp_time[1][k].split('-'):
                channel = chan1
            elif chan2 in temp_time[1][k].split('-'):
                channel = chan2
            if channel in [chan1, chan2]:
                t_start = max(df[2][i], temp_time[2][k])
                t_end = min(df[3][i], temp_time[3][k])

                anno_new = pd.DataFrame({'channel': [channel], 't_start': [t_start],
                                         't_end': [t_end], 'label': [df[4][i]]})

                #if ((anno_new['channel'] == anno_df['channel']) & (anno_new['t_start'] == anno_df['t_start'])
                #    & (anno_new['t_end'] == anno_df['t_end']) & (anno_new['label'] == anno_df['label'])).any():
                #    anno_df.append(anno_new)

                duplicates=anno_df[ ([df[4][i]]==anno_df['label']) &
                         (chan1==anno_df['channel'])  &
                        (((t_start<=anno_df['t_start']) & (anno_df['t_start']<=t_end)) |
                         ((t_start<=anno_df['t_end']) & (anno_df['t_end']<=t_end)) |
                         ((anno_df['t_start']<t_start) & (t_end<anno_df['t_end'])))]

                if duplicates:
                    new_t_start = min(duplicates['t_start'],t_start)
                    new_t_end = max(duplicates['t_end'], t_start)

                    anno_new = pd.DataFrame({'channel': [chan1], 't_start': [new_t_start],
                                             't_end': [new_t_end], 'label': [df[4][i]]})
                    anno_df.append(anno_new)

                    #delete overlapping rows from behind so the indexes are not confused:
                    for n in len(duplicates):
                        index=duplicates.index[-n]
                        anno_df.drop(index=index)

                # if no duplicates/overlaps found, then just save annotation for channel:
                else:
                    anno_new = pd.DataFrame({'channel': [chan1], 't_start': [t_start],
                                             't_end': [t_end], 'label': [df[4][i]]})
                    anno_df.append(anno_new)

            else:
                pass
                #print("No checks passed \n"
                #      "Channel 1: {} ({}-{} s) \n"
                #      "Channel 2: {} ({}-{} s)".format(chan1, df[2][i], df[3][i], chan2, df[2][k], df[3][k]))

        #check that annotation is covered in the dataframe on either one of the channels or both
        if not anno_df.empty:
            #check if annotation is covered in the dataframe
            if not (anno_df['t_start'].min() >= df[2][i] and anno_df['t_end'].max() <= df[3][i]):
                print("Annotation not covered in dataframe \n"
                      "Channel 1: {} ({}-{} s) \n"
                      "Channel 2: {} ({}-{} s)".format(chan1, df[2][i], df[3][i], chan2, df[2][k], df[3][k]))
                anno_df = pd.DataFrame()
            else:
                pass

    return anno_df




def labelChannels(annoPath, header = None):
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)

    # Split pairs into single channels
    channel_pairs = df[1].to_numpy().tolist()
    channel_pairs = [n.split('-') for n in channel_pairs]

    # Creating data frame:
    anno_df = pd.DataFrame(columns=['channel', 't_start', 't_end', 'label'])

    anno_dict = defaultdict(lambda: (0, 0))

    # Checking every entry in label data:
    for i in tqdm(range(len(channel_pairs))):
        # Check if label is the same in the two rows, eg. 'elec'=='elec':
        # Create two variables, one for each channel in the pair:
        chan1, chan2 = channel_pairs[i]
        for k in range(i+1,len(channel_pairs)):
            #Check if label is the same in the two rows, eg. 'elec'=='elec':
            if df[4][i] == df[4][k]:
                #Add both time frames to anno_dict
               anno_dict[chan1] = (df[2][i], df[3][i])
               anno_dict[chan2] = (df[2][i], df[3][i])



if __name__ == "__main__":
    path = "../TUH_data_sample/131/00013103/s001_2015_09_30/00013103_s001_t000.csv"

    solveLabelChannelRelation(annoPath=path)

    #labelChannels(annoPath=path)