# Prep function fra loadFunctions.py
# Taget fra linje 143
def prep(self, tWindow=100, tStep=100 * .25, plot=False):
    self.tWindow = tWindow
    self.tStep = tStep
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
            # Plot the energy voltage potential against frequency.
            # proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

            raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
            raw_anno.plot()
            plt.show()

        proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
        TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
        proc_subject["rawData"].pick_channels(ch_names=TUH_pick)
        proc_subject["rawData"].reorder_channels(TUH_pick)

        if k == 0 and plot:
            # Plot the energy voltage potential against frequency.
            proc_subject["rawData"].plot_psd(tmax=np.inf, fmax=128, average=True)

            raw_anno = annotate_TUH(proc_subject["rawData"], annoPath=self.EEG_dict[k]["csvpath"])
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