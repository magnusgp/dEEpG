import mne

def preprocessRaw(MNE_raw=None, lpfq=1, hpfq=40, notchfq=60, downSam=100, cap_setup="easycap-M1",ICA=False):
    #The raw signal is Band-pass filtered. Default is 1-100 (as to not remove the muscle artifacts of
    MNE_raw.filter(lpfq, hpfq, fir_design='firwin')

    # Channel names are set from the cap_setup
    MNE_raw.set_montage(mne.channels.make_standard_montage(kind=cap_setup, head_size=0.095), on_missing="warn")

    # In america there is a line-noise at around 60 Hz, which i
    MNE_raw.notch_filter(freqs=notchfq, notch_widths=5)

    # Step 7: Downsample
    MNE_raw.resample(sfreq=downSam)

    # Step 8
    #MNE_raw.interpolate_bads(reset_bads=True, origin='auto')

    # Re-reference the raw signal to average of all channels
    MNE_raw.set_eeg_reference()

    if ICA:
        ica=mne.preprocessing.ICA(n_components=20)
        ica.fit(MNE_raw)
        MNE_raw.load_data()
        ica.apply(MNE_raw)

    return MNE_raw