import mne


def preprocessRaw(MNE_raw=None, lpfq=1, hpfq=40, notchfq=50, downSam=100, cap_setup="easycap-M1"):
    # Follows Makoto's_Preprocessing_Pipeline recommended for EEGlab's ICA
    # Step 4: HP-filter [1Hz] -> BP-filter [1Hz; 40Hz] for this study
    MNE_raw.filter(lpfq, hpfq, fir_design='firwin')

    # Step 5: Import channel info -> configure cap setup and channel names
    MNE_raw.set_montage(mne.channels.make_standard_montage(kind=cap_setup, head_size=0.095), on_missing="warn")

    # Step 6 utilizing data knowledge for line-noise removal
    MNE_raw.notch_filter(freqs=notchfq, notch_widths=5)

    # Step 7: Downsample
    MNE_raw.resample(sfreq=downSam)

    # Step 8
    # MNE_raw.interpolate_bads(reset_bads=True, origin='auto')

    # Step 9 Re-reference the data to average
    MNE_raw.set_eeg_reference()

    # Step 10 through 15 is conducted in separate ICLabel or ANN

    return MNE_raw