import os
import numpy as np
from matplotlib import pyplot as plt

def sessionStat(EEG_dict):
    session_lengths = []
    sfreqs = []
    nchans = []
    years = []
    age = []
    gender = []
    for k in range(len(EEG_dict)):
        #Collect data about the files:
        data=EEG_dict[k]["rawData"]
        session_lengths.append(data.n_times / data.info['sfreq'])
        sfreqs.append(data.info['sfreq'])
        nchans.append(data.info['nchan'])
        years.append(data.info['meas_date'].year)

        #Collect data about the patients:
        txtPath=os.path.splitext(EEG_dict[k]["path"])[0][:-5]+'.txt'
        with open(txtPath, "rb") as file:
            s = file.read().decode('latin-1').lower()
            try:
                #Find age:
                if s.find('year') != -1:
                    index = s.find('year')
                    age.append(int("".join(filter(str.isdigit, s[index - 10: index]))))
                elif s.find('yr') != -1:
                    index = s.find('yr')
                    age.append(int("".join(filter(str.isdigit, s[index - 10: index]))))
                elif s.find('yo ') != -1:
                    index = s.find('yo ')
                    age.append(int("".join(filter(str.isdigit, s[index - 10: index]))))
            except:
                pass

            try:
                #Find gender:
                if s.find('female') != -1:
                    gender.append('Female')
                elif s.find('woman') != -1:
                    gender.append('Female')
                elif s.find('girl') != -1:
                    gender.append('Female')
                elif s.find('male') != -1:
                    gender.append('Male')
                elif s.find('man') != -1:
                    gender.append('Male')
                elif s.find('boy') != -1:
                    gender.append('Male')
            except:
                pass

    print("Average session length: {:.3f}".format(np.mean(session_lengths)))
    print("Average patient age: {:.3f}".format(np.mean(age)))

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    ax[0].hist(session_lengths, bins=20, rwidth=0.90, color="#000088")
    ax[0].grid(axis='y')
    ax[0].set_ylabel(r'{Count}', size=18)
    ax[0].set_xlabel(r'{Session length}', size=18)
    # ax[0].set_yscale('log')
    ax[1].hist(years, bins=8, rwidth=0.90, color="#000088")
    ax[1].grid(axis='y')
    ax[1].set_xlabel(r'{Year of recording}', size=18)
    ax[2].hist(age, bins=20, rwidth=0.90, color="#000088")
    ax[2].grid(axis='y')
    ax[2].set_xlabel(r'{Age of patient}', size=18)
    # plt.tight_layout()
    plt.savefig("patient_statistics.png", dpi=1000, bbox_inches='tight')
    plt.show()


