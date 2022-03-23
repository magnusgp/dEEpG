"""
import tueg_tools
ds = tueg_tools.Dataset('/Users/magnus/Desktop/DTU/Data')
#ds.download('https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/eval/abnormal/01_tcp_ar',
#             username='nedc', password='nedc_resources', maxSize=10**6)

ds = tueg_tools.Dataset('/Users/magnus/Desktop/DTU/EEG2/TUH_data_sample')
print(ds)
for eeg in ds.eeg_gen():
    print("Hello")
"""

