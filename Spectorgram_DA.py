from matplotlib import cm # to get a matplotlib.colors.ListedColormap
from matplotlib import style
style.use('fivethirtyeight')
from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

data = loadmat(file_name = './EEGrestingState.mat')
# transpose matrix and convert to 1D NumPy array
eeg = data['eegdata'].squeeze() # amplitude in microVolts
sr = int(data['srate']) # sampling rate in sec
time = np.arange(eeg.size)/sr
print('Sampling rate = %d samples/sec'%sr)

# plot the time course of the EEG
fig, ax = plt.subplots(2,1, figsize=(16,4), sharey=True)

ax[0].plot(time,eeg, lw=1)
ax[0].set_xlabel('Time (sec)'), ax[0].set_ylabel('Voltage ($\mu$Volts)')
ax[0].set_xticks(range(0,130,60))

ax[1].plot(time, eeg, lw=1, color = "k")
ax[1].set_xlim(42.5,45)
#ax[1].set_xlim(12,14.5)
ax[1].set_xticks(range(43,45,1));
ax[1].set_xlabel('Time (sec)');
#plt.show()

# Fourier transform
FourierCoeff = np.fft.fft(eeg)/eeg.size
DC = [np.abs(FourierCoeff[0])]
amp = np.concatenate((DC, 2*np.abs(FourierCoeff[1:])))

# compute frequencies vector until half the sampling rate
Nyquist = sr/2
print('Nyquist frequency = %2.4f Hz'%Nyquist)
Nsamples = int(math.floor(eeg.size/2) )
hz = np.linspace(0, Nyquist, num = Nsamples + 1 )
dhz = hz[1]
print('Spectral resolution = %2.4f Hz'%hz[1])

# now we will analyze window lenghts of 500 ms in 25 ms steps.
# Signals will overlap 475 ms
WinLength = int(0.5*sr) # 500 points (0.5 sec, 500 ms)
step = int(0.025*sr) # 25 points (or 25 ms)

# we have less resolution here because the signals are smaller
Nsamples = int( np.floor(WinLength/2) )
hz = np.linspace(0,Nyquist, Nsamples + 1)
dfreq = hz[1]
print('Spectral resolution = %2.4f Hz'%dfreq)

nsteps = int(np.floor ( (eeg.size - WinLength)/step) )
print(eeg.size, nsteps)

myamp = list()
for i in range(nsteps):
    # signal duration 500 ms (512 data points)
    data = eeg[i * step:i * step + WinLength]

    FourierCoeff = np.fft.fft(data) / WinLength
    DC = [np.abs(FourierCoeff[0])]  # DC component
    amp = np.concatenate((DC, 2 * np.abs(FourierCoeff[1:])))

    amp = amp[:int(45 / dfreq)]
    myamp.append(amp)

power = np.power(myamp, 2)
# logpower = 10*np.log10(power)
fig, ax = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
# fig.suptitle('Time-frequency power via short-time FFT')

ax[0].plot(time, eeg, lw=1, color='C0')
ax[0].set_ylabel('Amplitude ($\mu V$)')
ax[0].set_title('EEG signal')

# spectrum is a ContourSet object
dt = 120 / nsteps  # 120 seconds in number of steps
X = np.arange(nsteps) * dt
Y = hz[:int(45 / dfreq)]
Z = np.array(myamp).T


# flip spectrum for augmentation
X = X[::-1]
print("length of tuple:", len(X))
#X = tuple([np.zeros(len(X))])
Y = Y[::-1]
Z = Z[::-1]
#Dropout method
Y = np.where(Y > 20, 0, Y)
#Enhance spectogram aplitude or frekvens by multiplying through Y or Z

levels = 45
spectrum = ax[1].contourf(X, Y, Z, levels, cmap='jet')  # ,'linecolor','none')

# get the colormap
cbar = plt.colorbar(spectrum)  # , boundaries=np.linspace(0,1,5))
cbar.ax.set_ylabel('Amplitude ($\mu$V)', rotation=90)
cbar.set_ticks(np.arange(0, 50, 10))

# A working example (for any value range) with five ticks along the bar is:

m0 = int(np.floor(np.min(myamp)))  # colorbar min value
m4 = int(np.ceil(np.max(myamp)))  # colorbar max value
m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
cbar.set_ticks([m0, m1, m2, m3, m4])
cbar.set_ticklabels([m0, m1, m2, m3, m4])

# cbar.set_ticks(np.arange(0, 1.1, 0.5))

ax[1].axhline(y=8, linestyle='--', linewidth=1.5, color='white')
ax[1].axhline(y=12, linestyle='--', linewidth=1.5, color='white')
ax[1].set_ylim([0, 40])
ax[1].set_yticks(range(0, 45, 5))
ax[1].set_ylabel('Frequency (Hz)')
ax[1].invert_xaxis()
plt.show()

# invert_xaxis() - if augmentation used around y-axis

for myax in ax:
    myax.set_xlim(0, 120)
    myax.invert_xaxis()
    myax.set_xticks(np.arange(0, 121, 30))
    myax.set_xlabel('Time (sec.)')





