import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import commpy

# LUT
# 00 -3
# 01 -1
# 11  1
# 10  3

num_symbols = 20
sps = 8  # samples per symbol

# 40 bits to be grouped in 2 bits and mapped into symbols using LUT
bits = np.random.randint(0, 2, 2 * num_symbols)  # Our data to be transmitted, 1's and 0's
sampling = np.ones(20)   

def LUT(groupedbits):
    if groupedbits == "00":
        return -3
    elif groupedbits == "01":
        return -1
    elif groupedbits == "11":
        return 1
    else:
        return 3

def symbolGen(bits):
    symbollist = []
    n = len(bits)
    for i in range(0, n, 2):
        symbollist.append(LUT(str(bits[i]) + str(bits[i + 1])))
    return symbollist

symbollist = symbolGen(bits)

def pad(symbollist, sps):
    n = len(symbollist)
    x = np.array([])
    for j in range(0, n):
        pulse = np.zeros(sps)
        pulse[0] = symbollist[j]
        x = np.concatenate((x, pulse))
    return x

paddedOnes = pad(sampling, sps) # This can be used for sampling and Ts instances

paddedSymbolList = pad(symbollist, sps)

nlength = len(paddedSymbolList)
time = np.linspace(0, 20*0.1*(10**-6), nlength) 

plt.figure("Symbols being transmitted")
plt.plot(time, paddedSymbolList, '.-')
plt.xlabel("time")
plt.ylabel("Symbol Value")
plt.grid(True)
plt.show()

# Now we have generated symbols with padding

# Now we need to generate Pulse shaping filter name is h
num_taps = 100
beta = 0.25
t, h = commpy.filters.rrcosfilter(num_taps, beta, 0.1*(10**-6), 8*10**7)
plt.figure("Impulse response of a pulse shaping filter")
plt.plot(t, h, '.')
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

nh = len(h)
energy = 0
for i in h:
    energy = energy + i**2
                   

# Pulse shape is ready now we need to convolve with the symbols
x_shaped = np.convolve(paddedSymbolList, h, mode = "same")

sampled = np.multiply(x_shaped, paddedOnes)

plt.figure("Transmiited signal and instants where we transmitted Symbols")
plt.plot(time, sampled, '.-')
plt.plot(time, x_shaped, '.-')
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


#Obtaing fourier transform of TRANSMITTED SIGNAL
sig_fft = fftpack.fft(x_shaped)
Amplitude = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(x_shaped.size, d=(1 / (10 * (10 ** 6) * sps)))
plt.figure("Spectrum of Modulated Signal")
plt.plot(sample_freq/1e6, 20*np.log10(Amplitude))
plt.xlabel("f in MHz")
plt.ylabel("PSD in dB")
plt.show()
###########################PASS BAND####################################################################
def findbandwidth(sig_fft, sample_freq):
    positivefreq = np.where(sample_freq > 0)
    maxIndex = np.argmax(np.abs(sig_fft[positivefreq]))
    bandwidth = sample_freq[positivefreq][maxIndex]
    return bandwidth

Fs = 8*10**7
fc = Fs/4

x_bb =  x_shaped
x_pb = x_bb * np.cos(2 * np.pi * fc * time)

x_rx = x_pb * np.cos(2 * np.pi * fc * time)

#f_BW = findbandwidth(sig_fft, sample_freq)
f_BW = max(sample_freq)
numtaps = 25
f = f_BW / Fs
h1 = signal.firwin(numtaps, f)

signalp = x_rx

########################## Removing the Group Delay #######################################

filtered_signal_forward = signal.lfilter(2*h1, 1.0, signalp)
    
filtered_signal_reverse = signal.lfilter(2*h1, 1.0, filtered_signal_forward[::-1])[::-1]

x_shaped =  0.5*(filtered_signal_forward + filtered_signal_reverse)

###########################################################################################

sig_fft1  = fftpack.fft(2*h1)
Amplitude2 = np.abs(sig_fft1)
sample_freq1 = fftpack.fftfreq(h1.size, d = (1/(10*(10**6)*sps)))
plt.figure("Spectrum of FIR FILTER")
plt.plot(Amplitude2)
plt.xlabel("f in MHz")
plt.ylabel("PSD in dB")
plt.show()

plt.figure("Signal After low passFilter")
plt.plot(time, x_shaped, '.-')
plt.xlabel("time(s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
#########################################################################################################




def MakeMatchFilter(signal):
    flipped_signal = np.flip(signal)
    T = 8
    shifted_signal = np.concatenate((np.zeros(T), flipped_signal[:-T]))
    return shifted_signal

# now we need to implement match filter
Ts = 0.1 * (10 ** -6)
matchedFilter = MakeMatchFilter(h)

#plotting the Match Filter
plt.figure("Matched Filter")
plt.plot(t, matchedFilter, '.')
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

#Passing the transmiited signal by Match Filter
rt = np.convolve(x_shaped, matchedFilter, mode = "same")

nx = len(paddedSymbolList)
paddedOnes2 = np.zeros(nx)
for i in range(0,nx):
    if(paddedSymbolList[i] != 0):
        paddedOnes2[i+1] = 1;

sampled2 = np.multiply(rt/energy, paddedOnes2)

plt.figure("MatchFilter response to transmitted Signal and instants where we transmitted Symbols")
plt.plot(time, sampled2, '.-')
plt.plot(time, rt/energy, '.-')
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

symbolsRecv = []

for i in sampled2:
    if i != 0:
        symbolsRecv.append(i)
        
plt.figure("Constellation Diagram")
#plt.scatter(symbolsRecv, c='red', marker = 'x', label = 'Received')
plt.scatter(np.real(symbolsRecv), np.imag(symbolsRecv), c='red', marker='x', label='Received')
plt.scatter(np.real(symbollist), np.imag(symbollist), c='blue', marker='o', label='Transmitted')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation Diagram')
plt.legend()
plt.grid(True)
plt.show()


