import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack


#LUT for 4 PAM
#00 -3
#01 -1
#11  1
#10  3


num_symbols = 20
sps = 8  #samples per symbol

# 40 bits to be grouped in 2 bits and mapped into symbols using LUT
bits = np.random.randint(0, 2, 2*num_symbols) # Our data to be transmitted, 1's and 0's
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
        symbollist.append(LUT(str(bits[i]) + str(bits[i+1])))
    return symbollist

symbollist = symbolGen(bits)

def pad(symbollist, sps):
    n = len(symbollist)
    x = np.array([])
    for j in range(0,n):
        pulse = np.zeros(sps)
        pulse[0] = symbollist[j]
        x = np.concatenate((x, pulse))
    return x

paddedSymbolList = pad(symbollist, sps)
paddedOnes1 = pad(sampling, sps) # This can be used for sampling and Ts instances

# To get correct sampling train
# To get correct location of sampling i.e getting correct ones, just where ever there is non zero value in paddedSymbolList replace it with one
nx = len(paddedSymbolList)
paddedOnes = np.zeros(nx)
for i in range(0,nx):
    if(paddedSymbolList[i] != 0):
        paddedOnes[i+1] = 1; # Delay in impulse train for sampling signal after match filter as delay of one sample was observed due to match filtering

nlength = len(paddedSymbolList)
time = np.linspace(0, 20*0.1*(10**-6), nlength)

plt.figure("Symbols Transmitted")
plt.plot(time, paddedSymbolList, '.-')
plt.ylabel("Symbol Value")
plt.xlabel("time(s)")
plt.grid(True)
plt.show()

#Now we have generated symbols with padding

#Now we need to generate Pulse shaping filter
def generate_square_pulse(total_time, num_samples):
    t = np.arange(0, total_time*2, 1/(10*(10**6)*8))
    signal = np.zeros_like(t)
    signal[:num_samples] = 1
    return t, signal

total_time = 0.1 * 10**-6  # seconds
num_samples = 8

t, square_pulse = generate_square_pulse(total_time, num_samples)

plt.stem(t, square_pulse, basefmt='b-', linefmt='r-', markerfmt='ro')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Single Square Pulse')
plt.show()

#Generating Transmitting Signal
pulse_shaped = np.convolve(paddedSymbolList, square_pulse, mode='same')

sampled = np.multiply(pulse_shaped, paddedOnes1)

plt.figure("Transmited Signal and instances where symbols were sent")
plt.plot(time, sampled, '.-')
plt.plot(time, pulse_shaped, '.-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# Spectrum of Modulated Signal (signal after pulse shaping filter)
sig_fft  = fftpack.fft(pulse_shaped)
Amplitude = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(pulse_shaped.size, d = (1/(10*(10**6)*sps)))
#print(sample_freq)
plt.figure("Spectrum of Modulated Signal")
plt.plot(sample_freq/1e6, 20*np.log10(Amplitude))
#plt.plot(Amplitude)
plt.xlabel("f in MHz")
plt.ylabel("PSD in dB")
plt.show()

#############################################################PASSBAND IMPLEMENTATION############################################################################################
##def findbandwidth(sig_fft, sample_freq):
##    positivefreq = np.where(sample_freq > 0)
##    maxIndex = np.argmax(np.abs(sig_fft[positivefreq]))
##    bandwidth = sample_freq[positivefreq][maxIndex]
##    return bandwidth

Fs = 8*10**7
fc = Fs/4

x_bb =  pulse_shaped
x_pb = x_bb * np.cos(2 * np.pi * fc * time)

x_rx = x_pb * np.cos(2 * np.pi * fc * time)

plt.figure("Signal before low Pass Filter")
plt.plot(time, x_rx, '.-')
plt.xlabel("time(s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


#f_BW = findbandwidth(sig_fft, sample_freq)
f_BW = max(sample_freq)
numtaps = 25
f = f_BW / Fs
h = signal.firwin(numtaps, f)
signalp = x_rx

##########      REMOVING THE GROUP DELAY DUE TO FIR FILTER       #########################################
filtered_signal_forward = signal.lfilter(2*h, 1.0, signalp)
    
filtered_signal_reverse = signal.lfilter(2*h, 1.0, filtered_signal_forward[::-1])[::-1]

x_shaped =  0.5*(filtered_signal_forward + filtered_signal_reverse)
#############################################################################################
##pulse_shaped = signal.lfilter(2*h, 1.0, x_rx)
##pulse_shaped = np.roll(pulse_shaped, 12)
sig_fft1  = fftpack.fft(2*h)
Amplitude2 = np.abs(sig_fft1)
sample_freq1 = fftpack.fftfreq(h.size, d = (1/(10*(10**6)*sps)))
plt.figure("Spectrum of FIR Filter")
plt.plot(Amplitude2)
plt.ylabel("PSD in dB")

plt.figure("Signal After low Pass Filter")
plt.plot(time, pulse_shaped, '.-')
plt.xlabel("time(s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
#########################################################################################################################################################################

Ts = 0.1 * (10 ** -6)
#Fs = 8*10**7
#fc = fs/4
matchedFilter = square_pulse 

#plotting the Match Filter
plt.figure("Matched Filter")
plt.stem(t, matchedFilter, '.')
plt.xlabel("time(s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

rt = np.convolve(matchedFilter, pulse_shaped, mode = 'same')

sampled2 = np.multiply(rt/8, paddedOnes)
# Here received symbols are symbolValueTransmitted*(Energy of the pulse) 
# And here energy of the pulse train is 1^2 + 1^2 + .... = 8

plt.figure("MatchFilter response to transmitted Signal and instants where we transmitted Symbols")
#plt.plot(time, sampled2, '.-')
#plt.plot(time, rt, '.-')
plt.plot(time, sampled2, '.-')
plt.plot(time, rt/8, '.-')
plt.xlabel("time(s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

symbolsRecv = []
for i in sampled2:
    if i != 0:
        symbolsRecv.append(i)

# Constellation Diagram
plt.figure(3)
plt.scatter(np.real(symbolsRecv), np.imag(symbolsRecv), c='red', marker='x', label='Received')
plt.scatter(np.real(symbollist), np.imag(symbollist), c='blue', marker='o', label='Transmitted')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation Diagram')
plt.legend()
plt.grid(True)
plt.show()
