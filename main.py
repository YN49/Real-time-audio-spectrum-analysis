import pyaudio
import time
import matplotlib.pyplot as plt
import wave
from scipy.fftpack import fft, ifft
from scipy import signal
import numpy as np

N=24*1
CHUNK=1024*N
RATE=88200 #11025 #22050  #44100 #88200
CHANNELS = 1             # 1;monoral 2;ステレオ-
p=pyaudio.PyAudio()
WAVE_OUTPUT_FILENAME = "output.wav"
FORMAT = pyaudio.paInt16 #int16型

stream=p.open(  format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

s=1
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Signal')
ax2 = fig.add_subplot(312)
ax2.set_ylabel('Freq[Hz]')
ax2.set_xlabel('Time [sec]')
ax3 = fig.add_subplot(313)
ax3.set_xlabel('Freq[Hz]')
ax3.set_xscale('log')
ax3.set_ylabel('Power')
start=time.time()
stop_time=time.time()

while stream.is_active():

    fig.delaxes(ax1)
    fig.delaxes(ax3)
    ax1 = fig.add_subplot(311)
    ax1.set_title('passed time; {:.2f}(sec)'.format(time.time()-start))
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')
    ax3 = fig.add_subplot(313)

    start_time=time.time()
    print(start_time-stop_time)
    
    stop_time=time.time()
    input = stream.read(CHUNK)

    
    frames = []
    frames.append(input)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    wavfile = WAVE_OUTPUT_FILENAME
    wr = wave.open(wavfile, "rb")
    ch = CHANNELS #wr.getnchannels()
    width = p.get_sample_size(FORMAT) #wr.getsampwidth()
    fr = RATE  #wr.getframerate()
    fn = wr.getnframes()
    fs = fn / fr
    print("fn,fs",fn,fs,stop_time-start_time)

    origin = wr.readframes(wr.getnframes())
    data = origin[:fn]
    wr.close()


    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,fs, fn/2, endpoint=False)
    ax1.set_ylim(-0.75,0.75)
    ax1.set_xlim(0,fs)
    ax1.plot(t, sig)


    nperseg = 1024
    f, t, Zxx = signal.stft(sig, fs=fn/2, nperseg=nperseg)
    Zxxout = np.abs(Zxx)
    ax2.pcolormesh(fs*t, f*RATE/N/200, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,fs)
    ax2.set_ylim(200,20000)
    ax2.set_yscale('log')
    #Zxxが出力のスペクトラム

    print(np.abs(Zxx).shape)

    freq =fft(sig,int(fn/2))
    Pyy = np.sqrt(freq*freq.conj())*2/fn
    f = np.arange(200,RATE*100/50,(RATE*100/50-200)/int(fn/2)) #RATE11025,22050;N50,100
    ax3.set_ylim(0,0.0075)
    ax3.set_xlim(200,20000)
    ax3.set_xlabel('Freq[Hz]')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.plot(f,Pyy)

    plt.pause(0.01)
    s += 1


stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")
