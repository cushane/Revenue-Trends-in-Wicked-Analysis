#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Wicked Broadway Gross (Raw Data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, detrend

# The raw data for Broadway sales 
wicked_broad = pd.read_csv('broadway.csv')


week = wicked_broad.Week
movie_gross = wicked_broad.Gross


movie = movie_gross.str.strip()
movie_gross = movie.str.replace('$','',regex=False)
movie_gross = movie_gross.str.replace(',','',regex=False).astype(float)


week = pd.to_datetime(week)



#coef1 = np.polyfit(week ,movie_gross, 3)
#p1 = np.poly1d(coef1)


movie_gross.dropna(inplace=True)
week.dropna(inplace=True)
mean2 = detrend(movie_gross, type = 'linear')

plt.plot(week[1017:],mean2[1017:], label = 'Weekly Gross Earnings', color ='red')
plt.legend()
plt.xlabel("Months")
plt.xticks(rotation = 30)
plt.ylabel('Broadway Gross')
plt.grid()
plt.savefig('brawdata1', bbox_inches = 'tight')
plt.show()

plt.plot(week[978:],mean2[978:], label = 'Weekly Gross Earnings', color ='red')
plt.legend()
plt.xlabel("Months")
plt.xticks(rotation = 70)
plt.ylabel('Broadway Gross')
plt.grid()
plt.savefig('mrawdata2',bbox_inches = 'tight')
plt.show()


plt.plot(week,mean2, label = 'Weekly Gross Earnings', color ='red')
plt.legend()
plt.xlabel("Years")
plt.xticks(rotation = 70)
plt.ylabel('Broadway Gross')
plt.grid()
plt.savefig('brawdata',bbox_inches = 'tight')
plt.show()



# In[3]:


import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt


def lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def filters(data, cutoff, fs, order=5):
    b, a = lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


order = 2
fs = 52     
cutoff = 4

b, a = lowpass(cutoff, fs, order)
'''
'''
# Plot the frequency response.
w, h = freqz(b, a, fs=fs, worN=8000)

plt.plot(w, np.abs(h), label = 'Frequency Response', color = 'red')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.xlabel('Frequency $Hz$')
plt.ylabel('Frequency Response')
plt.grid()
plt.legend()
plt.savefig('freqres',bbox_inches = 'tight')
plt.show()


# Filter the data, and plot both the original and filtered signals.
y = filters(mean2, cutoff, fs, order)
#978-1044

plt.plot(week, mean2,  label=' Raw data', color = 'red')
plt.plot(week, y , linewidth=2, label='Filtered data')
plt.xlabel('Years')
plt.ylabel('Broadway Gross')
plt.grid()
plt.legend()
plt.savefig('filtb',bbox_inches = 'tight')
plt.show()

plt.plot(week[1015:], mean2[1015:],  label=' Raw data', color = 'red')
plt.plot(week[1015:], y[1015:] , linewidth=2, label='Filtered data')
plt.xlabel('Months')
plt.ylabel('Broadway Gross')
plt.grid()
plt.xticks(rotation = 70)
plt.legend()
plt.savefig('filtbro',bbox_inches = 'tight')
plt.show()


# In[4]:


# The raw data for movie sales

wicked_movie = pd.read_csv('Movie wicked 2.csv')


week = wicked_movie.Date
movie_gross = wicked_movie.Weekly


movie = movie_gross.str.strip()
movie_gross = movie.str.replace('$','',regex=False)
movie_gross = movie_gross.str.replace(',','',regex=False).astype(float)


week = pd.to_datetime(week)

plt.plot(week,movie_gross, label = 'Weekly Gross Earnings', color = 'red')
plt.legend()
plt.xlabel("Weeks")
plt.xticks(rotation = 70)
plt.ylabel('Broadway Gross')
plt.grid()
plt.savefig('mrawdata', bbox_inches = 'tight')
plt.show()


# In[5]:


import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt


def lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def filters(data, cutoff, fs, order=5):
    b, a = lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


order = 2
fs = 52     
cutoff = 4

b, a = lowpass(cutoff, fs, order)
'''
'''
# Plot the frequency response.
w, h = freqz(b, a, fs=fs, worN=8000)

plt.plot(w, np.abs(h), label = 'Frequency Response', color = 'red')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.xlabel('Frequency $Hz$')
plt.ylabel('Frequency Response')
plt.grid()
plt.legend()
plt.savefig('freqres',bbox_inches = 'tight')
plt.show()


# Filter the data, and plot both the original and filtered signals.
y = filters(movie_gross, cutoff, fs, order)
#978-1044

plt.plot(week, movie_gross,  label=' Raw data', color = 'red')
plt.plot(week, y , linewidth=2, label='Filtered data')
plt.xlabel('Years')
plt.ylabel('Broadway Gross')
plt.grid()
plt.legend()
plt.xticks(rotation = 70)
plt.savefig('movieb',bbox_inches = 'tight')
plt.show()


# In[6]:


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

avg = mean2.mean()
std = mean2.std()
normalized = (mean2- avg) / std

avg = movie_gross.mean()
std = movie_gross.std()
normalized2 = (movie_gross- avg) / std

corr = signal.correlate(mean2[1016:], movie_gross)
lags = signal.correlation_lags(len(mean2[1016:]), len(movie_gross))
corr /= np.max(corr)



plt.stem(lags,corr,label = "Cross Correlation", )
plt.xlabel('Lags')
plt.ylabel('Cross-Correlation')
plt.grid()
plt.legend()
plt.savefig('corr',bbox_inches = 'tight')
plt.show()


# In[ ]:




