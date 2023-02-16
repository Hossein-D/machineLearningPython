import numpy as np
from numpy import *
from joblib import Parallel, delayed
from multiprocessing import Pool
import numpy as np
from time import clock     
#import qiskit fg
import matplotlib
import numpy as np
from random import randrange

import multiprocessing

import matplotlib.pyplot as plt
import sympy
from sympy import *
import itertools
from IPython.display import display
init_printing()
import math
from tempfile import TemporaryFile
#qiskit.__qiskit_version__
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import csv
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


        
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os, fnmatch
import sys, getopt

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

def learning(L, p, T, PT, intScramble, c1, c2, Nc, NofNTR, NTotal, Nbatch, cvec, 
             delNNT, nnt1, TequalsPT, lightCone, deltaLDim, Ncircuit, TFile, middleInd, new=[]):
             
######  Analyzing the data files:
    
    newLabel = ""
    if new==1:
        newLabel = "New"
        
    DecimalP = (str(p).replace('0.',''))
    Prob = "0p{}".format(DecimalP)
    
    NpureT = 10;
    circIndArr = np.zeros((Nc, NpureT)) 

    ScrambleLabel = ""
    if bool(intScramble):
        ScrambleLabel = "Scrambled"

    ScrambleNewLabel=ScrambleLabel+newLabel;        
    
    if not(bool(intScramble)):        
        if p==0.3: #and (L==2 or L==4 or L==8 or L==16 or L==32 or L==64 or L==128) :
            try:
                circIndArr = globals()["circIndArrL{}".format(L)][:, PT]
            except KeyError:
                try:
                    circIndArr = globals()["circIndArrL{}P{}".format(L, Prob)][:, PT]
                except KeyError:                
                    with open("circIndArrL{}-Ncirc{}-p{}-T{}{}.csv".
                    format(L, Ncircuit, p, T, ScrambleLabel), newline='') as csvfile:
                    #spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
                        csv_reader = csv.reader(csvfile, delimiter=',')
                        rowc = 0                
                        for row in csv_reader: 
                            #print(row)
                            if rowc==0:                        
                                rowc += 1
                                continue
                            for i in range(Nc):
                                circIndArr[i, rowc-1] = int(float(row[i]))                        
                            rowc+=1       
                    
    elif bool(intScramble): # For scrambled samples we don't have circIndArr in this file.         
        with open("circIndArrL{}-Ncirc{}-p{}-T{}{}.csv".
            format(L, Ncircuit, p, T, ScrambleLabel), newline='') as csvfile:
            #spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
            csv_reader = csv.reader(csvfile, delimiter=',')
            rowc = 0                
            for row in csv_reader: 
                if rowc==0:                        
                    rowc += 1
                    continue
                for i in range(Nc):
                    circIndArr[i, rowc-1] = int(float(row[i]))                        
                rowc+=1       
                


    
    nnt2 = nnt1 + delNNT*(NofNTR)
    NSampVec = np.arange(nnt1, nnt2, delNNT)
    print("NSampVec = ", NSampVec)
    fileNameArr = []
    NRepLearn = 1

    scoresArr = np.zeros((Nc, NofNTR))
    print("middleInd = ", middleInd)
    cwdir = os.getcwd()
    print("cwdir = ", cwdir)
        
    if cvec==False:
        CircIndArr = circIndArr[c1:c2, PT]
    else:
        CircIndArr = cvec
    
    measureLabelNew = "measure{}".format(newLabel)
    measureLabel = "measure"
    print("CircIndArr = ", CircIndArr)
    for c in range(len(CircIndArr)):
        CrcInd = CircIndArr[c]
        print("c = ", c)
        for ntr in range(NofNTR):
            NSamples = NSampVec[ntr]
            if NSamples > 2000:
                epoch = 400
            else:
                epoch = 200
            print("NSamples = ", NSamples)
            NTest = min(abs(NTotal-NSamples), 400)
            print("NTest = ", NTest)
            
            NRandUs = 100
            shiftInData = 0
            
            disentanglTime = np.zeros((NTotal, 1))
            Ncircuit=1        
            if Nbatch==False or Nbatch==1:
                try:
                    fileNameArr.append("{}L{}T{}N{}P{}CrcInd{}{}.txt".format(measureLabel, L, TFile, NTotal, Prob, CrcInd, ScrambleNewLabel))    
                    file = open(fileNameArr[-1], "r")
                    print(fileNameArr[-1])
                    print("try")                    
                except FileNotFoundError:
                    try:                   
                        fileNameArr.append("{}L{}T{}N{}P{}CrcInd{}{}.txt".format(measureLabel, L, TFile, NTotal, Prob, CrcInd, ScrambleNewLabel))                    
                        file = open(fileNameArr[-1], "r")
                        print(fileNameArr[-1])                        
                    except FileNotFoundError:
                        try:
                            fileNameArr.append("{}L{}N{}P{}CrcInd{}{}.txt".format(measureLabel, L, NTotal, Prob, CrcInd, ScrambleNewLabel))
                            file = open(fileNameArr[-1], "r")
                            print(fileNameArr[-1])                            
                            print("except")
                        except FileNotFoundError:
                            fileNameArr.append("{}L{}N{}P{}CrcInd{}{}.txt".format(measureLabel, L, NTotal, Prob, CrcInd, ScrambleNewLabel))
                            file = open(fileNameArr[-1], "r")
                            print(fileNameArr[-1])                        
            else:
                fileNameArr.append("{}L{}T{}N{}P{}CrcInd{}Nbatch{}{}.txt".format(measureLabel, L, TFile, NTotal, Prob, CrcInd, Nbatch, ScrambleNewLabel))
                print(fileNameArr[-1])                
            print("fileNameArr[-1] = ", fileNameArr[-1])
            file = open(fileNameArr[-1], "r")
            n = 0
    
            
            ancillaState = np.zeros(NSamples)
            ancillaIndex = -1
            counter = 0
            for x in file:
                #print("x = ", x)
                
                if counter == 0:                    
                    realT = int((len(x[0:-1:2])-4)/(2*L)) ## This is the real length of the time evolution obtained from inputs                    
                    #realT=T                    
                    counter += 1
                    print("x[:20] = ", x[:20])
                else:
                    break
            #realT = 2        
            print("realT = ", realT)
            if lightCone:
                    #print("lght")
                if L-1<realT+1:
                    LDimOfArrays = 2*L
                else:
                    LDimOfArrays = 2*(realT+1)+2*deltaLDim
            else:
                LDimOfArrays = 2*L
            refInd = 2*L+1;
            #middleInd = 2*L; #In the new version of the time evolution code for ancillas, reference qubit is entangled to the middle qubit which is at the "end" of the chain
            #middleInd = L; #In the old version of the time evolution code for ancillas, reference qubit is entangled to the middle qubit which is at the "middle" of the chain
            
            
            halfLDim = int(float(LDimOfArrays/2))
            print("halfLDim = ", halfLDim, " ,LDimOfArrays = ", LDimOfArrays)
            print("L1, L2 = ", middleInd-halfLDim, middleInd+halfLDim)
            measureArr = np.zeros((NSamples, realT, LDimOfArrays))
            testMeasureArr = np.zeros((NTest, realT, LDimOfArrays))                
            testAncillaState = np.zeros(NTest)
            
            convMeasure = np.zeros((NSamples, realT, LDimOfArrays, 1))
            testConvMeasure = np.zeros((NTest, realT, LDimOfArrays, 1))            
            convAncilla = np.zeros((NSamples, 1))
            testConvAncilla = np.zeros((NSamples, 1))
            
            file = open(fileNameArr[-1], "r")    
            print("file = ", file)    

            for x in file:    
                
                #print("x in train = ", x)
                #print("x[-7:-1] = ", x[-7:-1])
                commaStr = x[-4]
                if commaStr == " ":
                    commaInd = -4
                #disentanglTime[i] = int(x[-3:-1])
                elif x[-3] != " " and x[-2] != " ":
                    commaInd = -4
                else:
                    commaInd = -3
                #disentanglTime[i] = int(x[-2])
                #print("x[-4], x[-3], x[-2] = ", x[-4], x[-3], x[-2])                 
                if n==0:
                    ancillaZ = int(x[commaInd - 1])
                    ancillaY = int(x[commaInd - 3])
                    ancillaX = int(x[commaInd - 5])
                    
                    print("ancillaInd, ancillaInd, ancillaInd = ", commaInd - 1, commaInd - 3, commaInd - 5)                    
                    print("ancillaX, ancillaY, ancillaZ = ", ancillaX, ancillaY, ancillaZ)
                    
                    for t in range(realT):
                        if not(lightCone): #L-1<realT+1:
                            for l in range(2*L):                            
                                measureArr[n-shiftInData, t, l] = int(x[4*L*t+2*l])
                                if measureArr[n-shiftInData, t, l] == 2:
                                    measureArr[n-shiftInData, t, l] = -1
                                    
                        elif lightCone and L-1 >= realT+1:      
                            #for l in range(L-1-realT-1-deltaLDim, L-1+realT+1+deltaLDim):                                                                
                            for l in range(middleInd-halfLDim, middleInd+halfLDim):                                                                                            
                                #print("l = ", l)
                                #l = l%(2*L)
                                #print("l new = ", l)
                                measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] = int(x[4*L*t+2*(l%(2*L))])
                                if measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] == 2:
                                    measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] = -1
                    if ancillaZ!=0:
                        ancillaIndex = commaInd - 1
                    elif ancillaY!=0:
                        ancillaIndex = commaInd - 3  
                    elif ancillaX!=0:
                        ancillaIndex = commaInd - 5
                    elif ancillaX==0 and ancillaY==0 and ancillaZ==0:                        
                        # In the new versions after 12/10/2021, although the qubit has not purified,
                        # we use the measurement results along the Z axis
                        ancillaIndex = commaInd - 1;
                        print("break")
                        #break
                    else:
                        break
                        
                    ancillaState[n-shiftInData] = x[ancillaIndex]
                    
                    if ancillaX==0 and ancillaY==0 and ancillaZ==0:
                        #print("ancillaX==0 and ancillaY==0 and ancillaZ==0")
                        randNum = randrange(1, 3);
                        ancillaState[n-shiftInData] = randNum;
                        #print("ancillaState[n-shiftInData] = ", ancillaState[n-shiftInData])
                        
                    if ancillaState[n-shiftInData] == 2:
                        ancillaState[n-shiftInData] = 0
                        
                    n+=1
                    continue
                if n < shiftInData:
                    continue
                elif n < NSamples+shiftInData and n >= shiftInData:
                    #print("n in train = ", n)
                    for t in range(realT):
                        if not(lightCone): # or L-1<realT+1:
                            for l in range(2*L):
                                measureArr[n-shiftInData, t, l] = int(x[4*L*t+2*l])
                                if measureArr[n-shiftInData, t, l] == 2:
                                    measureArr[n-shiftInData, t, l] = -1
                                    
                        elif lightCone and L-1 >= realT+1:
                        #halfLDim
                            #for l in range(L-1-realT-1, L-1+realT+1):   middleInd-halfLDim                             
                            for l in range(middleInd-halfLDim, middleInd+halfLDim):  
                                measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] = int(x[4*L*t+2*(l%(2*L))])
                                #if measureArr[n-shiftInData, t, l-(L-1-realT-1)] == 2:
                                if measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] == 2:
                                    measureArr[n-shiftInData, t, l-(middleInd-halfLDim)] = -1
                                                            
                                    
                    ancillaState[n-shiftInData] = x[ancillaIndex]   # For L=16 x[-2] is non-zero. 

                    if ancillaX==0 and ancillaY==0 and ancillaZ==0:
                        randNum = randrange(1, 3);
                        ancillaState[n-shiftInData] = randNum;
                    
                    if ancillaState[n-shiftInData] == 2:
                        ancillaState[n-shiftInData] = 0
                else:
                    break
                n+=1
                counter += 1                
            print("AncillaState[:10] = ", ancillaState[:10])                    
            file = open(fileNameArr[-1], "r")        
            n = 0 
            print("NSamples+shiftInData = ", NSamples+shiftInData)
            for x in file:

                n+=1
                
                if n < NSamples+shiftInData:
                    continue
                
                elif n < NTest+NSamples+shiftInData and n >= NSamples+shiftInData:
                    
                    for t in range(realT):
                        if not(lightCone): #L-1<realT+1:
                            for l in range(2*L):                         
                                testMeasureArr[n-NSamples-shiftInData, t, l] = int(x[4*L*t+2*l])
                                if testMeasureArr[n-NSamples-shiftInData, t, l] == 2:
                                    testMeasureArr[n-NSamples-shiftInData, t, l] = -1
                                    
                        elif lightCone and L-1 >= realT+1:
                            
                            for l in range(middleInd-halfLDim, middleInd+halfLDim):                                
                                testMeasureArr[n-NSamples-shiftInData, t, l-(middleInd-halfLDim)] = int(x[4*L*t+2*(l%(2*L))])

                                if testMeasureArr[n-NSamples-shiftInData, t, l-(middleInd-halfLDim)] == 2:
                                    testMeasureArr[n-NSamples-shiftInData, t, l-(middleInd-halfLDim)] = -1

                    
                    testAncillaState[n-NSamples-shiftInData] = x[ancillaIndex]
                    if ancillaX==0 and ancillaY==0 and ancillaZ==0:
                        #print("in test ancillaX==0 and ancillaY==0 and ancillaZ==0")                        
                        randNum = randrange(1, 3);
                        testAncillaState[n-NSamples-shiftInData] = randNum;
                    if testAncillaState[n-NSamples-shiftInData] == 2:
                        testAncillaState[n-NSamples-shiftInData] = 0
                else:
                    break
                
            print("testAncillaState[:10] = ", testAncillaState[:10])        
            nnn = 512*(1 + 2*int(NSamples//2000))
            print("nnn = ", nnn)
            
            convMeasure[:, :, :, 0] = np.copy(measureArr)
            convAncilla = np.copy(ancillaState)
            testConvMeasure[:, :, :, 0] = np.copy(testMeasureArr)
            testConvAncilla = np.copy(testAncillaState)            
            
######  Learning Algorithm
    
    
            for nl in range(NRepLearn):
                model = Sequential()
                model.add(Conv2D(int(float(LDimOfArrays/2)), (4, 4), activation='relu', kernel_initializer='he_uniform', padding='same', 
                input_shape=(realT, LDimOfArrays, 1)))
                model.add(Conv2D(LDimOfArrays, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
                if realT>1:
                    model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.2))
                model.add(Flatten())
                model.add(Dense(nnn, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1, activation='sigmoid'))
        
       
                if NSamples > 20000:
                    n_epoch = 800
                elif NSamples <= 20000 and NSamples > 1000:
                    n_epoch = 400
                else:
                    n_epoch = 200

                lrate = 0.01
                decay = .9

                sgd = SGD(lr=lrate, momentum=1, decay=decay, nesterov=False)
                optimizer = tf.keras.optimizers.Adam(0.001)    
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
                AccHist = []
                valAccHist = []
                histLen = 40
                histLen1 = 10
                for epoch in range(n_epoch):
                    history = model.fit(convMeasure, convAncilla, epochs = 1, batch_size = 100, validation_split=0.1, 
                    verbose=0)
                    print("acc = ", epoch, history.history['val_accuracy'][-1], history.history['accuracy'][-1])
                    if epoch > histLen1:
                        if (history.history['val_accuracy'][-1]<.55 and history.history['accuracy'][-1]>.8) or \
                        (history.history['val_accuracy'][-1] > .98 and history.history['accuracy'][-1] > .98):
                            break

                    AccHist.append(history.history['accuracy'][-1])                
                    valAccHist.append(history.history['val_accuracy'][-1])                                                
                    if epoch > histLen:
                        if np.average(AccHist[-histLen:-1])-AccHist[-histLen]<0.001 :
                            print(np.average(AccHist[-histLen:-1]), AccHist[-histLen])
                            print("no increase")
                            break
                        elif np.average(valAccHist[-histLen:-1])-valAccHist[-histLen]<0.001:
                            print(np.average(valAccHist[-histLen:-1]), valAccHist[-histLen])
                            print("no increase")
                            break
                sys.stdout.flush()           
                sys.stdout.flush()
                
                scores = model.evaluate(testConvMeasure, testConvAncilla, verbose=0)
                print("scores[1] = ", scores[1], "NRepLearn = ", NRepLearn)
                #scoresArr[i-2, 0] = scores[0]
                print("c = ", c," ntr = ", ntr, " nl = ", nl)
                scoresArr[c, ntr] += scores[1]/NRepLearn
                if scores[1]>.96:
                    break
            
            if scoresArr[c, ntr]>.96:
                print("break")
                break
    print(scoresArr)    
    df = pd.DataFrame(scoresArr)
    if TequalsPT == 0:
        df.to_csv("accuracy-L{}-p{}-c1_{}-c2_{}-Nc{}-PT{}-nti{}-ntf{}-delNNT{}-NLrn{}.csv".format(L, p, c1, c2, Nc, PT, nnt1, nnt2, delNNT, NRepLearn))
    if TequalsPT == 1:
        df.to_csv("accuracy-L{}-p{}-TeqPT-c1_{}-c2_{}-Nc{}-PT{}-nti{}-ntf{}-delNNT{}-NLrn{}.csv".format(L, p, c1, c2, Nc, PT, nnt1, nnt2, delNNT, NRepLearn))    
    
    #print(x)
    return x

