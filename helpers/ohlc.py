import scipy.io
import numpy as np
def load_pieces(date_str, contract_str,time_start=120,time_end=900):
    temp_ohlc= get_ohlc(date_str,contract_str,'1 min',fill_in=True)
    if len(temp_ohlc) < int(time_end-time_start+1):
            raise ValueError('Unexpected Num of Bars')
    myinds = (temp_ohlc[:,0]>time_start-0.5)&(temp_ohlc[:,0]<time_end+0.5)
    times = np.round(temp_ohlc[myinds,0]).astype(np.int32)
    closes = np.array(temp_ohlc[myinds,-1],dtype=np.float32)
    primemap = np.array(get_primemap_16(date_str,contract_str,time_start,time_end),dtype=np.float32)

    return times,closes,primemap
def load_ohlc(date_str, contract_str,time_start=120,time_end=900):
    temp_ohlc=get_ohlc(date_str,contract_str,'1 min',fill_in=True)
    if len(temp_ohlc) < int(time_end-time_start+1):
            raise ValueError('Unexpected Num of Bars')
    myinds = (temp_ohlc[:,0]>time_start-0.5)&(temp_ohlc[:,0]<time_end+0.5)
    return temp_ohlc[myinds]
def get_ohlc_piece(d_str,c_str,startTime,endTime):
    ohlc = get_ohlc(d_str,c_str,'1 min',True)
    myinds = np.logical_and(ohlc[:,0]>startTime-0.5,ohlc[:,0]<endTime+0.5)
    return ohlc[myinds]
def get_ohlc(date_str,contract_str,time_type_str,fill_in=False): #Note OHLC is a matrix in minutes or seconds (NOT clock)
    basedir = 'C:/Users/ericy/Desktop/CL Data/new_python/'
    #example '5 min/CL 01-18/2017-11-20.mat' (must be exact match)
    t_str = basedir+time_type_str+'/'+contract_str+'/'+date_str+'.npy'
    myout = np.load(t_str)
    myout=quick_fix(myout)#TODO remove quick fix. and fix create_ohlc and run file
    if time_type_str=='1 sec':
        myout = fill_in_sec(myout)
    else:
        if fill_in:#2018-03-15 has known holes in minute data
            myout = fill_in_sec(myout)
    return myout
def quick_fix(ohlc):
    while ohlc[-1,0]<0:
        ohlc=ohlc[:-1]
    return ohlc
def fill_in_sec(the_ohlc):
    firstTime = int(the_ohlc[0,0])
    lastTime = int(the_ohlc[-1,0])
    
    theTimes = np.arange(firstTime,lastTime+1)
    newOHLC = np.zeros((len(theTimes),5))
    newOHLC[:,0]=theTimes
    oldind = 0
    oldrow = the_ohlc[oldind,:]
    for i in range(len(theTimes)):
        newind = oldind+1
        newrow = the_ohlc[newind,:]
        if abs(newrow[0]-(newOHLC[i,0]))<.1:
            newOHLC[i,:]=newrow
            oldind = newind;
            oldrow = newrow;
        else:
            newOHLC[i,1::]= oldrow[1::]

    return newOHLC

def clock_to_second(innum):
    seconds = innum % 100
    minutes = (innum % 10000)//100
    hours = innum//10000
    return hours * 3600 + minutes * 60 + seconds
def clock_to_minute(innum):
    minutes = innum % 100
    hours = innum//100
    return hours * 60 + minutes


#======================================================================================================
#PRIMEMAP
def normalize_16_pm(mat):
    myMeans = [4.35629356e-05, 2.91196094e-05, 6.28983563e-05, 1.60274330e-04, 
                5.04817548e-04, 2.17721495e-03, 1.37202281e-02, 1.65170851e-01,
                6.36734737e-01, 1.64625033e-01, 1.37922118e-02, 2.14786239e-03,
                5.34636028e-04, 1.63535726e-04, 6.42960975e-05, 6.87222781e-05]
    myStds = [0.00096634, 0.00071258, 0.00111287, 0.00171029, 0.00313377, 0.00688838,
                0.01915968, 0.05931542, 0.12920062, 0.05873829, 0.0190645,  0.00682061,
                0.00324098, 0.00171904, 0.00106282, 0.00121111]
    myMeans = np.array(myMeans)
    myStds = np.array(myStds)
    return (mat - myMeans)/myStds
def get_primemap_16(date_str,contract_str,startTime,endTime):
    raw = get_primemap_raw(date_str,contract_str,startTime,endTime)
    return normalize_16_pm(raw)
def get_primemap_raw(date_str,contract_str,startTime,endTime,do_normalize=False):
    # Params
    category_vals = np.arange(-0.08,0.075,0.01)#16 instead of 17 (even number used for vq training)
    #
    my_ohlc_min = get_ohlc(date_str,contract_str,'1 min',fill_in=True)
    my_ohlc_sec = get_ohlc(date_str,contract_str,'1 sec')
    tempinds= (my_ohlc_min[:,0]>(startTime-0.5))&(my_ohlc_min[:,0]<(endTime+0.5))
    temp_ohlc = my_ohlc_min[tempinds]
    startMin = int(temp_ohlc[0,0])
    endMin = int(temp_ohlc[-1,0])
    wholeMat = np.zeros((endMin-startMin +1, len(category_vals)))
    for i in np.arange(startMin,endMin+1):
        lastSec = i * 60
        firstSec = lastSec-60 #Includes 0 - 60 (delta includes 60 points 1-60)

        tempInds = np.where((my_ohlc_sec[:,0]<(lastSec+0.1))&(my_ohlc_sec[:,0]>(firstSec-0.1)))
        miniOHLC_sec = my_ohlc_sec[tempInds]
        atempMapPrime = miniOHLC_sec[1:,-1] - miniOHLC_sec[:-1,-1]
        wholeMat[i-startMin,:]= convert_to_bar(atempMapPrime,category_vals)
    wholeMat = wholeMat / 60
    if do_normalize:
        return log_and_normalise_images(wholeMat)
    return wholeMat
def convert_to_bar(theTempMapPrime,the_cat_vals):
    outMat = np.zeros(len(the_cat_vals))
    theMax = the_cat_vals[-1]
    theMin = the_cat_vals[0]
    for ctbi in theTempMapPrime:
        if ctbi > theMax:
            outMat[-1] +=1
        elif ctbi < theMin:
            outMat[0] +=1
        else:
            someInd = (abs(the_cat_vals - ctbi) < 0.001)
            outMat[someInd] +=1
            #print(ctbi,outMat)
    return outMat