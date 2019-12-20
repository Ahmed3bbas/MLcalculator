# coding: utf-8
import gzip
import numpy as np
import os
#from google_drive_downloader import GoogleDriveDownloader as gdd

#Split Data of operators
from sklearn.model_selection import train_test_split

os.chdir(r"/mnt/d/3 year of comuter engineering/second term/SW2/DigitRecognition-master/src/Calculator/data")

def load_data():
    tipath = "train-images-idx3-ubyte.gz"    #train digiits images
    tlpath = "train-labels-idx1-ubyte.gz"    #train digiits labels
    sipath = "t10k-images-idx3-ubyte.gz"    #test digiits images
    slpath = "t10k-labels-idx1-ubyte.gz"    #test digiits labels

    operator_label_id = "1PvBA8Le18U4XMTCgN5pmi3ATyXnN-OGg"
    operator_label_name = "operator_Labels.npy"

    operator_features_id = "1njlJfbu8gEEKNoFkcVsa1EdZMQtZ20GR"
    operator_features_name = "operator_features.npz"




    def operator_data(filename,id,zip = False):
        if not os.path.exists(filename):
            raise AssertionError()
          #print("Downloading ",filename)
          #gdd.download_file_from_google_drive(file_id= id, dest_path='./' + filename, unzip = zip)
          #print("Downloading is completed")
        data = np.load(filename)
        return data
          
    def download(filename,source = "http://yann.lecun.com/exdb/mnist/"):
        print("Downloading ",filename)
        import urllib.request as ur
        ur.urlretrieve(source+filename,filename)
        print("Downloading is completed")
        
    def mnist_images(filename,offs = 16):
        if not os.path.exists(filename):
            download(filename)
            
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset = offs)
            data = data.reshape(-1,1,28,28)
        return data/np.float32(256)
        
    def mnist_label(filename,offs = 8):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset = offs)
        return data
        
    #Digit data Train & test
    dxtrain = mnist_images(tipath)
    dytrain = mnist_label(tlpath)
    dxtest = mnist_images(sipath)
    dytest = mnist_label(slpath)

    #operators Data
    opFeaturses = operator_data(operator_features_name, operator_features_id)['arr_0']
    opLabels = operator_data(operator_label_name, operator_label_id)



    #X_train, X_test, y_train, y_test
    opxtrain, opxtest, opytrain, opytest = train_test_split(opFeaturses, opLabels, test_size=0.8)

    X_train = np.concatenate((dxtrain, opxtrain))
    Y_train = np.concatenate((dytrain, opytrain))
    X_test =  np.concatenate((dxtest, opxtest))
    Y_test =  np.concatenate((dytest, opytest))
    os.chdir(r"/mnt/d/3 year of comuter engineering/second term/SW2/DigitRecognition-master/src/Calculator")

    return X_train, Y_train, X_test,Y_test
#print (X_train.shape)
