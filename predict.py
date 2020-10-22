import pandas as pd 
import numpy as np
import joblib
import time
import datetime
from jax_unirep import get_reps
import argparse 



def fasta2pd(inFasta): ##convert a FASTA file into pandas.DataFrame
    FastaRead=pd.read_csv(inFasta,header=None)
     
    seqNum=int(FastaRead.shape[0]/2)
    csvFile=open("testFasta.csv","w")
    csvFile.write("Seq,Target\n")
    
    for i in range(seqNum):
        if  i < seqNum/2:
            csvFile.write(str(FastaRead.iloc[2*i+1,0])+",1\n")
            
        if i >=seqNum/2:
            csvFile.write(str(FastaRead.iloc[2*i+1,0])+",0\n")
    csvFile.close()
    SeqLabel=pd.read_csv("testFasta.csv",header=0)
    print(SeqLabel.shape)
    
    return SeqLabel

def predict(inputFasta):##input a FASTA file
    
    
    InData=fasta2pd(inputFasta) ##Read FASTA file and convert it into a pandas DataFrame
    Seq=InData["Seq"]
    
    Features=pd.read_csv("Features.csv",header=0).columns ##Features for Prediction
    scale=joblib.load("TrainScale.pkl") ##Data Standardization
    
    ##Convert protein Sequences into UniRep Features Vectors
    print("Convert protein Sequences into UniRep Features Vectors...")
    col_h_vag=[]
    for i in range(1900):
        col_h_vag.append("H_VAG_"+str(i+1)) 
        
    h_vag,h_final,c_final=get_reps(Seq)
    HVAG=pd.DataFrame(h_vag,columns=col_h_vag) 
        
    X=HVAG[Features]
    X=scale.transform(X)
    X=X[:,:107] ## Data Vectors for prediction inputs
    
   
    print("Predicting...")
    model=joblib.load("isGP-DRLF_Model.pkl") ##Load isGP-DRLF trained Model
    y_pred_prob=model.predict_proba(X)
    df_out=pd.DataFrame(np.zeros((y_pred_prob.shape[0],3)),columns=["Index","Prediction","Sequences"])
   
    y_pred=model.predict(X)
    for i in range(y_pred.shape[0]):
        df_out.iloc[i,0]="Seq"+str(1+i)
        df_out.iloc[i,2]=str(InData.iloc[i,0])
        
        if y_pred[i]==1:
            df_out.iloc[i,1]="cis-Golgi"
             
        if y_pred[i]==0:
            df_out.iloc[i,1]="trans-Golgi"
   
    return df_out


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Proteins sub-Golgi Localization Prediction by UniRep Deep Representaion Learning Features')
    parser.add_argument('-i', default=None,help='input a FASTA file')
    parser.add_argument('-o', default="DemoResults.csv",help='output a CSV results file')
    args = parser.parse_args()
        
    start=time.time()
    print(datetime.datetime.now())
    
    df = predict(args.i)
    df.to_csv(args.o)
   
    print(df.shape[0], " protein sequence")
    print("\n\n",datetime.datetime.now())
    print("Elapsed time =%.2f hours"%((time.time()-start)/3600))
 