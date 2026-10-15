# Chemical-level aggregation of ML CV predictions (Supplementary Table 9). Run after run_gs1_gs2_analysis.py / run_gs3_analysis.py (expects ./outputs/*/CV_folds/fold*/).
import os,glob,numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,matthews_corrcoef,confusion_matrix
R='./outputs/'
CFG={('GS1','multi'):'CACTUS_GeneSet182/CV_folds',('GS2','multi'):'CACTUS_GeneSet150/CV_folds',('GS3','multi'):'CV_folds_CACTUS_GeneSet190/CV_folds',
     ('GS1','trans'):'CACTUS_GeneSet182/CV_folds',('GS2','trans'):'NonCACTUS_GeneSet150/CV_folds',('GS3','trans'):'CV_folds_NonCACTUS_GeneSet190/CV_folds'}
def models():
    return {'LR':LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=42),
            'SVM':SVC(kernel='rbf',C=1.0,probability=True,class_weight='balanced',random_state=42),
            'RF':RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',random_state=42)}
def chem(idx): return [i.rsplit('_',1)[0] for i in idx]
rows=[]
for (gs,mod),d in CFG.items():
    for f in range(1,6):
        p=f'{R}{d}/fold{f}/'
        X1=pd.read_csv(p+'input_data1.csv',index_col=0);y1=pd.read_csv(p+'output_data1.csv').iloc[:,0].values
        X2=pd.read_csv(p+'input_data2.csv',index_col=0);y2=pd.read_csv(p+'output_data2.csv').iloc[:,0].values
        if mod=='trans' and gs=='GS1':
            dc=['MW','LogP','TPSA','HBD','HBA','QED','BBB_penetrant','BBB_score','Brenk_alerts','PAINS_alerts']
            X1=X1.drop(columns=dc);X2=X2.drop(columns=dc)
        sc=StandardScaler().fit(X1);A=sc.transform(X1);B=sc.transform(X2)
        ch=np.array(chem(X2.index))
        for m,clf in models().items():
            clf.fit(A,y1);pr=clf.predict_proba(B)[:,1];yp=clf.predict(B)
            tn,fp,fn,tp=confusion_matrix(y2,yp,labels=[0,1]).ravel()
            r=dict(GS=gs,mod=mod,model=m,fold=f,s_spec=tn/(tn+fp),s_sens=tp/(tp+fn),n_s=len(y2))
            df=pd.DataFrame({'c':ch,'y':y2,'p':pr}).groupby('c',sort=False).agg(y=('y','first'),p=('p','mean'))
            yc=(df.p>=0.5).astype(int).values;yt=df.y.values
            tn,fp,fn,tp=confusion_matrix(yt,yc,labels=[0,1]).ravel()
            r.update(n_c=len(df),AUC=roc_auc_score(yt,df.p),sens=tp/(tp+fn),spec=tn/(tn+fp),bacc=(tp/(tp+fn)+tn/(tn+fp))/2,acc=(tp+tn)/len(df),MCC=matthews_corrcoef(yt,yc))
            rows.append(r)
d=pd.DataFrame(rows);d.to_csv('chem_level_folds.csv',index=False)
g=d.groupby(['GS','mod','model']).agg(['mean',lambda x:np.std(x,ddof=0)])
pd.set_option('display.width',250);pd.set_option('display.max_columns',50)
cols=['s_spec','AUC','sens','spec','bacc','MCC']
out=d.groupby(['GS','mod','model'])[cols].agg(['mean',lambda x:np.std(x,ddof=0)]).round(3)
print(out); out.to_csv('chem_level_summary.csv')
print(d.n_c.unique(),d.n_s.unique())
