# Sensitivity analysis: Gene Set 1 re-selected within each training split (Supplementary Table 10). Run from scripts/ml_pipeline. Inputs: ../../data/nested_selection_inputs (logFC of the 1,000 candidate genes; INGOR/Wilcoxon node table) and ../../data/multisigma_inputs/CACTUS_GeneSet182_CV_folds.
import re,numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from scipy.stats import mannwhitneyu
def bh(p):
    p=np.asarray(p);n=len(p);o=np.argsort(p);r=p[o]*n/(np.arange(n)+1)
    r=np.minimum.accumulate(r[::-1])[::-1];out=np.empty(n);out[o]=np.minimum(r,1);return out
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,matthews_corrcoef,confusion_matrix,accuracy_score
src=open('./run_gs1_gs2_analysis.py').read()
NO2PFX=eval(re.search(r'NO2PFX = (\{.*?\n\})',src,re.S).group(1))
lf=pd.read_csv('../../data/nested_selection_inputs/40_Chem_logFC_top1000.csv',index_col=0)
w=pd.read_csv('../../data/nested_selection_inputs/1000_ECv_wilcoxon_nodes_LabelA_complete.tsv',sep='\t')
genes1000=[g for g,pv in zip(w.name,w.p_value) if g in lf.index and not np.isnan(pv)]
sname={}
for no in range(1,36):
    p=NO2PFX[no]
    cols=sorted([c for c in lf.columns if c.startswith(p+'_') or c.startswith(p+' ')])
    for c in cols: sname[f"{p}_{c.split('_')[-1]}"]=(c,no)
R='../../data/multisigma_inputs/CACTUS_GeneSet182_CV_folds/'
DC=['MW','LogP','TPSA','HBD','HBA','QED','BBB_penetrant','BBB_score','Brenk_alerts','PAINS_alerts']
def models():
    return {'LR':LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=42),
            'SVM':SVC(kernel='rbf',C=1.0,probability=True,class_weight='balanced',random_state=42),
            'RF':RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',random_state=42)}
def select(train_names,ytr):
    cols=[sname[n][0] for n in train_names]
    A=lf.loc[genes1000,cols].values.astype(float)
    p=np.array([mannwhitneyu(A[i,ytr==1],A[i,ytr==0]).pvalue for i in range(len(genes1000))])
    fdr=bh(p)
    return [g for g,f in zip(genes1000,fdr) if f<=0.01]
# sanity: full-data selection
allnames=list(sname); yall=np.array([1 if sname[n][1]<=20 else 0 for n in allnames])
full=select(allnames,yall);print('full-data FDR<=0.01 genes:',len(full))
rows=[]
for f in range(1,6):
    X1=pd.read_csv(f'{R}fold{f}/input_data1.csv');X1.index=pd.read_csv(f'{R}fold{f}/input_data1_rowkey.csv')['sample'].values;y1=pd.read_csv(f'{R}fold{f}/output_data1.csv').iloc[:,0].values
    X2=pd.read_csv(f'{R}fold{f}/input_data2.csv');X2.index=pd.read_csv(f'{R}fold{f}/input_data2_rowkey.csv')['sample'].values;y2=pd.read_csv(f'{R}fold{f}/output_data2.csv').iloc[:,0].values
    sel=select(list(X1.index),y1);print('fold',f,'selected',len(sel),'overlap with GS1',len(set(sel)&set(full)))
    def feat(X,names,mod):
        G=lf.loc[sel,[sname[n][0] for n in names]].T.values.astype(float)
        return np.hstack([G,X[DC].values.astype(float)]) if mod=='multi' else G
    for mod in ('multi','trans'):
        A=feat(X1,list(X1.index),mod);B=feat(X2,list(X2.index),mod)
        sc=StandardScaler().fit(A);A=sc.transform(A);B=sc.transform(B)
        ch=np.array([i.rsplit('_',1)[0] for i in X2.index])
        for m,clf in models().items():
            clf.fit(A,y1);pr=clf.predict_proba(B)[:,1];yp=clf.predict(B)
            tn,fp,fn,tp=confusion_matrix(y2,yp,labels=[0,1]).ravel()
            df=pd.DataFrame({'c':ch,'y':y2,'p':pr}).groupby('c',sort=False).agg(y=('y','first'),p=('p','mean'))
            yc=(df.p>=0.5).astype(int).values;yt=df.y.values
            a,b,c_,d=confusion_matrix(yt,yc,labels=[0,1]).ravel()
            rows.append(dict(mod=mod,model=m,fold=f,nsel=len(sel),AUC=roc_auc_score(y2,pr),Acc=accuracy_score(y2,yp),Sens=tp/(tp+fn),Spec=tn/(tn+fp),MCC=matthews_corrcoef(y2,yp),
                 cAUC=roc_auc_score(yt,df.p),cBacc=(d/(d+c_)+a/(a+b))/2,cMCC=matthews_corrcoef(yt,yc)))
d=pd.DataFrame(rows);d.to_csv('nested_folds.csv',index=False)
cols=['AUC','Acc','Sens','Spec','MCC','cAUC','cBacc','cMCC']
print(d.groupby(['mod','model'])[cols].agg(['mean',lambda x:np.std(x,ddof=0)]).round(3).to_string())
print(d.groupby('fold').nsel.first().tolist())
