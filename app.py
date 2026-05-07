from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import sqlite3
import json
import os
import threading
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = 'model.pkl'
DB_PATH = 'colorai.db'
CSV_PATH = 'training_data.csv'
training_status = {'status': 'idle', 'message': '尚未訓練', 'progress': 0}

def build_features(L_arr, a_arr, b_arr, luster_arr):
    L=np.array(L_arr,dtype=float); a=np.array(a_arr,dtype=float)
    b=np.array(b_arr,dtype=float); luster=np.array(luster_arr,dtype=float)
    C=np.sqrt(a**2+b**2); H=np.degrees(np.arctan2(b,a))%360
    feats=[L,a,b,luster,C,H,L**2,a**2,b**2,C**2,luster**2,
           L*a,L*b,a*b,L*C,luster*L,luster*C,luster*a,luster*b,
           np.log1p(np.abs(a)),np.log1p(np.abs(b)),np.log1p(C),
           np.sin(np.radians(H)),np.cos(np.radians(H)),
           (L>80).astype(float),(L>60).astype(float),
           (L>40).astype(float),(L<30).astype(float),
           (luster>70).astype(float),(luster>40).astype(float),(luster<20).astype(float)]
    return np.column_stack(feats)

def load_model():
    if os.path.exists(MODEL_PATH):
        return pickle.load(open(MODEL_PATH,'rb'))
    return None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        target_L REAL, target_a REAL, target_b REAL, target_luster REAL,
        actual_L REAL, actual_a REAL, actual_b REAL, actual_luster REAL,
        delta_e REAL, formula_json TEXT, note TEXT
    )''')
    conn.commit(); conn.close()

init_db()

def delta_e_2000(L1,a1,b1,L2,a2,b2):
    C1=np.sqrt(a1**2+b1**2); C2=np.sqrt(a2**2+b2**2)
    Cavg=(C1+C2)/2; C7=Cavg**7
    G=0.5*(1-np.sqrt(C7/(C7+25**7)))
    a1p=a1*(1+G); a2p=a2*(1+G)
    C1p=np.sqrt(a1p**2+b1**2); C2p=np.sqrt(a2p**2+b2**2)
    h1p=np.degrees(np.arctan2(b1,a1p))%360; h2p=np.degrees(np.arctan2(b2,a2p))%360
    dLp=L2-L1; dCp=C2p-C1p
    dhp=h2p-h1p
    if abs(dhp)>180: dhp=dhp-360 if dhp>0 else dhp+360
    dHp=2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
    Lp=(L1+L2)/2; Cp=(C1p+C2p)/2; hp=(h1p+h2p)/2
    if abs(h1p-h2p)>180: hp+=180 if h1p+h2p<360 else -180
    T=1-0.17*np.cos(np.radians(hp-30))+0.24*np.cos(np.radians(2*hp))+0.32*np.cos(np.radians(3*hp+6))-0.20*np.cos(np.radians(4*hp-63))
    SL=1+0.015*(Lp-50)**2/np.sqrt(20+(Lp-50)**2)
    SC=1+0.045*Cp; SH=1+0.015*Cp*T
    Cp7=Cp**7; RC=2*np.sqrt(Cp7/(Cp7+25**7))
    dt=30*np.exp(-((hp-275)/25)**2); RT=-np.sin(np.radians(2*dt))*RC
    return float(np.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH)))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        L=float(data['L']); a=float(data['a']); b=float(data['b']); luster=float(data['luster'])
        model_data = load_model()
        if not model_data: return jsonify({'error':'模型未載入'}),500
        rf=model_data['rf']; xgb_models=model_data['xgb']; pigments=model_data['pigments']
        X_input = build_features([L],[a],[b],[luster])
        rf_pred = np.clip(rf.predict(X_input)[0],0,None)
        xgb_pred = np.clip(np.array([m.predict(X_input)[0] for m in xgb_models]),0,None)
        ensemble = (rf_pred+xgb_pred)/2
        total = ensemble.sum()
        if total>0: ensemble = ensemble/total*100
        formula = {p:round(float(ensemble[i]),2) for i,p in enumerate(pigments) if ensemble[i]>0.3}
        return jsonify({'success':True,'formula':formula,'target':{'L':L,'a':a,'b':b,'luster':luster}})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    try:
        data=request.json; target=data['target']; actual=data['actual']
        formula=data['formula']; note=data.get('note','')
        dE=delta_e_2000(target['L'],target['a'],target['b'],actual['L'],actual['a'],actual['b'])
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute('''INSERT INTO feedback
            (created_at,target_L,target_a,target_b,target_luster,
             actual_L,actual_a,actual_b,actual_luster,delta_e,formula_json,note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (datetime.now().strftime('%Y-%m-%d %H:%M'),
             target['L'],target['a'],target['b'],target.get('luster',0),
             actual['L'],actual['a'],actual['b'],actual.get('luster',0),
             dE,json.dumps(formula),note))
        conn.commit()
        count=c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
        conn.close()
        return jsonify({'success':True,'delta_e':round(dE,3),'total_feedback':count})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/history', methods=['GET'])
def history():
    try:
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        rows=c.execute('''SELECT id,created_at,target_L,target_a,target_b,target_luster,
                         actual_L,actual_a,actual_b,actual_luster,delta_e,formula_json,note
                         FROM feedback ORDER BY id DESC LIMIT 50''').fetchall()
        conn.close()
        records=[{'id':r[0],'date':r[1],
                  'target':{'L':r[2],'a':r[3],'b':r[4],'luster':r[5]},
                  'actual':{'L':r[6],'a':r[7],'b':r[8],'luster':r[9]},
                  'delta_e':r[10],'formula':json.loads(r[11]),'note':r[12]} for r in rows]
        return jsonify({'success':True,'records':records})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    count=c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
    conn.close()
    return jsonify({**training_status,'feedback_count':count})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_status
    if training_status['status']=='running':
        return jsonify({'error':'訓練中，請稍候'}),400

    def run_training():
        global training_status
        try:
            training_status={'status':'running','message':'讀取資料...','progress':10}
            df=pd.read_csv(CSV_PATH)
            training_status={'status':'running','message':'合併回填資料...','progress':20}
            conn=sqlite3.connect(DB_PATH)
            rows=conn.execute('SELECT target_L,target_a,target_b,target_luster,formula_json FROM feedback').fetchall()
            conn.close()
            model_data=load_model(); pigments=model_data['pigments']
            extra_rows=[]
            for row in rows:
                formula=json.loads(row[4])
                new_row={'Target_L':row[0],'Target_a':row[1],'Target_b':row[2],'Target_luster':row[3]}
                for p in pigments: new_row[p]=formula.get(p,0)
                extra_rows.append(new_row)
            if extra_rows:
                extra_df=pd.DataFrame(extra_rows)
                for col in df.columns:
                    if col not in extra_df.columns: extra_df[col]=0
                df=pd.concat([df,extra_df],ignore_index=True)
            training_status={'status':'running','message':f'訓練中（{len(df)} 筆資料）...','progress':35}
            X=build_features(df['Target_L'].values,df['Target_a'].values,df['Target_b'].values,df['Target_luster'].values)
            y=df[pigments].values
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)
            xgb_models=[]
            for i,col in enumerate(pigments):
                y_col=y_train[:,i]; hv=(y_col>0.01).mean()
                if hv>0.5: params=dict(n_estimators=500,max_depth=8,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,min_child_weight=3)
                else: params=dict(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5)
                m=XGBRegressor(**params,random_state=42,verbosity=0)
                m.fit(X_train,y_col); xgb_models.append(m)
                training_status['progress']=35+int(40*(i+1)/len(pigments))
            training_status={'status':'running','message':'訓練 Random Forest...','progress':78}
            rf=RandomForestRegressor(n_estimators=500,max_depth=20,min_samples_leaf=2,max_features=0.7,random_state=42,n_jobs=-1)
            rf.fit(X_train,y_train)
            xgb_pred=np.clip(np.column_stack([m.predict(X_test) for m in xgb_models]),0,None)
            rf_pred=np.clip(rf.predict(X_test),0,None)
            mae=mean_absolute_error(y_test,(xgb_pred+rf_pred)/2)
            pickle.dump({'rf':rf,'xgb':xgb_models,'pigments':pigments,'feature_count':X.shape[1]},open(MODEL_PATH,'wb'))
            training_status={'status':'done','message':f'✅ 完成！{len(df)} 筆資料，平均誤差 ±{mae:.3f}%','progress':100}
        except Exception as e:
            training_status={'status':'error','message':f'❌ 失敗：{str(e)}','progress':0}

    threading.Thread(target=run_training,daemon=True).start()
    return jsonify({'success':True,'message':'訓練已開始'})

@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        count=c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
        avg_de=c.execute('SELECT AVG(delta_e) FROM feedback').fetchone()[0]
        good=c.execute('SELECT COUNT(*) FROM feedback WHERE delta_e < 2').fetchone()[0]
        conn.close()
        df=pd.read_csv(CSV_PATH)
        return jsonify({'success':True,'original_data':len(df),'feedback_count':count,
                       'total_data':len(df)+count,
                       'avg_delta_e':round(avg_de,2) if avg_de else None,
                       'good_rate':round(good/count*100,1) if count>0 else None})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/')
def index():
    return send_from_directory('static','index.html')

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
