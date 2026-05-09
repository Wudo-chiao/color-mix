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
 
app = Flask(__name__, static_folder='static', static_url_path='')
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
 
@app.route('/', defaults={'path': ''})
@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 調色系統 | 美藝堅塗料</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500&family=Noto+Serif+TC:wght@600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--navy:#042C53;--teal:#1D9E75;--teal-dark:#0F6E56;--teal-pale:#E1F5EE;--text:#1a1a1a;--muted:#777;--border:rgba(0,0,0,0.09);--bg:#f7f7f5;--white:#fff;}
body{font-family:'Noto Sans TC',sans-serif;font-weight:300;color:var(--text);background:var(--bg);min-height:100vh;}
header{background:var(--navy);padding:16px 36px;display:flex;align-items:center;justify-content:space-between;}
header h1{font-family:'Noto Serif TC',serif;font-size:18px;color:white;font-weight:600;}
.header-right{display:flex;align-items:center;gap:12px;}
.badge{font-size:11px;padding:4px 12px;border-radius:20px;letter-spacing:0.5px;}
.badge-teal{background:rgba(29,158,117,0.2);border:1px solid rgba(29,158,117,0.4);color:#9FE1CB;}
.badge-white{background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);color:rgba(255,255,255,0.6);}
 
/* TABS */
.tabs{background:white;border-bottom:1px solid var(--border);display:flex;padding:0 36px;}
.tab{padding:14px 20px;font-size:14px;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all 0.2s;}
.tab.active{color:var(--teal);border-bottom-color:var(--teal);font-weight:500;}
.tab-content{display:none;} .tab-content.active{display:block;}
 
main{max-width:1100px;margin:0 auto;padding:28px 24px;}
 
/* CARDS */
.card{background:white;border:1px solid var(--border);border-radius:14px;padding:24px;margin-bottom:20px;}
.card-title{font-family:'Noto Serif TC',serif;font-size:16px;font-weight:600;color:var(--navy);margin-bottom:5px;}
.card-sub{font-size:13px;color:var(--muted);margin-bottom:20px;line-height:1.7;}
 
/* PREDICT LAYOUT */
.predict-layout{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start;}
 
/* INPUTS */
.divider-label{font-size:11px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin:14px 0 10px;display:flex;align-items:center;gap:8px;}
.divider-label::before,.divider-label::after{content:'';flex:1;height:1px;background:var(--border);}
.input-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px;}
.ig label{display:block;font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;}
.ig input{width:100%;padding:11px 10px;border:1.5px solid var(--border);border-radius:9px;font-size:17px;font-family:'Noto Sans TC',sans-serif;font-weight:500;text-align:center;transition:border-color 0.2s;}
.ig input:focus{outline:none;border-color:var(--teal);}
.luster-row{margin-bottom:16px;}
.luster-row label{font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;display:block;margin-bottom:6px;}
.luster-input-row{display:flex;gap:10px;align-items:center;}
.luster-input-row input[type=number]{width:100px;padding:11px 10px;border:1.5px solid var(--border);border-radius:9px;font-size:17px;font-family:'Noto Sans TC',sans-serif;font-weight:500;text-align:center;}
.luster-input-row input[type=number]:focus{outline:none;border-color:#8B5CF6;}
.luster-slider{flex:1;height:6px;border-radius:3px;appearance:none;cursor:pointer;background:linear-gradient(to right,#1a1a1a 0%,#888 35%,#ddd 70%,#fff 100%);}
.luster-slider::-webkit-slider-thumb{appearance:none;width:18px;height:18px;border-radius:50%;background:#8B5CF6;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.2);cursor:pointer;}
 
/* PREVIEW */
.preview-row{display:flex;gap:12px;align-items:center;margin-bottom:16px;}
.swatch{width:72px;height:72px;border-radius:10px;border:1px solid var(--border);flex-shrink:0;position:relative;overflow:hidden;transition:background 0.3s;}
.swatch-gloss{position:absolute;top:0;left:0;right:0;height:40%;background:linear-gradient(to bottom,rgba(255,255,255,0.55),transparent);border-radius:10px 10px 0 0;}
.preview-text .lab{font-size:13px;color:var(--muted);margin-bottom:3px;} .preview-text .lab span{font-weight:500;color:var(--text);}
.preview-text .desc{font-size:12px;color:var(--muted);}
 
/* BUTTONS */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:13px 24px;border-radius:10px;font-size:15px;font-family:'Noto Sans TC',sans-serif;font-weight:400;border:none;cursor:pointer;transition:all 0.2s;width:100%;}
.btn-teal{background:var(--teal);color:white;} .btn-teal:hover{background:var(--teal-dark);}
.btn-navy{background:var(--navy);color:white;} .btn-navy:hover{background:#063a6e;}
.btn-outline{background:transparent;color:var(--teal);border:1.5px solid var(--teal);} .btn-outline:hover{background:var(--teal);color:white;}
.btn:disabled{background:#ccc;cursor:not-allowed;}
.btn-sm{padding:8px 16px;font-size:13px;width:auto;border-radius:7px;}
 
/* FORMULA RESULT */
.formula-result{background:white;border:1px solid var(--border);border-radius:14px;overflow:hidden;}
.formula-header{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.formula-header h3{font-size:15px;font-weight:500;}
.formula-badges{display:flex;gap:8px;}
.fbadge{font-size:11px;padding:3px 10px;border-radius:20px;}
.fbadge-color{background:var(--teal-pale);color:var(--teal-dark);}
.fbadge-luster{background:#F3EFFE;color:#7C3AED;}
.formula-body{padding:8px 0;}
.fgroup-title{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;padding:10px 20px 5px;}
.formula-row{display:flex;align-items:center;gap:10px;padding:9px 20px;transition:background 0.15s;}
.formula-row:hover{background:var(--bg);}
.pdot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.pname{font-size:14px;font-weight:500;min-width:68px;}
.pcode{font-size:11px;color:var(--muted);min-width:108px;}
.pbar{flex:1;height:7px;background:var(--bg);border-radius:4px;overflow:hidden;}
.pbar-fill{height:100%;border-radius:4px;transition:width 0.6s ease;}
.ppct{font-size:14px;font-weight:500;min-width:46px;text-align:right;}
.pgram{font-size:12px;color:var(--muted);min-width:56px;text-align:right;}
.formula-placeholder{padding:48px;text-align:center;color:var(--muted);font-size:14px;line-height:1.9;}
 
/* BATCH */
.batch-row{display:flex;gap:10px;margin-top:14px;}
.batch-row input{flex:1;padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:15px;font-family:inherit;}
.batch-row input:focus{outline:none;border-color:var(--teal);}
.batch-row select{padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:13px;font-family:inherit;background:white;}
.batch-result{margin-top:12px;font-size:13px;color:var(--muted);line-height:2.1;}
.batch-result strong{color:var(--text);font-weight:500;}
 
/* FEEDBACK */
.fb-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:14px;}
.fb-grid label{font-size:11px;color:var(--muted);display:block;margin-bottom:5px;letter-spacing:0.5px;}
.fb-grid input{width:100%;padding:9px 10px;border:1.5px solid var(--border);border-radius:8px;font-size:15px;font-family:inherit;text-align:center;}
.fb-grid input:focus{outline:none;border-color:var(--teal);}
.de-box{display:flex;align-items:center;gap:14px;padding:14px;background:var(--bg);border-radius:10px;margin-bottom:14px;}
.de-val{font-size:32px;font-weight:600;font-family:'Noto Serif TC',serif;min-width:70px;}
.de-good{color:var(--teal);} .de-warn{color:#D97706;} .de-bad{color:#DC2626;}
.de-info{font-size:13px;color:var(--muted);line-height:1.7;}
.success-msg{display:none;background:var(--teal-pale);border:1px solid rgba(29,158,117,0.3);border-radius:8px;padding:10px 14px;font-size:13px;color:var(--teal-dark);margin-top:10px;}
 
/* HISTORY TABLE */
.history-table{width:100%;border-collapse:collapse;font-size:13px;}
.history-table th{text-align:left;padding:10px 14px;font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);font-weight:400;}
.history-table td{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}
.history-table tr:last-child td{border-bottom:none;}
.history-table tr:hover td{background:var(--bg);}
.color-dot{width:24px;height:24px;border-radius:5px;border:1px solid var(--border);display:inline-block;vertical-align:middle;}
.de-chip{font-size:11px;padding:3px 8px;border-radius:10px;font-weight:500;}
.chip-good{background:#D1FAE5;color:#065F46;}
.chip-warn{background:#FEF3C7;color:#92400E;}
.chip-bad{background:#FEE2E2;color:#991B1B;}
 
/* TRAINING */
.train-status{padding:16px;background:var(--bg);border-radius:10px;margin-bottom:16px;}
.train-bar-wrap{height:8px;background:#e0e0e0;border-radius:4px;margin:10px 0 6px;overflow:hidden;}
.train-bar{height:100%;background:var(--teal);border-radius:4px;transition:width 0.4s ease;}
.train-msg{font-size:13px;color:var(--muted);}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;}
.stat-box{background:var(--bg);border-radius:10px;padding:16px;text-align:center;}
.stat-box .val{font-size:24px;font-weight:600;font-family:'Noto Serif TC',serif;color:var(--teal);margin-bottom:4px;}
.stat-box .lbl{font-size:12px;color:var(--muted);}
 
/* STATUS */
.status-bar{padding:10px 14px;border-radius:8px;font-size:13px;margin-top:10px;display:none;}
.status-err{background:#fef2f2;color:#b91c1c;border:1px solid rgba(220,38,38,0.2);}
.status-ok{background:var(--teal-pale);color:var(--teal-dark);border:1px solid rgba(29,158,117,0.2);}
 
@media(max-width:800px){.predict-layout{grid-template-columns:1fr;}}
</style>
</head>
<body>
 
<header>
  <h1>美藝堅 AI 調色系統</h1>
  <div class="header-right">
    <span class="badge badge-teal" id="data-badge">載入中...</span>
    <span class="badge badge-white">XGBoost + RF 模型</span>
  </div>
</header>
 
<div class="tabs">
  <div class="tab active" onclick="switchTab('predict')">🎨 配方預測</div>
  <div class="tab" onclick="switchTab('feedback')">📋 打樣回饋</div>
  <div class="tab" onclick="switchTab('history')">📂 歷史記錄</div>
  <div class="tab" onclick="switchTab('training')">⚙️ 模型訓練</div>
</div>
 
<!-- ── TAB 1：配方預測 ── -->
<div class="tab-content active" id="tab-predict">
<main>
  <div class="predict-layout">
    <div>
      <div class="card">
        <div class="card-title">輸入目標顏色</div>
        <div class="card-sub">輸入來樣的 LAB 值與光澤度，AI 自動預測配方</div>
 
        <div class="divider-label">顏色 LAB 值</div>
        <div class="input-grid-3">
          <div class="ig"><label>L*（明度）</label><input id="L" type="number" min="0" max="100" step="0.1" placeholder="0~100" oninput="updatePreview()"></div>
          <div class="ig"><label>a*（紅綠）</label><input id="a" type="number" min="-128" max="127" step="0.1" placeholder="-128~127" oninput="updatePreview()"></div>
          <div class="ig"><label>b*（黃藍）</label><input id="b" type="number" min="-128" max="127" step="0.1" placeholder="-128~127" oninput="updatePreview()"></div>
        </div>
 
        <div class="divider-label">光澤度</div>
        <div class="luster-row">
          <label>光澤值（0=消光　91=高光）</label>
          <div class="luster-input-row">
            <input type="number" id="luster" min="0" max="100" step="0.5" placeholder="例：85" oninput="syncSlider();updatePreview()">
            <input type="range" class="luster-slider" id="luster-slider" min="0" max="91" step="0.5" value="50" oninput="syncLuster();updatePreview()">
          </div>
        </div>
 
        <div class="preview-row">
          <div class="swatch" id="swatch"><div class="swatch-gloss" id="gloss-overlay"></div></div>
          <div class="preview-text">
            <div class="lab" id="preview-lab">請輸入 LAB 數值</div>
            <div class="lab" id="preview-luster" style="color:#8B5CF6;">光澤：—</div>
            <div class="desc" id="preview-desc">—</div>
          </div>
        </div>
 
        <button class="btn btn-teal" id="btn-predict" onclick="predict()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          預測配方
        </button>
        <div class="status-bar status-err" id="predict-status"></div>
      </div>
    </div>
 
    <div>
      <div class="formula-result" id="formula-result">
        <div class="formula-placeholder">
          在左側輸入 LAB 值與光澤度<br>系統將使用真實 AI 模型預測最佳配方
        </div>
      </div>
 
      <div class="card" id="batch-card" style="display:none;margin-top:16px;">
        <div class="card-title" style="margin-bottom:12px;">用量計算</div>
        <div class="batch-row">
          <input type="number" id="batch-kg" placeholder="輸入總重量" min="0" step="0.1" oninput="calcBatch()">
          <select id="batch-unit" onchange="calcBatch()">
            <option value="1000">公斤（kg）</option>
            <option value="1">公克（g）</option>
          </select>
          <button class="btn btn-navy btn-sm" onclick="calcBatch()">計算</button>
        </div>
        <div class="batch-result" id="batch-result"></div>
      </div>
    </div>
  </div>
</main>
</div>
 
<!-- ── TAB 2：打樣回饋 ── -->
<div class="tab-content" id="tab-feedback">
<main>
  <div class="card">
    <div class="card-title">打樣回饋記錄</div>
    <div class="card-sub">打樣完成後，輸入實際量測的 LAB 值，系統自動計算 ΔE2000 並儲存。回填的資料將用於下次模型重新訓練。</div>
 
    <div class="divider-label">目標顏色（預測時的輸入值）</div>
    <div class="fb-grid">
      <div><label>目標 L*</label><input type="number" id="fb-tL" step="0.1" placeholder="—"></div>
      <div><label>目標 a*</label><input type="number" id="fb-ta" step="0.1" placeholder="—"></div>
      <div><label>目標 b*</label><input type="number" id="fb-tb" step="0.1" placeholder="—"></div>
      <div><label>目標光澤</label><input type="number" id="fb-tluster" step="0.1" placeholder="—"></div>
    </div>
 
    <div class="divider-label">實際打樣量測值</div>
    <div class="fb-grid">
      <div><label>實測 L*</label><input type="number" id="fb-aL" step="0.1" placeholder="—" oninput="calcDE()"></div>
      <div><label>實測 a*</label><input type="number" id="fb-aa" step="0.1" placeholder="—" oninput="calcDE()"></div>
      <div><label>實測 b*</label><input type="number" id="fb-ab" step="0.1" placeholder="—" oninput="calcDE()"></div>
      <div><label>實測光澤</label><input type="number" id="fb-aluster" step="0.1" placeholder="—"></div>
    </div>
 
    <div class="divider-label">使用的配方（可手動輸入正確配方）</div>
    <div id="fb-formula-display" style="font-size:13px;color:var(--muted);padding:12px;background:var(--bg);border-radius:8px;margin-bottom:14px;min-height:48px;">
      請先在「配方預測」頁預測一次，配方會自動帶入；或手動輸入下方欄位。
    </div>
    <div style="margin-bottom:14px;">
      <label style="font-size:12px;color:var(--muted);display:block;margin-bottom:6px;">備注（選填）</label>
      <input type="text" id="fb-note" placeholder="例：第一次打樣，需微調黑色" style="width:100%;padding:10px 12px;border:1.5px solid var(--border);border-radius:8px;font-size:14px;font-family:inherit;">
    </div>
 
    <div class="de-box" id="de-box" style="display:none;">
      <div class="de-val" id="de-val">—</div>
      <div class="de-info" id="de-info">ΔE2000 色差值</div>
    </div>
 
    <button class="btn btn-teal" onclick="saveFeedback()">儲存打樣記錄</button>
    <div class="success-msg" id="fb-success">✅ 記錄已儲存！回填資料將納入下次模型訓練。</div>
    <div class="status-bar status-err" id="fb-status"></div>
  </div>
</main>
</div>
 
<!-- ── TAB 3：歷史記錄 ── -->
<div class="tab-content" id="tab-history">
<main>
  <div class="card">
    <div class="card-title">打樣記錄歷史</div>
    <div class="card-sub">所有回填記錄，儲存在雲端資料庫，永久保存。</div>
    <div id="history-content"><div style="text-align:center;color:var(--muted);padding:40px;">載入中...</div></div>
  </div>
</main>
</div>
 
<!-- ── TAB 4：模型訓練 ── -->
<div class="tab-content" id="tab-training">
<main>
  <div class="card">
    <div class="card-title">資料統計</div>
    <div class="stats-grid" id="stats-grid">
      <div class="stat-box"><div class="val" id="s-original">—</div><div class="lbl">原始訓練資料</div></div>
      <div class="stat-box"><div class="val" id="s-feedback">—</div><div class="lbl">回填筆數</div></div>
      <div class="stat-box"><div class="val" id="s-total">—</div><div class="lbl">總資料量</div></div>
      <div class="stat-box"><div class="val" id="s-good">—</div><div class="lbl">ΔE &lt; 2 比率</div></div>
    </div>
  </div>
 
  <div class="card">
    <div class="card-title">重新訓練模型</div>
    <div class="card-sub">點擊下方按鈕，系統會合併原始資料與所有回填資料，重新訓練 XGBoost + Random Forest 模型。訓練時間約 3~5 分鐘。</div>
 
    <div class="train-status" id="train-status">
      <div style="font-size:13px;font-weight:500;margin-bottom:8px;" id="train-title">模型狀態</div>
      <div class="train-bar-wrap"><div class="train-bar" id="train-bar" style="width:0%"></div></div>
      <div class="train-msg" id="train-msg">尚未訓練</div>
    </div>
 
    <button class="btn btn-teal" id="btn-train" onclick="startTraining()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
      開始重新訓練
    </button>
  </div>
</main>
</div>
 
<script>
const API = '';  // 同域，不需要前綴
const PIGMENTS = ['5000-6900','5000-6909','5000-6906','5000-6941','5000-6900N',
  '5000-6937','5000-6912','5000-6926','5000-6930','5000-6903','5000-6911',
  '5000-6916','5000-6905','5000-6914','P0502','P0508','P0509','P0604','M7001','M7005','TH-6980'];
const PNAMES={'5000-6900':'白色','5000-6909':'黃色','5000-6906':'紅色','5000-6941':'黑色',
  '5000-6900N':'白色N','5000-6937':'藍綠','5000-6912':'橙黃','5000-6926':'深藍',
  '5000-6930':'藍色','5000-6903':'橘紅','5000-6911':'洋紅','5000-6916':'黃綠',
  '5000-6905':'深紅','5000-6914':'黃色2','P0502':'消光劑','P0508':'助劑P08',
  'P0509':'助劑P09','P0604':'助劑P04','M7001':'助劑M01','M7005':'助劑M05','TH-6980':'稀釋劑'};
const PCOLORS={'5000-6900':'#E8E8E8','5000-6909':'#F5C842','5000-6906':'#C84B2F','5000-6941':'#333',
  '5000-6900N':'#EFEFEF','5000-6937':'#2E7D6E','5000-6912':'#E8963C','5000-6926':'#1A3A6E',
  '5000-6930':'#2255AA','5000-6903':'#D4521E','5000-6911':'#C8387A','5000-6916':'#8DB832',
  '5000-6905':'#8B1A1A','5000-6914':'#E8C020','P0502':'#9F7AEA','P0508':'#aaa',
  'P0509':'#bbb','P0604':'#ccc','M7001':'#ddd','M7005':'#7C3AED','TH-6980':'#6B7280'};
const ADDITIVES=['P0502','P0508','P0509','P0604','M7001','M7005','TH-6980'];
 
let currentFormula=null, currentTarget=null;
 
function switchTab(t){
  document.querySelectorAll('.tab').forEach((el,i)=>{
    const tabs=['predict','feedback','history','training'];
    el.classList.toggle('active', tabs[i]===t);
  });
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  if(t==='history') loadHistory();
  if(t==='training'){loadStats();checkTrainingStatus();}
}
 
function lab2rgb(L,a,b){
  let y=(L+16)/116,x=a/500+y,z=y-b/200;
  x=0.95047*(x*x*x>0.008856?x*x*x:(x-16/116)/7.787);
  y=1.00000*(y*y*y>0.008856?y*y*y:(y-16/116)/7.787);
  z=1.08883*(z*z*z>0.008856?z*z*z:(z-16/116)/7.787);
  let r=x*3.2406+y*(-1.5372)+z*(-0.4986);
  let g=x*(-0.9689)+y*1.8758+z*0.0415;
  let bl=x*0.0557+y*(-0.2040)+z*1.0570;
  const clamp=v=>Math.min(255,Math.max(0,Math.round((v>0.0031308?1.055*Math.pow(v,1/2.4)-0.055:12.92*v)*255)));
  return `rgb(${clamp(r)},${clamp(g)},${clamp(bl)})`;
}
 
function deltaE2000(L1,a1,b1,L2,a2,b2){
  const C1=Math.sqrt(a1*a1+b1*b1),C2=Math.sqrt(a2*a2+b2*b2);
  const Cavg=(C1+C2)/2,C7=Math.pow(Cavg,7);
  const G=0.5*(1-Math.sqrt(C7/(C7+Math.pow(25,7))));
  const a1p=a1*(1+G),a2p=a2*(1+G);
  const C1p=Math.sqrt(a1p*a1p+b1*b1),C2p=Math.sqrt(a2p*a2p+b2*b2);
  const h1p=((Math.atan2(b1,a1p)*180/Math.PI)+360)%360;
  const h2p=((Math.atan2(b2,a2p)*180/Math.PI)+360)%360;
  const dLp=L2-L1,dCp=C2p-C1p;
  let dhp=h2p-h1p;
  if(Math.abs(dhp)>180) dhp=dhp>0?dhp-360:dhp+360;
  const dHp=2*Math.sqrt(C1p*C2p)*Math.sin(dhp/2*Math.PI/180);
  const Lp=(L1+L2)/2,Cp=(C1p+C2p)/2;
  let hp=(h1p+h2p)/2;
  if(Math.abs(h1p-h2p)>180) hp+=h1p+h2p<360?180:-180;
  const T=1-0.17*Math.cos((hp-30)*Math.PI/180)+0.24*Math.cos(2*hp*Math.PI/180)+0.32*Math.cos((3*hp+6)*Math.PI/180)-0.20*Math.cos((4*hp-63)*Math.PI/180);
  const SL=1+0.015*(Lp-50)*(Lp-50)/Math.sqrt(20+(Lp-50)*(Lp-50));
  const SC=1+0.045*Cp,SH=1+0.015*Cp*T;
  const Cp7=Math.pow(Cp,7),RC=2*Math.sqrt(Cp7/(Cp7+Math.pow(25,7)));
  const dt=30*Math.exp(-Math.pow((hp-275)/25,2));
  const RT=-Math.sin(2*dt*Math.PI/180)*RC;
  return Math.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH));
}
 
function syncSlider(){document.getElementById('luster-slider').value=Math.min(91,parseFloat(document.getElementById('luster').value)||50);}
function syncLuster(){document.getElementById('luster').value=document.getElementById('luster-slider').value;}
 
function descColor(L,a,b,luster){
  const parts=[];
  if(L>80)parts.push('淺色');else if(L>55)parts.push('中明度');else if(L>30)parts.push('中深色');else parts.push('深色');
  const C=Math.sqrt(a*a+b*b);
  if(C<5)parts.push('中性灰');
  else if(a>20&&b>10)parts.push('橘紅色系');
  else if(a>15)parts.push('紅色系');
  else if(b>20)parts.push('黃色系');
  else if(b<-15)parts.push('藍色系');
  else if(a<-10)parts.push('綠色系');
  if(luster>70)parts.push('高光');else if(luster>30)parts.push('半光');else parts.push('消光');
  return parts.join('・');
}
 
function updatePreview(){
  const L=parseFloat(document.getElementById('L').value);
  const a=parseFloat(document.getElementById('a').value);
  const b=parseFloat(document.getElementById('b').value);
  const luster=parseFloat(document.getElementById('luster').value);
  if(isNaN(L)||isNaN(a)||isNaN(b))return;
  document.getElementById('swatch').style.background=lab2rgb(L,a,b);
  document.getElementById('gloss-overlay').style.opacity=isNaN(luster)?0.3:Math.min(luster/91,1)*0.55;
  document.getElementById('preview-lab').innerHTML=`L* <span style="color:#1a1a1a;font-weight:500">${L.toFixed(1)}</span>　a* <span style="color:#1a1a1a;font-weight:500">${a.toFixed(1)}</span>　b* <span style="color:#1a1a1a;font-weight:500">${b.toFixed(1)}</span>`;
  document.getElementById('preview-luster').textContent=`光澤：${isNaN(luster)?'—':luster+' ('+（luster>70?'高光':luster>30?'半光':'消光')+')'}`; 
  document.getElementById('preview-desc').textContent=descColor(L,a,b,isNaN(luster)?50:luster);
}
 
async function predict(){
  const L=parseFloat(document.getElementById('L').value);
  const a=parseFloat(document.getElementById('a').value);
  const b=parseFloat(document.getElementById('b').value);
  const luster=parseFloat(document.getElementById('luster').value);
  if(isNaN(L)||isNaN(a)||isNaN(b)){showStatus('predict-status','請輸入完整的 L*、a*、b* 數值');return;}
  if(isNaN(luster)){showStatus('predict-status','請輸入光澤度');return;}
 
  const btn=document.getElementById('btn-predict');
  btn.disabled=true; btn.textContent='AI 計算中...';
 
  try{
    const res=await fetch('/api/predict',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({L,a,b,luster})
    });
    const data=await res.json();
    if(!data.success){showStatus('predict-status',data.error);return;}
 
    currentFormula=data.formula;
    currentTarget={L,a,b,luster};
    renderFormula(data.formula,L,a,b,luster);
    document.getElementById('batch-card').style.display='block';
 
    // 自動帶入回饋頁
    document.getElementById('fb-tL').value=L;
    document.getElementById('fb-ta').value=a;
    document.getElementById('fb-tb').value=b;
    document.getElementById('fb-tluster').value=luster;
    updateFbFormula(data.formula);
    hideStatus('predict-status');
  }catch(e){
    showStatus('predict-status','連線失敗，請確認伺服器是否啟動');
  }finally{
    btn.disabled=false;
    btn.innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>重新預測';
  }
}
 
function renderFormula(formula,L,a,b,luster){
  const colorP=Object.entries(formula).filter(([p])=>!ADDITIVES.includes(p)).sort((x,y)=>y[1]-x[1]);
  const addP=Object.entries(formula).filter(([p])=>ADDITIVES.includes(p)).sort((x,y)=>y[1]-x[1]);
 
  let html=`<div class="formula-header">
    <h3>建議配方</h3>
    <div class="formula-badges">
      <span class="fbadge fbadge-color">L${L.toFixed(0)} a${a.toFixed(0)} b${b.toFixed(0)}</span>
      <span class="fbadge fbadge-luster">光澤 ${luster}</span>
    </div>
  </div><div class="formula-body">`;
 
  if(colorP.length){
    html+=`<div class="fgroup-title">色母</div>`;
    colorP.forEach(([p,v])=>{
      html+=`<div class="formula-row">
        <div class="pdot" style="background:${PCOLORS[p]||'#ccc'}"></div>
        <div class="pname">${PNAMES[p]||p}</div>
        <div class="pcode">${p}</div>
        <div class="pbar"><div class="pbar-fill" style="width:${Math.min(v,100)}%;background:${PCOLORS[p]||'var(--teal)'}"></div></div>
        <div class="ppct">${v.toFixed(1)}%</div>
        <div class="pgram" id="g-${p.replace(/[^a-z0-9]/gi,'_')}">—</div>
      </div>`;
    });
  }
  if(addP.length){
    html+=`<div class="fgroup-title">添加劑</div>`;
    addP.forEach(([p,v])=>{
      html+=`<div class="formula-row">
        <div class="pdot" style="background:${PCOLORS[p]||'#9F7AEA'}"></div>
        <div class="pname">${PNAMES[p]||p}</div>
        <div class="pcode">${p}</div>
        <div class="pbar"><div class="pbar-fill" style="width:${Math.min(v,100)}%;background:#9F7AEA"></div></div>
        <div class="ppct">${v.toFixed(1)}%</div>
        <div class="pgram" id="g-${p.replace(/[^a-z0-9]/gi,'_')}">—</div>
      </div>`;
    });
  }
  html+=`</div>`;
  document.getElementById('formula-result').innerHTML=html;
}
 
function calcBatch(){
  if(!currentFormula)return;
  const kg=parseFloat(document.getElementById('batch-kg').value);
  const unit=parseFloat(document.getElementById('batch-unit').value);
  if(isNaN(kg)||kg<=0)return;
  const totalG=kg*unit;
  let html=`<strong>總量 ${kg} ${unit===1000?'kg':'g'} 用量：</strong><br>`;
  Object.entries(currentFormula).sort((a,b)=>b[1]-a[1]).forEach(([p,v])=>{
    const g=(v/100*totalG).toFixed(1);
    html+=`${PNAMES[p]||p}（${p}）：<strong>${g} g</strong><br>`;
    const el=document.getElementById('g-'+p.replace(/[^a-z0-9]/gi,'_'));
    if(el)el.textContent=g+'g';
  });
  document.getElementById('batch-result').innerHTML=html;
}
 
function updateFbFormula(formula){
  const text=Object.entries(formula).sort((a,b)=>b[1]-a[1])
    .map(([p,v])=>`${PNAMES[p]||p}（${p}）：${v.toFixed(1)}%`).join('　　');
  document.getElementById('fb-formula-display').textContent=text||'—';
}
 
function calcDE(){
  const tL=parseFloat(document.getElementById('fb-tL').value);
  const ta=parseFloat(document.getElementById('fb-ta').value);
  const tb=parseFloat(document.getElementById('fb-tb').value);
  const aL=parseFloat(document.getElementById('fb-aL').value);
  const aa=parseFloat(document.getElementById('fb-aa').value);
  const ab=parseFloat(document.getElementById('fb-ab').value);
  if([tL,ta,tb,aL,aa,ab].some(isNaN))return;
  const dE=deltaE2000(tL,ta,tb,aL,aa,ab);
  const el=document.getElementById('de-val');
  el.textContent=dE.toFixed(2);
  el.className='de-val '+(dE<2?'de-good':dE<4?'de-warn':'de-bad');
  let msg='';
  if(dE<1)msg='✅ 極佳 — 人眼幾乎無法分辨';
  else if(dE<2)msg='✅ 優良 — 仔細比對才有差異';
  else if(dE<3.5)msg='⚠️ 可接受 — 一般工程標準';
  else if(dE<5)msg='⚠️ 偏差 — 建議微調';
  else msg='❌ 差異明顯 — 需重新調色';
  document.getElementById('de-info').innerHTML=`ΔE2000 色差<br><small style="color:#999">${msg}</small>`;
  document.getElementById('de-box').style.display='flex';
}
 
async function saveFeedback(){
  const tL=parseFloat(document.getElementById('fb-tL').value);
  const ta=parseFloat(document.getElementById('fb-ta').value);
  const tb=parseFloat(document.getElementById('fb-tb').value);
  const tluster=parseFloat(document.getElementById('fb-tluster').value);
  const aL=parseFloat(document.getElementById('fb-aL').value);
  const aa=parseFloat(document.getElementById('fb-aa').value);
  const ab=parseFloat(document.getElementById('fb-ab').value);
  const aluster=parseFloat(document.getElementById('fb-aluster').value);
  const note=document.getElementById('fb-note').value;
 
  if([tL,ta,tb,aL,aa,ab].some(isNaN)){showStatus('fb-status','請填入目標與實測 LAB 值');return;}
  const formula=currentFormula||{};
 
  try{
    const res=await fetch('/api/feedback',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        target:{L:tL,a:ta,b:tb,luster:isNaN(tluster)?0:tluster},
        actual:{L:aL,a:aa,b:ab,luster:isNaN(aluster)?0:aluster},
        formula, note
      })
    });
    const data=await res.json();
    if(data.success){
      document.getElementById('fb-success').style.display='block';
      document.getElementById('data-badge').textContent=`訓練資料 ${data.total_feedback} 筆回填`;
      hideStatus('fb-status');
    }else{showStatus('fb-status',data.error);}
  }catch(e){showStatus('fb-status','連線失敗');}
}
 
async function loadHistory(){
  try{
    const res=await fetch('/api/history');
    const data=await res.json();
    if(!data.success){document.getElementById('history-content').innerHTML='<div style="color:red;padding:20px;">載入失敗</div>';return;}
    if(!data.records.length){document.getElementById('history-content').innerHTML='<div style="text-align:center;color:var(--muted);padding:40px;">尚無記錄</div>';return;}
    let html=`<table class="history-table"><thead><tr>
      <th>顏色</th><th>目標 LAB</th><th>光澤</th><th>ΔE2000</th><th>日期</th><th>備注</th>
    </tr></thead><tbody>`;
    data.records.forEach(r=>{
      const deClass=r.delta_e<2?'chip-good':r.delta_e<4?'chip-warn':'chip-bad';
      html+=`<tr>
        <td><span class="color-dot" style="background:${lab2rgb(r.target.L,r.target.a,r.target.b)}"></span></td>
        <td>L${r.target.L.toFixed(1)} a${r.target.a.toFixed(1)} b${r.target.b.toFixed(1)}</td>
        <td>${r.target.luster||'—'}</td>
        <td><span class="de-chip ${deClass}">ΔE ${r.delta_e.toFixed(2)}</span></td>
        <td>${r.date}</td>
        <td style="color:var(--muted)">${r.note||'—'}</td>
      </tr>`;
    });
    html+=`</tbody></table>`;
    document.getElementById('history-content').innerHTML=html;
  }catch(e){document.getElementById('history-content').innerHTML='<div style="color:red;padding:20px;">連線失敗</div>';}
}
 
async function loadStats(){
  try{
    const res=await fetch('/api/stats');
    const data=await res.json();
    if(data.success){
      document.getElementById('s-original').textContent=data.original_data;
      document.getElementById('s-feedback').textContent=data.feedback_count;
      document.getElementById('s-total').textContent=data.total_data;
      document.getElementById('s-good').textContent=data.good_rate?data.good_rate+'%':'—';
      document.getElementById('data-badge').textContent=`訓練資料 ${data.total_data} 筆`;
    }
  }catch(e){}
}
 
let trainPoll=null;
async function startTraining(){
  const btn=document.getElementById('btn-train');
  btn.disabled=true; btn.textContent='訓練中...';
  try{
    const res=await fetch('/api/training/start',{method:'POST'});
    const data=await res.json();
    if(data.success){
      trainPoll=setInterval(checkTrainingStatus,2000);
    }
  }catch(e){btn.disabled=false;btn.textContent='開始重新訓練';}
}
 
async function checkTrainingStatus(){
  try{
    const res=await fetch('/api/training/status');
    const data=await res.json();
    document.getElementById('train-bar').style.width=data.progress+'%';
    document.getElementById('train-msg').textContent=data.message;
    if(data.status==='done'||data.status==='error'){
      clearInterval(trainPoll);
      document.getElementById('btn-train').disabled=false;
      document.getElementById('btn-train').innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>開始重新訓練';
      loadStats();
    }
  }catch(e){}
}
 
function showStatus(id,msg){const el=document.getElementById(id);el.textContent=msg;el.style.display='block';}
function hideStatus(id){document.getElementById(id).style.display='none';}
 
// 初始載入
loadStats();
['L','a','b','luster'].forEach(id=>{
  document.getElementById(id)?.addEventListener('keydown',e=>{if(e.key==='Enter')predict();});
});
</script>
</body>
</html>
"""
 
if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
