import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix, precision_recall_curve)
import warnings
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Pengaturan Judul Utama Aplikasi Web
st.set_page_config(page_title="Online Shoppers Intention Dashboard", layout="wide")
st.title("📊 Online Shoppers Intention - Final Project Dashboard")
st.markdown("Aplikasi web ini menampilkan analisis data dan hasil modeling machine learning dari Google Colab Mas Imam.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA (Menggunakan Cache agar tidak reload terus menerus)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Pastikan file 'online_shoppers_intention.csv' berada satu folder dengan script app.py
    df = pd.read_csv('online_shoppers_intention.csv')
    return df

try:
    df = load_data()
    st.success("✅ Dataset 'online_shoppers_intention.csv' berhasil di-load!")
except FileNotFoundError:
    st.error("❌ File 'online_shoppers_intention.csv' tidak ditemukan. Harap unggah file tersebut di repository GitHub Anda bersama app.py.")
    st.stop()

# Membuat Menu Navigasi di Sidebar
menu = st.sidebar.selectbox("Pilih Halaman / Analisis:", 
                            ["Informasi Data", "Analisis Univariat", "Analisis Bivariat & Multivariat", "Modeling & Evaluasi"])

# ──────────────────────────────────────────────────────────────────────────────
# HALAMAN 1: INFORMASI DATA
# ──────────────────────────────────────────────────────────────────────────────
if menu == "Informasi Data":
    st.header("📋 Ringkasan & Struktur Dataset")
    
    st.subheader("5 Data Pertama (Head)")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informasi Dataset")
        # Menangkap info dataframe ke string agar bisa ditampilkan di Streamlit
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    with col2:
        st.subheader("Pengecekan Missing Values")
        mv = df.isnull().sum()
        if mv.any():
            st.dataframe(mv[mv > 0])
        else:
            st.info("Tidak ada missing values dalam dataset.")

# ──────────────────────────────────────────────────────────────────────────────
# HALAMAN 2: ANALISIS UNIVARIAT
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "Analisis Univariat":
    st.header("📈 Analisis Univariat")
    
    # 3.1 Distribusi Target (Revenue)
    st.subheader("3.1 Distribusi Kelas Target (Revenue)")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.countplot(x='Revenue', data=df, palette='viridis', ax=ax1)
    ax1.set_title('Distribusi Kelas Target (Revenue)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Transaksi Terjadi (Revenue)')
    ax1.set_ylabel('Jumlah Pengunjung')
    total = len(df)
    for p in ax1.patches:
        pct = f'{100 * p.get_height() / total:.1f}%'
        ax1.annotate(pct, (p.get_x() + p.get_width()/2 - 0.1,
                          p.get_y() + p.get_height() + 100), size=12)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # 3.2 Distribusi Fitur Numerik
    st.subheader("3.2 Distribusi Fitur Perilaku Pengguna")
    numerical_features = ['BounceRates', 'ExitRates', 'PageValues', 'Administrative_Duration']
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Distribusi Fitur Perilaku Pengguna', fontsize=16, fontweight='bold')
    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], kde=True, ax=axes2[i//2, i%2], color='coral', bins=30)
        axes2[i//2, i%2].set_title(f'Distribusi {col}')
    plt.tight_layout()
    st.pyplot(fig2)

# ──────────────────────────────────────────────────────────────────────────────
# HALAMAN 3: BIVARIAT & MULTIVARIAT
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "Analisis Bivariat & Multivariat":
    st.header("📉 Analisis Hubungan Antar Variabel")
    
    st.subheader("4. Pengaruh Perilaku Halaman terhadap Transaksi (Revenue)")
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle('Pengaruh Perilaku Halaman terhadap Transaksi (Revenue)', fontsize=16, fontweight='bold')
    sns.boxplot(x='Revenue', y='PageValues',  data=df, ax=axes3[0], palette='Set2')
    axes3[0].set_title('Page Values vs Revenue')
    sns.boxplot(x='Revenue', y='BounceRates', data=df, ax=axes3[1], palette='Set2')
    axes3[1].set_title('Bounce Rates vs Revenue')
    sns.boxplot(x='Revenue', y='ExitRates',   data=df, ax=axes3[2], palette='Set2')
    axes3[2].set_title('Exit Rates vs Revenue')
    plt.tight_layout()
    st.pyplot(fig3)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Niat Beli Berdasarkan Tipe Pengunjung")
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        sns.countplot(x='VisitorType', hue='Revenue', data=df, palette='magma', ax=ax4)
        ax4.set_title('Niat Beli Berdasarkan Tipe Pengunjung', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Tipe Pengunjung')
        ax4.set_ylabel('Jumlah')
        st.pyplot(fig4)
        
    with col2:
        st.subheader("5. Peta Korelasi Fitur (Multivariat)")
        df_corr = df.copy()
        df_corr['Revenue'] = df_corr['Revenue'].astype(int)
        numeric_cols = df_corr.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr()

        fig5, ax5 = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5,
                    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)), ax=ax5)
        ax5.set_title('Peta Korelasi Fitur Dataset Online Shoppers Intention', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig5)

# ──────────────────────────────────────────────────────────────────────────────
# HALAMAN 4: MODELING & EVALUASI
# ──────────────────────────────────────────────────────────────────────────────
elif menu == "Modeling & Evaluasi":
    st.header("🤖 Preprocessing, Tuning Model & Hasil Evaluasi")
    
    # Membungkus proses modeling berat ke fungsi cache resource agar web responsif
    @st.cache_resource
    def run_machine_learning_pipeline(_df_input):
        # --- Preprocessing & Encoding ---
        df_model = _df_input.copy()
        df_model['Revenue'] = df_model['Revenue'].astype(int)
        df_model['Weekend'] = df_model['Weekend'].astype(int)

        le = LabelEncoder()
        for col in ['Month', 'VisitorType']:
            df_model[col] = le.fit_transform(df_model[col])

        X = df_model.drop('Revenue', axis=1)
        y = df_model['Revenue']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # ADASYN Imbalance Handling
        adasyn = ADASYN(random_state=42, n_neighbors=5, sampling_strategy=0.8)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

        # StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- OPTUNA TUNING (Dibatasi n_trials lebih kecil demi efisiensi server web Streamlit) ---
        # Mengurangi n_trials dari 30 ke 5 agar waktu deploy tidak meledak/timeout di Streamlit Share
        
        # 1. LightGBM Tuning
        def objective_lgb(trial):
            param = {
                'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 60),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'random_state': 42
            }
            model = lgb.LGBMClassifier(**param)
            return cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='roc_auc').mean()
        
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(objective_lgb, n_trials=5)
        
        # 2. CatBoost Tuning
        def objective_cb(trial):
            param = {
                'iterations': trial.suggest_int('iterations', 100, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 7),
                'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': False, 'random_seed': 42
            }
            model = CatBoostClassifier(**param)
            return cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='roc_auc').mean()
            
        study_cb = optuna.create_study(direction='maximize')
        study_cb.optimize(objective_cb, n_trials=5)

        # 3. XGBoost Tuning
        def objective_xgb(trial):
            param = {
                'objective': 'binary:logistic', 'eval_metric': 'auc',
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 7),
                'random_state': 42
            }
            model = xgb.XGBClassifier(**param)
            return cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='roc_auc').mean()
            
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(objective_xgb, n_trials=5)

        # Fit Final Model
        final_lgb = lgb.LGBMClassifier(**study_lgb.best_params, objective='binary', random_state=42, verbosity=-1)
        final_cb  = CatBoostClassifier(**study_cb.best_params, loss_function='Logloss', eval_metric='AUC', verbose=False, random_seed=42)
        final_xgb = xgb.XGBClassifier(**study_xgb.best_params, objective='binary:logistic', eval_metric='auc', random_state=42)

        final_lgb.fit(X_train_res, y_train_res)
        final_cb.fit(X_train_res, y_train_res)
        final_xgb.fit(X_train_res, y_train_res)

        # Predictions
        results = {
            'X': X, 'y': y, 'y_test': y_test,
            'y_pred_lgb': final_lgb.predict(X_test), 'y_prob_lgb': final_lgb.predict_proba(X_test)[:, 1],
            'y_pred_cb': final_cb.predict(X_test), 'y_prob_cb': final_cb.predict_proba(X_test)[:, 1],
            'y_pred_xgb': final_xgb.predict(X_test), 'y_prob_xgb': final_xgb.predict_proba(X_test)[:, 1],
            'final_lgb': final_lgb, 'final_cb': final_cb, 'final_xgb': final_xgb
        }
        return results

    # Menjalankan fungsi ML pipeline
    st.info("Sedang memproses pelatihan model & tuning Optuna (ini hanya berjalan sekali saat halaman dibuka)...")
    res = run_machine_learning_pipeline(df)
    st.success("🤖 Pelatihan Model Selesai!")

    # 10.1 Tampilkan Classification Report
    st.subheader("📋 10.1 Hasil Classification Report")
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown("**LightGBM Model**")
        st.text(classification_report(res['y_test'], res['y_pred_lgb']))
        st.write(f"ROC-AUC: {roc_auc_score(res['y_test'], res['y_prob_lgb']):.4f}")
    with colB:
        st.markdown("**CatBoost Model**")
        st.text(classification_report(res['y_test'], res['y_pred_cb']))
        st.write(f"ROC-AUC: {roc_auc_score(res['y_test'], res['y_prob_cb']):.4f}")
    with colC:
        st.markdown("**XGBoost Model**")
        st.text(classification_report(res['y_test'], res['y_pred_xgb']))
        st.write(f"ROC-AUC: {roc_auc_score(res['y_test'], res['y_prob_xgb']):.4f}")

    # 10.2 Confusion Matrix Visualisasi
    st.subheader("10.2 Confusion Matrix Perbandingan")
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 5))
    fig6.suptitle('Confusion Matrix: LightGBM vs CatBoost vs XGBoost', fontsize=16, fontweight='bold')
    
    for ax, name, y_p, y_prb, cmap in zip(axes6,
            ['LightGBM', 'CatBoost', 'XGBoost'],
            [res['y_pred_lgb'], res['y_pred_cb'], res['y_pred_xgb']],
            [res['y_prob_lgb'], res['y_prob_cb'], res['y_prob_xgb']],
            ['Blues', 'Oranges', 'Greens']):
        cm = confusion_matrix(res['y_test'], y_p)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Pred 0','Pred 1'], yticklabels=['Aktual 0','Aktual 1'])
        ax.set_title(f'{name}\nAcc={((cm[0,0]+cm[1,1])/cm.sum()):.3f} | AUC={roc_auc_score(res['y_test'], y_prb):.4f}')
    plt.tight_layout()
    st.pyplot(fig6)

    # 10.3 Kurva ROC
    st.subheader("10.3 Kurva ROC")
    fpr_lgb, tpr_lgb, _ = roc_curve(res['y_test'], res['y_prob_lgb'])
    fpr_cb,  tpr_cb,  _ = roc_curve(res['y_test'], res['y_prob_cb'])
    fpr_xgb, tpr_xgb, _ = roc_curve(res['y_test'], res['y_prob_xgb'])

    fig7, ax7 = plt.subplots(figsize=(9, 6))
    ax7.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={roc_auc_score(res['y_test'], res['y_prob_lgb']):.4f})', color='blue')
    ax7.plot(fpr_cb,  tpr_cb,  label=f'CatBoost (AUC={roc_auc_score(res['y_test'], res['y_prob_cb']):.4f})',  color='darkorange')
    ax7.plot(fpr_xgb, tpr_xgb, label=f'XGBoost  (AUC={roc_auc_score(res['y_test'], res['y_prob_xgb']):.4f})', color='green')
    ax7.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax7.set_title('Kurva ROC: LightGBM vs CatBoost vs XGBoost', fontsize=14, fontweight='bold')
    ax7.set_xlabel('False Positive Rate')
    ax7.set_ylabel('True Positive Rate')
    ax7.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig7)

    # 10.4 Feature Importance (LightGBM)
    st.subheader("10.4 Feature Importance (LightGBM — Model Terbaik)")
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    feat_imp = pd.Series(res['final_lgb'].feature_importances_, index=res['X'].columns).sort_values(ascending=False)
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis', ax=ax8)
    ax8.set_title('Feature Importance – LightGBM', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig8)
