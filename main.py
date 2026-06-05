import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==========================================
# 設定區
# ==========================================
TRAIN_FILE = 'train_FD001.txt'
TEST_FILE = 'test_FD001.txt'
RUL_FILE = 'RUL_FD001.txt'

MAX_RUL = 125           
WARNING_THRESHOLD = 30  
CRITICAL_THRESHOLD = 15 

COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

# ==========================================
# Step 1: 讀取資料與預處理
# ==========================================
def preprocess_train_data(filepath):
    print("\n" + "="*50)
    print("處理訓練集資料")
    print("="*50)
    
    if not os.path.exists(filepath):
        print(f"找不到檔案 {filepath}。")
        return None, None
        
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)
    print(f"成功讀取訓練資料，維度: {df.shape}")
    
    sensors = [f's_{i}' for i in range(1, 22)]
    useless_features = [col for col in sensors if df[col].std() == 0]
    df_filtered = df.drop(columns=useless_features)
    print(f"剔除無效感測器: {useless_features}")
    
    max_cycles = df_filtered.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']
    df_merged = df_filtered.merge(max_cycles, on='unit_nr', how='left')
    df_merged['RUL'] = df_merged['max_cycle'] - df_merged['time_cycles']
    df_merged['RUL'] = df_merged['RUL'].clip(upper=MAX_RUL)
    
    def classify_risk(rul):
        if rul <= CRITICAL_THRESHOLD: return 2
        elif rul <= WARNING_THRESHOLD: return 1
        else: return 0
        
    df_merged['Risk_Class'] = df_merged['RUL'].apply(classify_risk)
    df_final = df_merged.drop(columns=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 'RUL', 'max_cycle'])
    
    return df_final, useless_features

# ==========================================
# Step 2: 模型訓練與內部驗證
# ==========================================
def train_and_evaluate_models(df_final):
    print("\n" + "="*50)
    print("訓練 Logistic Regression 與 Random Forest")
    print("="*50)
    
    X = df_final.drop(columns=['Risk_Class'])
    y = df_final['Risk_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Logistic Regression ---
    print("正在訓練 Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # --- Random Forest ---
    print("正在訓練 Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # 繪製圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'], yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[0].set_title('LR Internal Validation Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'], yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[1].set_title('RF Internal Validation Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), hue='Feature', palette='viridis', legend=False, ax=axes[2])
    axes[2].set_title('Top 10 Sensor Features')
    axes[2].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig('01_Internal_Validation_Results.png', dpi=300, bbox_inches='tight')
    print("內部驗證圖表已儲存為 '01_Internal_Validation_Results.png'")
    # plt.show()
    
    # 回傳 LR模型、RF模型與 Scaler，供盲測使用
    return lr_model, rf_model, scaler

# ==========================================
# Step 3: NASA 官方盲測
# ==========================================
# 接收兩個模型進入盲測函數
def evaluate_official_blind_test(lr_model, rf_model, scaler, useless_features):
    print("\n" + "="*50)
    print("啟動官方盲測驗證")
    print("="*50)
    
    if not (os.path.exists(TEST_FILE) and os.path.exists(RUL_FILE)):
        print(f"找不到盲測檔案 {TEST_FILE} 或 {RUL_FILE}。")
        return
        
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', header=None, names=COLUMNS)
    df_rul = pd.read_csv(RUL_FILE, sep=r'\s+', header=None, names=['True_RUL'])
    
    df_test_last = df_test.groupby('unit_nr').last().reset_index()
    X_blind = df_test_last.drop(columns=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + useless_features)
    
    def classify_risk(rul):
        if rul <= CRITICAL_THRESHOLD: return 2
        elif rul <= WARNING_THRESHOLD: return 1
        else: return 0
        
    y_blind_true = df_rul['True_RUL'].apply(classify_risk)
    X_blind_scaled = scaler.transform(X_blind)
    
    # 模型預測
    y_blind_pred_lr = lr_model.predict(X_blind_scaled)
    y_blind_pred_rf = rf_model.predict(X_blind_scaled)
    
    print("\n=== Logistic Regression 官方盲測報告 ===")
    print(classification_report(y_blind_true, y_blind_pred_lr, target_names=['Healthy(0)', 'Warning(1)', 'Critical(2)']))
    
    print("\n=== Random Forest 官方盲測報告 ===")
    print(classification_report(y_blind_true, y_blind_pred_rf, target_names=['Healthy(0)', 'Warning(1)', 'Critical(2)']))
    
    # 繪製盲測圖表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(confusion_matrix(y_blind_true, y_blind_pred_lr), annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'], yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[0].set_title('LR Blind Test Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(confusion_matrix(y_blind_true, y_blind_pred_rf), annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'], yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[1].set_title('RF Blind Test Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    importance_df = pd.DataFrame({'Feature': X_blind.columns, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), hue='Feature', palette='viridis', legend=False, ax=axes[2])
    axes[2].set_title('Top 10 Sensor Features')
    axes[2].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig('02_Official_Blind_Test_Results.png', dpi=300, bbox_inches='tight')
    print("盲測圖表已自動儲存為 '02_Official_Blind_Test_Results.png'")
    # plt.show()
    print("預測性維護專案執行完畢")

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    df_processed, useless_cols = preprocess_train_data(TRAIN_FILE)
    
    if df_processed is not None:
        lr_model, rf_model, fitted_scaler = train_and_evaluate_models(df_processed)
        evaluate_official_blind_test(lr_model, rf_model, fitted_scaler, useless_cols)