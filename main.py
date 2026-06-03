import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 設定區
# ==========================================
DATA_FILE = 'train_FD001.txt' 
MAX_RUL = 125          # 分段線性 RUL 的最大健康閾值
WARNING_THRESHOLD = 30 # 進入警告狀態的 RUL 閾值 (w0)
CRITICAL_THRESHOLD = 15 # 進入危險狀態的 RUL 閾值 (w1)

# ==========================================
# Step 1: 讀取資料與欄位命名
# ==========================================
def load_data(filepath):
    """讀取 CMAPSS 資料並賦予正確的欄位名稱"""
    # 原始資料沒有標頭，我們需要自行定義
    columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    sensors = [f's_{i}' for i in range(1, 22)]
    columns.extend(sensors)
    
    # 讀取以空格分隔的資料
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns)
        print(f"✅ 成功讀取資料，維度: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {filepath}。請確認 NASA 資料集已放置於正確路徑。")
        return None

# ==========================================
# Step 2: Data Preprocessing 與特徵工程
# ==========================================
def preprocess_data(df):
    """剔除無效感測器、計算 RUL、並轉換為風險標籤"""
    # 1. 剔除常數感測器 (標準差為 0)
    sensors = [f's_{i}' for i in range(1, 22)]
    useless_features = [col for col in sensors if df[col].std() == 0]
    df_filtered = df.drop(columns=useless_features)
    print(f"剔除無效感測器: {useless_features}")
    
    # 2. 計算分段線性 RUL
    max_cycles = df_filtered.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max_cycle']
    df_merged = df_filtered.merge(max_cycles, on='unit_nr', how='left')
    df_merged['RUL'] = df_merged['max_cycle'] - df_merged['time_cycles']
    df_merged['RUL'] = df_merged['RUL'].clip(upper=MAX_RUL)
    df_merged = df_merged.drop(columns=['max_cycle'])
    
    # 3. 轉換為風險等級標籤 (0: Healthy, 1: Warning, 2: Critical)
    def classify_risk(rul):
        if rul <= CRITICAL_THRESHOLD: return 2
        elif rul <= WARNING_THRESHOLD: return 1
        else: return 0
        
    df_merged['Risk_Class'] = df_merged['RUL'].apply(classify_risk)
    
    # 4. 移除不再需要的非特徵欄位
    df_final = df_merged.drop(columns=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 'RUL'])
    
    return df_final

# ==========================================
# Step 3: 模型訓練與評估視覺化
# ==========================================
def train_and_evaluate(df_final):
    # 切分特徵 (X) 與標籤 (y)
    X = df_final.drop(columns=['Risk_Class'])
    y = df_final['Risk_Class']
    
    # Train-Test Split (確保類別比例一致)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 特徵標準化 (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ----------------------------------------
    # Model 1: Logistic Regression
    # ----------------------------------------
    print("\n正在訓練 Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    print("\n=== Logistic Regression 報告 ===")
    print(classification_report(y_test, y_pred_lr))
    
    # ----------------------------------------
    # Model 2: Random Forest
    # ----------------------------------------
    print("正在訓練 Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train) # 為了公平比較，這裡也使用 scaled 過的資料
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    print("\n=== Random Forest 報告 ===")
    print(classification_report(y_test, y_pred_rf))
    
    # ----------------------------------------
    # 視覺化輸出 (confusion matrix與特徵重要性)
    # ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 圖 1: LR confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'],
                yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[0].set_title('Logistic Regression Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # 圖 2: RF confusion matrix
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'],
                yticklabels=['Healthy(0)', 'Warning(1)', 'Critical(2)'])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # 圖 3: 特徵重要性 (Random Forest)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis', ax=axes[2])
    axes[2].set_title('Top 10 Sensor Features (Random Forest)')
    axes[2].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    print("啟動預測性維護分析腳本...")
    df_raw = load_data(DATA_FILE)
    if df_raw is not None:
        df_processed = preprocess_data(df_raw)
        train_and_evaluate(df_processed)
        print("分析完成！圖表已生成。")