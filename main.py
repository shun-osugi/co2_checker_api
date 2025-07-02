import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# --------------------------------------------------------------------------
# 1. FastAPIアプリケーションのインスタンス化
# --------------------------------------------------------------------------
app = FastAPI(
    title="CO2濃度予測API",
    description="過去6時間（72点）のタイムスタンプ付きセンサーデータから15分後のCO2濃度を予測します。",
    version="1.1", # バージョンアップ
)

# --------------------------------------------------------------------------
# 2. モデルとスケーラーのロード
# --------------------------------------------------------------------------
try:
    model = tf.keras.models.load_model('best_co2_model.keras')
    scaler = joblib.load('co2_scaler.gz')
    print("モデルとスケーラーの読み込みに成功しました。")
except Exception as e:
    print(f"エラー: モデルまたはスケーラーの読み込みに失敗しました: {e}")
    model = None
    scaler = None

# --------------------------------------------------------------------------
# 3. リクエストとレスポンスの型定義 (Pydantic) - ★修正点
# --------------------------------------------------------------------------
# 入力データ1点分に `timestamp` を追加
class SensorDataPoint(BaseModel):
    timestamp: datetime  # タイムスタンプを受け取る
    co2: float
    temperature: float
    humidity: float

# APIへのリクエストボディ全体の型を定義
class PredictionRequest(BaseModel):
    data: List[SensorDataPoint] = Field(..., min_length=72, max_length=72)

# APIからのレスポンスボディの型を定義
class PredictionResponse(BaseModel):
    predicted_co2: float

# --------------------------------------------------------------------------
# 4. 予測エンドポイントの作成 - ★修正点
# --------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="モデルまたはスケーラーがロードされていません。")

    # --- Step 1: 入力データをDataFrameに変換 ---
    input_data = [{'timestamp': p.timestamp, 'CO2': p.co2, '温度': p.temperature, '湿度': p.humidity} for p in request.data]
    input_df = pd.DataFrame(input_data)

    # --- Step 2: timestampをインデックスに設定 ---
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df.set_index('timestamp', inplace=True)

    # --- Step 3: 時間に関する特徴量を作成 ---
    # クライアントから受け取ったタイムスタンプを元に特徴量を作成
    input_df['hour_sin'] = np.sin(2 * np.pi * input_df.index.hour / 24)
    input_df['hour_cos'] = np.cos(2 * np.pi * input_df.index.hour / 24)
    input_df['dayofweek_sin'] = np.sin(2 * np.pi * input_df.index.dayofweek / 7)
    input_df['dayofweek_cos'] = np.cos(2 * np.pi * input_df.index.dayofweek / 7)

    # --- Step 4: データをスケーリング ---
    feature_order = ['CO2', '温度', '湿度', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    input_df_ordered = input_df[feature_order]
    scaled_data = scaler.transform(input_df_ordered)

    # --- Step 5: モデルの入力形式に整形 ---
    reshaped_data = np.expand_dims(scaled_data, axis=0)

    # --- Step 6: 予測の実行 ---
    scaled_prediction = model.predict(reshaped_data)

    # --- Step 7: 予測結果を元のスケールに戻す ---
    def inverse_transform_co2(val):
        dummy_array = np.zeros((1, len(feature_order)))
        dummy_array[:, 0] = val
        return scaler.inverse_transform(dummy_array)[0, 0]

    final_prediction = inverse_transform_co2(scaled_prediction)

    return PredictionResponse(predicted_co2=round(final_prediction, 2))