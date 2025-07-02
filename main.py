import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Header, HTTPException
from typing import List
from datetime import datetime

# --------------------------------------------------------------------------
# アプリケーションとモデルの準備
# --------------------------------------------------------------------------
app = FastAPI(title="CO2濃度予測API")

try:
    model = tf.keras.models.load_model('best_co2_model.keras')
    scaler = joblib.load('co2_scaler.gz')
    print("モデルとスケーラーの読み込みに成功しました。")
except Exception as e:
    print(f"エラー: モデルまたはスケーラーの読み込みに失敗しました: {e}")
    model, scaler = None, None

TRIGGER_SECRET = os.environ.get("TRIGGER_SECRET_KEY")
CO2_THRESHOLD = 1000  # 通知の閾値を設定 (例: 1000ppm)

# --------------------------------------------------------------------------
# 実際のAirocoデータ取得部分（将来的に実装）
# --------------------------------------------------------------------------
def get_latest_airoco_data() -> List[dict]:
    # !!! この部分は、将来的にAirocoからデータを取得する実際の処理に置き換えます !!!
    # 今回は、テストのためにダミーデータを生成します。
    print("Airocoデータ取得中（現在はダミーデータを生成）...")
    timestamps = pd.to_datetime([pd.Timestamp.now(tz='Asia/Tokyo') - pd.Timedelta(minutes=x*5) for x in range(72)])
    timestamps = timestamps.sort_values() # 時系列が昇順になるように並べ替え
    
    dummy_data = [
        {"timestamp": ts, "co2": 450 + (i*2) + np.random.randint(-5, 5), "temperature": 25.0 + np.random.rand(), "humidity": 60.0 + np.random.rand()}
        for i, ts in enumerate(timestamps)
    ]
    return dummy_data

# --------------------------------------------------------------------------
# スケーリング逆変換の補助関数
# --------------------------------------------------------------------------
def inverse_transform_co2(scaled_value, original_scaler, num_features):
    dummy_array = np.zeros((1, num_features))
    dummy_array[:, 0] = scaled_value
    inversed_array = original_scaler.inverse_transform(dummy_array)
    return inversed_array[0, 0]

# --------------------------------------------------------------------------
# GitHub Actionsから呼び出されるトリガーエンドポイント
# --------------------------------------------------------------------------
@app.post("/trigger-prediction")
async def trigger_prediction(x_trigger_secret: str = Header(None)):
    print("トリガーエンドポイントへのリクエストを受信しました。")
    if not all([model, scaler, TRIGGER_SECRET]):
        raise HTTPException(status_code=500, detail="サーバー設定が不完全です。")
    if x_trigger_secret != TRIGGER_SECRET:
        raise HTTPException(status_code=401, detail="認証に失敗しました。")

    try:
        # Step 1: 最新のデータを取得
        latest_data = get_latest_airoco_data()
        
        # Step 2: データを前処理
        input_df = pd.DataFrame(latest_data)
        input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
        input_df.set_index('timestamp', inplace=True)
        input_df.rename(columns={'co2': 'CO2', 'temperature': '温度', 'humidity': '湿度'}, inplace=True)

        input_df['hour_sin'] = np.sin(2 * np.pi * input_df.index.hour / 24)
        input_df['hour_cos'] = np.cos(2 * np.pi * input_df.index.hour / 24)
        input_df['dayofweek_sin'] = np.sin(2 * np.pi * input_df.index.dayofweek / 7)
        input_df['dayofweek_cos'] = np.cos(2 * np.pi * input_df.index.dayofweek / 7)
        
        feature_order = ['CO2', '温度', '湿度', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        input_df_ordered = input_df[feature_order]
        scaled_data = scaler.transform(input_df_ordered)
        reshaped_data = np.expand_dims(scaled_data, axis=0)

        # Step 3: 予測を実行
        scaled_prediction = model.predict(reshaped_data)
        final_prediction = inverse_transform_co2(scaled_prediction, scaler, len(feature_order))
        print(f"✅ 予測成功。15分後のCO2濃度予測値: {final_prediction:.2f} ppm")

        # Step 4: 閾値と比較
        if final_prediction > CO2_THRESHOLD:
            print(f"⚠️ 警報: 予測値({final_prediction:.2f} ppm)が閾値({CO2_THRESHOLD} ppm)を超えました。")
            # --- ここで将来的にFirebaseから宛先リストを取得し、メールを送信する ---
        else:
            print(f"INFO: 予測値({final_prediction:.2f} ppm)は正常範囲内です。")

        return {"message": "予測処理が正常に完了しました。", "predicted_co2": final_prediction}

    except Exception as e:
        print(f"❌ 予測処理中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")