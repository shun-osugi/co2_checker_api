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
# main.py の trigger_prediction 関数を、このデバッグ版に置き換えてください

@app.post("/trigger-prediction")
async def trigger_prediction(x_trigger_secret: str = Header(None)):
    print("トリガーエンドポイントへのリクエストを受信しました。")

    # --- ここからデバッグ用コード ---
    print("--- DEBUGGING START ---")
    
    # Renderに設定されている環境変数を取得
    expected_secret = os.environ.get("TRIGGER_SECRET_KEY")
    # GitHub Actionsから受け取ったヘッダーの値
    received_secret = x_trigger_secret

    if expected_secret:
        print(f"期待しているキー (Render側): 長さ={len(expected_secret)}, 最初='{expected_secret[0]}', 最後='{expected_secret[-1]}'")
    else:
        print("期待しているキー (Render側): 環境変数が設定されていません (None)")

    if received_secret:
        print(f"受け取ったキー (GitHub側): 長さ={len(received_secret)}, 最初='{received_secret[0]}', 最後='{received_secret[-1]}'")
    else:
        print("受け取ったキー (GitHub側): ヘッダーがありません (None)")
    
    if expected_secret and received_secret and expected_secret.strip() == received_secret.strip():
        print("キーは一致しました。")
    else:
        print("キーが一致しませんでした。")
    
    print("--- DEBUGGING END ---")
    # --- ここまでデバッグ用コード ---

    # 元の認証ロジック
    if not all([model, scaler, expected_secret]):
        raise HTTPException(status_code=500, detail="サーバー設定が不完全です。")
    if received_secret != expected_secret:
        # このエラーが現在発生しています
        raise HTTPException(status_code=401, detail="認証に失敗しました。")

    # (この後の予測処理は同じです)
    try:
        latest_data = get_latest_airoco_data()
        input_df = pd.DataFrame(latest_data)
        # ... (以降の予測処理は元のコードのままなので省略) ...
        # (この部分は元のコードから消さないでください)
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
        scaled_prediction = model.predict(reshaped_data)
        final_prediction = inverse_transform_co2(scaled_prediction, scaler, len(feature_order))
        print(f"✅ 予測成功。15分後のCO2濃度予測値: {final_prediction:.2f} ppm")
        if final_prediction > CO2_THRESHOLD:
            print(f"⚠️ 警報: 予測値({final_prediction:.2f} ppm)が閾値({CO2_THRESHOLD} ppm)を超えました。")
        else:
            print(f"INFO: 予測値({final_prediction:.2f} ppm)は正常範囲内です。")
        return {"message": "予測処理が正常に完了しました。", "predicted_co2": final_prediction}
    except Exception as e:
        print(f"❌ 予測処理中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")