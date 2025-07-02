import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Header, HTTPException
from datetime import datetime

# --------------------------------------------------------------------------
# 1. アプリケーションの準備
# --------------------------------------------------------------------------
app = FastAPI(
    title="CO2濃度予測API (TFLite版)",
    description="GitHub Actionsからのトリガーで、15分後のCO2濃度を予測します。",
    version="2.0",
)

# --------------------------------------------------------------------------
# 2. TFLiteモデルとスケーラーのロード
# --------------------------------------------------------------------------
try:
    # TFLiteモデルをロードし、インタプリタを準備
    interpreter = tf.lite.Interpreter(model_path='co2_model.tflite')
    interpreter.allocate_tensors()  # モデルのためにメモリを確保
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLiteモデルの読み込みに成功しました。")

    # スケーラーをロード
    scaler = joblib.load('co2_scaler.gz')
    print("✅ スケーラーの読み込みに成功しました。")

except Exception as e:
    print(f"❌ エラー: モデルまたはスケーラーの読み込みに失敗しました: {e}")
    interpreter, scaler = None, None

# --------------------------------------------------------------------------
# 3. 環境変数から秘密のキーを取得
# --------------------------------------------------------------------------
TRIGGER_SECRET = os.environ.get("TRIGGER_SECRET_KEY")
CO2_THRESHOLD = 1000  # 通知の閾値を設定 (例: 1000ppm)

# --------------------------------------------------------------------------
# 4. 補助関数
# --------------------------------------------------------------------------
def get_latest_airoco_data() -> list[dict]:
    # !!! この部分は、将来的にAirocoからデータを取得する実際の処理に置き換えます !!!
    print("Airocoデータ取得中（現在はダミーデータを生成）...")
    timestamps = pd.to_datetime([pd.Timestamp.now(tz='Asia/Tokyo') - pd.Timedelta(minutes=x*5) for x in range(72)])
    timestamps = timestamps.sort_values()
    dummy_data = [
        {"timestamp": ts, "co2": 450 + (i*2) + np.random.randint(-5, 5), "temperature": 25.0 + np.random.rand(), "humidity": 60.0 + np.random.rand()}
        for i, ts in enumerate(timestamps)
    ]
    return dummy_data

def inverse_transform_co2(scaled_value, original_scaler, num_features):
    dummy_array = np.zeros((1, num_features))
    dummy_array[:, 0] = scaled_value
    return original_scaler.inverse_transform(dummy_array)[0, 0]

# --------------------------------------------------------------------------
# 5. GitHub Actionsから呼び出されるトリガーエンドポイント
# --------------------------------------------------------------------------
@app.post("/trigger-prediction")
async def trigger_prediction(x_trigger_secret: str = Header(None)):
    print("トリガーエンドポイントへのリクエストを受信しました。")

    # --- 認証キーのデバッグログ ---
    print("--- DEBUGGING START ---")
    expected_secret = TRIGGER_SECRET
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
    # --- デバッグここまで ---

    if not all([interpreter, scaler, TRIGGER_SECRET]):
        raise HTTPException(status_code=500, detail="サーバー設定が不完全です。")
    if received_secret != TRIGGER_SECRET:
        raise HTTPException(status_code=401, detail="認証に失敗しました。")

    try:
        # Step 1: データ取得と前処理
        latest_data = get_latest_airoco_data()
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
        
        # Step 2: TFLiteモデルの入力形式に整形
        reshaped_data = np.expand_dims(scaled_data, axis=0).astype(np.float32)

        # Step 3: TFLiteモデルで予測を実行
        interpreter.set_tensor(input_details[0]['index'], reshaped_data)
        interpreter.invoke()
        scaled_prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Step 4: 結果を元のスケールに戻す
        final_prediction = inverse_transform_co2(scaled_prediction, scaler, len(feature_order))
        print(f"✅ 予測成功。15分後のCO2濃度予測値: {final_prediction:.2f} ppm")

        # Step 5: 閾値と比較
        if final_prediction > CO2_THRESHOLD:
            print(f"⚠️ 警報: 予測値({final_prediction:.2f} ppm)が閾値({CO2_THRESHOLD} ppm)を超えました。")
            # ここで将来的に通知ロジックを呼び出す
        else:
            print(f"INFO: 予測値({final_prediction:.2f} ppm)は正常範囲内です。")

        return {"message": "予測処理が正常に完了しました。", "predicted_co2": round(float(final_prediction), 2)}

    except Exception as e:
        print(f"❌ 予測処理中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")