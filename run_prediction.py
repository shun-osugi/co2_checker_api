import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf  # <-- 修正点1: tflite_runtimeから変更
from datetime import datetime
import pytz
from get_airoco_data import get_latest_airoco_data
import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firestore():
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

def inverse_transform_co2(scaled_value, original_scaler, num_features):
    dummy_array = np.zeros((1, num_features))
    dummy_array[:, 0] = scaled_value
    return original_scaler.inverse_transform(dummy_array)[0, 0]

def main():
    print("--- 予測ジョブを開始します ---")
    try:
        db = initialize_firestore()

        # Step 1: モデルとスケーラーをロード
        interpreter = tf.lite.Interpreter(model_path='co2_model.tflite')  # <-- 修正点2: Interpreter -> tf.lite.Interpreter
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        scaler = joblib.load('co2_scaler.gz')
        print("✅ モデルとスケーラーの読み込み成功")

        # Step 2: データ取得と前処理
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
        reshaped_data = np.expand_dims(scaled_data, axis=0).astype(np.float32)

        # Step 3: TFLiteモデルで予測を実行
        interpreter.set_tensor(input_details[0]['index'], reshaped_data)
        interpreter.invoke()
        scaled_prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Step 4: 結果を元のスケールに戻す
        final_prediction = inverse_transform_co2(scaled_prediction, scaler, len(feature_order))
        print(f"✅ 予測成功！ 15分後のCO2濃度予測値: {final_prediction:.2f} ppm")

        # Step 5: Firebase Firestoreに保存
        doc_ref = db.collection("co2-prediction").document("TWNDdsILjaOACEoFemFx")
        doc_ref.set({
            "latest": float(final_prediction),
        }, merge=True)
        print("✅ Firestore に保存完了")

        # Step 6: 閾値と比較 (将来の通知機能のため)
        CO2_THRESHOLD = 1000
        if final_prediction > CO2_THRESHOLD:
            print(f"⚠️ 警報: 予測値が閾値({CO2_THRESHOLD} ppm)を超えました。")
        else:
            print(f"INFO: 予測値は正常範囲内です。")

    except Exception as e:
        print(f"❌ 予測処理中にエラーが発生しました: {e}")
        exit(1)
        
    print("--- 予測ジョブが正常に完了しました ---")

if __name__ == "__main__":
    main()