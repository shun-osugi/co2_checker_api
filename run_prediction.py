import os
import joblib
import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
import pytz

# --------------------------------------------------------------------------
# 補助関数
# --------------------------------------------------------------------------
def get_latest_airoco_data() -> list[dict]:
    # !!! この部分は、将来的にAirocoからデータを取得する実際の処理に置き換えます !!!
    print("Airocoデータ取得中（現在はダミーデータを生成）...")
    jst = pytz.timezone('Asia/Tokyo')
    timestamps = pd.to_datetime([pd.Timestamp.now(jst) - pd.Timedelta(minutes=x*5) for x in range(72)])
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
# メインの処理
# --------------------------------------------------------------------------
def main():
    """
    15分ごとに実行されるメインの処理。
    モデルのロード、データ取得、予測、結果の表示を行う。
    """
    print("--- 予測ジョブを開始します ---")
    
    try:
        # Step 1: モデルとスケーラーをロード
        interpreter = Interpreter(model_path='co2_model.tflite')
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

        # Step 5: 閾値と比較 (将来の通知機能のため)
        CO2_THRESHOLD = 1000
        if final_prediction > CO2_THRESHOLD:
            print(f"⚠️ 警報: 予測値が閾値({CO2_THRESHOLD} ppm)を超えました。")
        else:
            print(f"INFO: 予測値は正常範囲内です。")

    except Exception as e:
        print(f"❌ 予測処理中にエラーが発生しました: {e}")
        # エラーが発生した場合、ワークフローを失敗させる
        exit(1)
        
    print("--- 予測ジョブが正常に完了しました ---")

# このスクリプトが直接実行された時だけ、main()関数を呼び出す
if __name__ == "__main__":
    main()