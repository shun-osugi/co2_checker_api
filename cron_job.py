# cron_job.py

from datetime import datetime
import pytz  # タイムゾーンを扱うためのライブラリ

def run_test_job():
    """
    Cron Jobがスケジュール通りに実行されているかを確認するための
    シンプルなテスト用関数。
    """
    # タイムゾーンを日本時間に設定
    jst = pytz.timezone('Asia/Tokyo')
    current_time_jst = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S %Z')

    print("=====================================")
    print(f"✅ Cron Job executed at: {current_time_jst}")
    print("This script is a test to confirm the 15-minute schedule.")
    print("This will eventually be replaced with the prediction logic.")
    print("=====================================")

if __name__ == "__main__":
    run_test_job()