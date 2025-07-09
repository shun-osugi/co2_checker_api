import requests
import csv
from io import StringIO
from datetime import datetime, timedelta
import pytz

def get_latest_airoco_data() -> list[dict]:
    print("Airocoデータ取得中（API使用）...")
    try:
        # JSTの現在時刻 → UNIXタイムスタンプ（秒）
        jst = pytz.timezone('Asia/Tokyo')
        now = datetime.now(jst)
        start_time = int((now - timedelta(days=1)).timestamp())  # 24時間前（UNIX秒）

        # APIエンドポイント
        url = f"https://airoco.necolico.jp/data-api/day-csv?id=CgETViZ2&subscription-key=6b8aa7133ece423c836c38af01c59880&startDate={start_time}"

        response = requests.get(url)
        response.raise_for_status()

        # CSVパース
        csv_data = response.text
        reader = csv.reader(StringIO(csv_data))

        extracted = []
        for row in reader:
            if len(row) < 7:
                continue
            if row[1] == "Ｒ３ー４０１":
                try:
                    co2 = float(row[3])
                    temp = float(row[4])
                    humid = float(row[5])
                    timestamp = datetime.fromtimestamp(int(row[6]), tz=jst)
                    extracted.append({
                        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                        "co2": co2,
                        "temperature": temp,
                        "humidity": humid,
                    })
                except ValueError:
                    continue

        print(f"✅ Airocoデータ取得成功（{len(extracted)}件）")
        return extracted[-72:]

    except Exception as e:
        print(f"❌ Airocoデータ取得エラー: {e}")
        return []
