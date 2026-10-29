# 水面・地表面 手動トレース

時系列画像の水面と地表面を手動でトレースし、水深・地表面変化・フルード数・シールズ数などを分析するPythonアプリです。

## セットアップ・起動

Python 3.12以上を用意し、`python -m pip install -r requirements.txt` を実行してから、`python app.py` で起動します。Windowsでは既存Anacondaがある場合に `run_manual_trace.bat` も使用できます。

詳細は README_JA.md を参照してください。

学習モデルで境界を自動推定する別ソフト `ImuraHaruki/boundary-lab` とは異なります。動画・計測CSV・トレース記録・図の背景画像などの研究データは同梱していません。
