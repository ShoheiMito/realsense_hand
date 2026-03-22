# RealSense L515 Hand Gesture Control

Intel RealSense L515（LiDARデプスカメラ）の手指トラッキングを使用したPC画面操作システム。

手指ジェスチャーでマウスカーソル移動・クリック・ドラッグ・スクロールをリアルタイムで実行します。

---

## ジェスチャー操作

| 操作 | ジェスチャー | 説明 |
|------|------------|------|
| **カーソル移動** | 人差し指を立てる | カメラ中央70%領域がスクリーン全体にマッピング |
| **クリック** | ピンチして素早く離す | 親指と人差し指をつまむ→離す（0.3秒以内） |
| **ドラッグ** | ピンチしたまま移動 | つまんだまま手を動かす |
| **スクロール** | 2本指を立てて上下 | 人差し指+中指を伸ばして上下移動 |

---

## 必要環境

- Python 3.10+
- Intel RealSense L515（USB 3.x接続）
- Windows（pynputによるマウス制御）

> **注意:** L515 は生産終了品であり、pyrealsense2 v2.55 以降では認識されません。
> `requirements.txt` で `pyrealsense2==2.54.1.5216` に固定しています。

---

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. MediaPipe モデルのダウンロード

```bash
# 手指トラッキングモデル
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

---

## 実行

### 基本（手指コントロール有効）

```bash
python -m src.main
```

### オプション

```bash
# マウス制御なし（手指トラッキング表示のみ）
python -m src.main --no-control

# 解像度変更
python -m src.main --resolution 1280x720

# 録画
python -m src.main --record
```

### ランタイムキーボード操作

| キー | 操作 |
|------|------|
| `c` | マウス制御 ON/OFF トグル |
| `q` | 終了 |

---

## アーキテクチャ

3スレッド構成のプロデューサー・コンシューマーパターン:

```
[Camera Thread]          [Processing Thread]         [Main Thread]
RealSense L515      →    HandLandmarker          →   draw_hands()
  Align                   3D Deprojection             HandController.update()
  Depth Filter            One Euro Smoother            → mouse move/click/scroll
  FrameData               HandResult                  draw_control_overlay()
  → frame_queue           → result_queue              cv2.imshow()
```

- `frame_queue` / `result_queue` は `maxsize=2` で古いフレームを破棄しレイテンシ蓄積を防止
- 手指検出のみで ~20ms/frame（50fps相当、カメラ30fps上限）

---

## 設定調整

`src/config.py` の `CONTROL_*` 定数でジェスチャー感度を調整可能:

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `CONTROL_ACTIVE_REGION` | 0.7 | カメラ画面の操作領域（中央何%） |
| `CONTROL_SMOOTHING_ALPHA` | 0.4 | カーソル平滑化（0=スムーズ, 1=即応） |
| `CONTROL_PINCH_THRESHOLD_PX` | 30 | ピンチ検出距離（px） |
| `CONTROL_SCROLL_SENSITIVITY` | 3.0 | スクロール感度 |
| `CONTROL_MIRROR_X` | True | X軸ミラーリング |

---

## テスト・品質チェック

```bash
# テスト実行
pytest tests/ -v

# カバレッジ付き
pytest --cov=src tests/

# 型チェック
mypy src/ --ignore-missing-imports

# リント
ruff check src/
ruff format src/
```

---

## RealSense L515 注意事項

- **USB 3.x接続必須**（USB 2.0では深度が 320×240 に制限）
- 赤外線ストリームはインデックス **0** のみ有効
- カラーと深度は同一解像度で設定（640×480 推奨）
- 深度フィルタチェーン: `spatial → temporal → hole_filling`

---

## ライセンス

MIT License

---

## 依存ライブラリ

| ライブラリ | ライセンス | 用途 |
|-----------|-----------|------|
| [pyrealsense2](https://github.com/IntelRealSense/librealsense) | Apache 2.0 | RealSense SDK |
| [mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache 2.0 | 手指ランドマーク検出 |
| [pynput](https://github.com/moses-palmer/pynput) | LGPL 3.0 | マウス制御 |
| [opencv-python](https://github.com/opencv/opencv) | Apache 2.0 | 描画・表示 |
| [numpy](https://numpy.org/) | BSD | 数値計算 |
