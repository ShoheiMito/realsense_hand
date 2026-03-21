# RealSense L515 3D Pose Estimation

Intel RealSense L515（LiDARデプスカメラ）を使用したリアルタイム3D全身ポーズ推定・手指トラッキング・表情認識システム。

研究・プロトタイプ用途。目標: 30fps以上のリアルタイム処理。

---

## 機能

| 機能 | 詳細 |
|------|------|
| **全身ポーズ推定** | MediaPipe PoseLandmarker（33点） |
| **手指トラッキング** | MediaPipe HandLandmarker（片手21点 × 両手） |
| **表情認識** | FaceLandmarker + ブレンドシェイプ → happy / surprise / angry / sad / neutral |
| **3D座標** | RealSense深度センサーによる各キーポイントのX/Y/Z座標（メートル） |
| **時間的スムージング** | One Euro Filterによるノイズ低減 |
| **機能トグル** | ポーズ・手・表情をCLI引数またはランタイムキーボードで独立ON/OFF |

---

## 必要環境

- Python 3.11+
- Intel RealSense L515（USB 3.x接続）
- Windows / Linux（macOSは未確認）

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
# 全身ポーズ推定モデル
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# 手指トラッキングモデル
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# 顔ランドマークモデル（表情認識用）
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

---

## 実行

### 基本

```bash
python -m src.main
```

### オプション

```bash
# 特定機能を無効化して起動
python -m src.main --no-hand          # 手検出なし
python -m src.main --no-expression    # 表情認識なし
python -m src.main --no-pose          # ポーズ推定なし（手+表情のみ）
python -m src.main --no-3d            # 3D座標HUDを非表示

# 解像度変更
python -m src.main --resolution 1280x720

# 録画
python -m src.main --record
```

### ランタイムキーボード操作

| キー | 操作 |
|------|------|
| `p` | ポーズ推定 ON/OFF |
| `h` | 手指トラッキング ON/OFF |
| `f` | 表情認識 ON/OFF |
| `q` | 終了 |

画面右下に現在のトグル状態 `P:ON  H:ON  F:OFF` が表示されます。

---

## アーキテクチャ

3スレッド構成のプロデューサー・コンシューマーパターン:

```
[Camera Thread]          [Processing Thread]         [Main Thread]
RealSense L515      →    PoseLandmarker          →   draw_skeleton()
  Align                  HandLandmarker              draw_hands()
  Depth Filter           3D Deprojection             draw_expression()
  FrameData              One Euro Smoother            draw_feature_status()
  → frame_queue          ExpressionRecognizer         cv2.imshow()
                         → result_queue
```

- `frame_queue` / `result_queue` は `maxsize=2` で古いフレームを破棄しレイテンシ蓄積を防止
- `FeatureFlags`（`threading.Event` × 3）でスレッドセーフな機能トグルを実現
- モデルは初回使用時に遅延初期化（起動時間・メモリ節約）

詳細は [docs/architecture.md](docs/architecture.md) を参照。

---

## パフォーマンス目安

| 構成 | 処理時間目安 | 30fps達成 |
|------|-------------|-----------|
| Pose + Face | ~18ms | ✅ |
| Pose + Hand（両手） | ~30–40ms | ⚠️ ギリギリ |
| Pose + Hand + Face（全機能） | ~35–45ms | ⚠️ `HAND_SKIP_FRAMES=2` 推奨 |
| Hand（両手）のみ | ~20ms | ✅ |

> `src/config.py` の `HAND_SKIP_FRAMES` を `2` にすると、手検出を1フレームおきに実行してCPU負荷を下げられます。

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

# フォーマット
ruff format src/
```

---

## RealSense L515 注意事項

- **USB 3.x接続必須**（USB 2.0では深度が 320×240 に制限）
- 赤外線ストリームはインデックス **0** のみ有効（1を指定すると RuntimeError）
- カラーと深度は同一解像度で設定（640×480 推奨）
- 深度フィルタチェーン: `spatial → temporal → hole_filling`（この順序で適用）
- ディスパリティ変換不要（LiDARのため）

---

## ライセンス

MIT License

---

## 依存ライブラリ

| ライブラリ | ライセンス | 用途 |
|-----------|-----------|------|
| [pyrealsense2](https://github.com/IntelRealSense/librealsense) | Apache 2.0 | RealSense SDK |
| [mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache 2.0 | ポーズ・手・顔推定 |
| [opencv-python](https://github.com/opencv/opencv) | Apache 2.0 | 描画・表示 |
| [numpy](https://numpy.org/) | BSD | 数値計算 |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | 将来の推論拡張用 |
