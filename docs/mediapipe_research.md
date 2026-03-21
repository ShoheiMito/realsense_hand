# MediaPipe API 比較調査レポート

## 概要

MediaPipeには2つの主要なAPIアプローチがあります：

- **Legacy API (Solutions API)**: `mp.solutions.pose` / `mp.solutions.face_mesh`
- **Tasks API**: `mediapipe.tasks.python.vision.PoseLandmarker` / `FaceLandmarker`

本ドキュメントでは、これらのAPIを5つの観点から比較します。

---

## 1. 出力データ構造（キーポイントタイプ、座標系）

### Legacy API

#### Pose (mp.solutions.pose)

```python
import mediapipe as mp

mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    results = pose.process(rgb_image)

    # 出力構造
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # 正規化座標 (0.0 ~ 1.0)
            x = landmark.x  # 画像幅で正規化
            y = landmark.y  # 画像高さで正規化
            z = landmark.z  # 深度（ヒップを基準とした相対値）
            visibility = landmark.visibility  # 可視性スコア (0.0 ~ 1.0)
```

**キーポイント数**: 33点（BlazePose）

**座標系**:
- `x`, `y`: 正規化座標（0.0〜1.0、画像の左上が原点）
- `z`: ヒップの中点を原点とした相対深度（スケールはxとほぼ同じ）

#### Face Mesh (mp.solutions.face_mesh)

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # 瞳のランドマークを追加
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = landmark.x  # 正規化座標
                y = landmark.y
                z = landmark.z  # 顔の中心からの相対深度
```

**キーポイント数**: 468点（refine_landmarks=Trueで478点、瞳10点追加）

### Tasks API

#### PoseLandmarker

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# オプション設定
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    # 出力構造
    for pose_landmarks in result.pose_landmarks:
        for landmark in pose_landmarks:
            # NormalizedLandmark
            x = landmark.x  # 正規化座標 (0.0 ~ 1.0)
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility
            presence = landmark.presence  # 存在確率（Tasks API固有）

    # ワールド座標（メートル単位）
    for pose_world_landmarks in result.pose_world_landmarks:
        for landmark in pose_world_landmarks:
            x = landmark.x  # メートル単位
            y = landmark.y
            z = landmark.z
```

**キーポイント数**: 33点

**座標系**:
- `pose_landmarks`: 正規化座標（Legacy APIと同じ）
- `pose_world_landmarks`: ワールド座標（メートル単位、ヒップ中心が原点）

#### FaceLandmarker

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,  # ブレンドシェイプ出力を有効化
    output_facial_transformation_matrixes=True  # 変換行列出力を有効化
)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    # ランドマーク
    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x = landmark.x
            y = landmark.y
            z = landmark.z

    # 変換行列 (4x4)
    for matrix in result.facial_transformation_matrixes:
        # numpy配列として取得可能
        transformation = matrix
```

**キーポイント数**: 478点

### 比較表

| 項目 | Legacy API | Tasks API |
|------|------------|-----------|
| Poseキーポイント数 | 33 | 33 |
| Faceキーポイント数 | 468/478 | 478 |
| 正規化座標 | あり | あり |
| ワールド座標 | なし（Poseのみz相対） | あり（Poseのみ、メートル単位） |
| presence属性 | なし | あり |
| 変換行列 | なし | あり（Face） |

---

## 2. ブレンドシェイプ出力の可用性と取得方法

### Legacy API (mp.solutions.face_mesh)

**ブレンドシェイプ出力: 非対応**

Legacy APIのFace Meshではブレンドシェイプ出力をサポートしていません。ランドマーク座標のみが取得可能です。

### Tasks API (FaceLandmarker)

**ブレンドシェイプ出力: 対応**

52種類のARKit互換ブレンドシェイプを出力できます。

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True  # 重要：これを有効化
)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    # ブレンドシェイプの取得
    for face_blendshapes in result.face_blendshapes:
        for blendshape in face_blendshapes:
            name = blendshape.category_name  # 例: "browDownLeft"
            score = blendshape.score  # 0.0 ~ 1.0
            index = blendshape.index

            print(f"{name}: {score:.3f}")
```

**利用可能なブレンドシェイプ（52種類、ARKit互換）**:

```python
# 主要なブレンドシェイプカテゴリ
BLENDSHAPE_NAMES = [
    # 眉
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",

    # 頬
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",

    # 目
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",

    # 顎
    "jawForward", "jawLeft", "jawOpen", "jawRight",

    # 口
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",

    # 鼻
    "noseSneerLeft", "noseSneerRight",

    # その他
    "_neutral"
]
```

### ブレンドシェイプをディクショナリとして取得するユーティリティ

```python
def get_blendshapes_dict(result):
    """FaceLandmarkerResultからブレンドシェイプを辞書形式で取得"""
    blendshapes_dict = {}

    if result.face_blendshapes:
        for blendshape in result.face_blendshapes[0]:  # 最初の顔
            blendshapes_dict[blendshape.category_name] = blendshape.score

    return blendshapes_dict

# 使用例
blendshapes = get_blendshapes_dict(result)
eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
mouth_open = blendshapes.get("jawOpen", 0.0)
```

---

## 3. LIVE_STREAMモードのコールバック指定

### Legacy API

Legacy APIには専用のLIVE_STREAMモードはありません。リアルタイム処理は通常のループで実装します。

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,  # ビデオモード
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)  # 同期処理

        # 結果の処理
        if results.pose_landmarks:
            # ランドマーク描画など
            pass

        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
```

### Tasks API (LIVE_STREAM モード)

Tasks APIでは3つの実行モードがあります：
- `IMAGE`: 単一画像処理（同期）
- `VIDEO`: ビデオファイル処理（同期）
- `LIVE_STREAM`: リアルタイムストリーム処理（非同期、コールバック必須）

#### PoseLandmarker LIVE_STREAM

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# グローバル変数で結果を保持
latest_result = None
latest_timestamp = 0

def pose_result_callback(
    result: vision.PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    """非同期コールバック関数"""
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms

    # コールバック内で結果を処理
    if result.pose_landmarks:
        print(f"Timestamp: {timestamp_ms}ms, Poses detected: {len(result.pose_landmarks)}")

# オプション設定（LIVE_STREAMモード）
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # LIVE_STREAMモード
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=pose_result_callback  # コールバック関数を指定
)

# ランドマーカーの作成と使用
cap = cv2.VideoCapture(0)
landmarker = vision.PoseLandmarker.create_from_options(options)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # タイムスタンプを取得（ミリ秒、単調増加が必須）
        timestamp_ms = int(time.time() * 1000)

        # MediaPipe Image形式に変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 非同期で検出（結果はコールバックで受け取る）
        landmarker.detect_async(mp_image, timestamp_ms)

        # latest_resultを使って描画など
        if latest_result and latest_result.pose_landmarks:
            # 描画処理
            pass

        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    landmarker.close()
    cap.release()
```

#### FaceLandmarker LIVE_STREAM

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class FaceData:
    """顔データを保持するクラス"""
    landmarks: List
    blendshapes: Dict[str, float]
    transformation_matrix: Optional[any]
    timestamp_ms: int

# 結果保持用
face_data: Optional[FaceData] = None

def face_result_callback(
    result: vision.FaceLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    """FaceLandmarker用コールバック"""
    global face_data

    if not result.face_landmarks:
        face_data = None
        return

    # ブレンドシェイプを辞書に変換
    blendshapes = {}
    if result.face_blendshapes:
        for bs in result.face_blendshapes[0]:
            blendshapes[bs.category_name] = bs.score

    # 変換行列
    matrix = None
    if result.facial_transformation_matrixes:
        matrix = result.facial_transformation_matrixes[0]

    face_data = FaceData(
        landmarks=result.face_landmarks[0],
        blendshapes=blendshapes,
        transformation_matrix=matrix,
        timestamp_ms=timestamp_ms
    )

# オプション設定
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    result_callback=face_result_callback
)

# 使用例
cap = cv2.VideoCapture(0)
landmarker = vision.FaceLandmarker.create_from_options(options)

try:
    frame_timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 単調増加するタイムスタンプ
        frame_timestamp += 33  # 約30fps想定

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarker.detect_async(mp_image, frame_timestamp)

        # 結果を使用
        if face_data:
            print(f"Eye blink L: {face_data.blendshapes.get('eyeBlinkLeft', 0):.2f}")

        cv2.imshow('Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    landmarker.close()
    cap.release()
```

### LIVE_STREAMモードの重要な注意点

```python
# 1. タイムスタンプは必ず単調増加させる
# 誤った例（エラーの原因）
timestamp_ms = int(time.time() * 1000)  # システム時刻は戻る可能性あり

# 正しい例
frame_count = 0
def get_monotonic_timestamp():
    global frame_count
    frame_count += 1
    return frame_count * 33  # 33ms間隔（約30fps）

# 2. コールバックは別スレッドで実行される可能性がある
# スレッドセーフな実装が必要
import threading

class ThreadSafeResult:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()

    def set(self, result):
        with self._lock:
            self._result = result

    def get(self):
        with self._lock:
            return self._result

# 3. detect_async()は即座に返る（非ブロッキング）
# 結果はコールバックで受け取る
```

### 比較表

| 項目 | Legacy API | Tasks API LIVE_STREAM |
|------|------------|----------------------|
| 処理方式 | 同期（ブロッキング） | 非同期（コールバック） |
| コールバック | 不要 | 必須 |
| タイムスタンプ管理 | 不要 | 必須（単調増加） |
| フレームドロップ | なし（待機） | あり（自動スキップ） |
| レイテンシ | 高め | 低め |

---

## 4. パフォーマンス（CPU/GPU）

### Legacy API

```python
import mediapipe as mp

# Pose - model_complexityでモデルサイズを選択
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,  # 0: Lite, 1: Full, 2: Heavy
    # GPUは自動的に使用される（利用可能な場合）
)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,  # 追加の計算が必要
)
```

**特徴**:
- GPUは内部で自動的に利用（設定不可）
- TensorFlow Liteベースの推論
- モデル切り替えは`model_complexity`で制御

### Tasks API

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CPU使用
base_options_cpu = python.BaseOptions(
    model_asset_path='pose_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU
)

# GPU使用（CUDA/OpenCL）
base_options_gpu = python.BaseOptions(
    model_asset_path='pose_landmarker.task',
    delegate=python.BaseOptions.Delegate.GPU
)

# オプション設定
options = vision.PoseLandmarkerOptions(
    base_options=base_options_gpu,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=callback_fn
)
```

### パフォーマンス比較（参考値）

| 項目 | Legacy API | Tasks API |
|------|------------|-----------|
| **Pose (CPU)** | | |
| - Lite | ~15ms | ~12ms |
| - Full | ~30ms | ~25ms |
| - Heavy | ~60ms | ~50ms |
| **Pose (GPU)** | | |
| - Full | ~10ms | ~8ms |
| **Face Mesh/Landmarker (CPU)** | ~15ms | ~12ms |
| **Face Mesh/Landmarker (GPU)** | ~8ms | ~6ms |

*注: 実際のパフォーマンスはハードウェア、画像サイズ、検出対象数により変動*

### GPU設定の詳細

```python
# Tasks APIでのGPU設定（詳細）
from mediapipe.tasks import python

# 基本的なGPU設定
base_options = python.BaseOptions(
    model_asset_path='model.task',
    delegate=python.BaseOptions.Delegate.GPU
)

# 注意: WindowsではGPUサポートが限定的
# Linux/macOSの方がGPUサポートが充実
```

### ベンチマーク用コード

```python
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def benchmark_legacy_pose(image, iterations=100):
    """Legacy API Poseのベンチマーク"""
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(model_complexity=1) as pose:
        # ウォームアップ
        for _ in range(10):
            pose.process(image)

        # 計測
        start = time.perf_counter()
        for _ in range(iterations):
            pose.process(image)
        elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms

def benchmark_tasks_pose(image, iterations=100):
    """Tasks API Poseのベンチマーク"""
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # ウォームアップ
        for _ in range(10):
            landmarker.detect(mp_image)

        # 計測
        start = time.perf_counter()
        for _ in range(iterations):
            landmarker.detect(mp_image)
        elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000  # ms

# 使用例
if __name__ == "__main__":
    # テスト画像の準備
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    legacy_time = benchmark_legacy_pose(test_image)
    tasks_time = benchmark_tasks_pose(test_image)

    print(f"Legacy API: {legacy_time:.2f}ms per frame")
    print(f"Tasks API:  {tasks_time:.2f}ms per frame")
```

---

## 5. 必要なモデルファイル（.task）のダウンロード方法

### Legacy API

Legacy APIではモデルファイルの手動ダウンロードは**不要**です。パッケージインストール時に自動的にダウンロードされます。

```bash
pip install mediapipe
```

モデルは以下の場所に自動配置されます：
- Windows: `%LOCALAPPDATA%\mediapipe\...`
- Linux/macOS: `~/.mediapipe/...`

### Tasks API

Tasks APIでは`.task`ファイルを明示的にダウンロードする必要があります。

#### ダウンロードURL一覧

**PoseLandmarker**:
```
# Lite (最軽量、低精度)
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

# Full (標準)
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# Heavy (高精度、重い)
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

**FaceLandmarker**:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

#### ダウンロード方法

**方法1: curlコマンド**

```bash
# PoseLandmarker Full
curl -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# FaceLandmarker
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**方法2: Pythonスクリプト**

```python
import urllib.request
import os

MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

def download_models(output_dir="models"):
    """モデルファイルをダウンロード"""
    os.makedirs(output_dir, exist_ok=True)

    for filename, url in MODELS.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"Already exists: {filepath}")
            continue

        print(f"Downloading: {filename}")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved: {filepath}")

if __name__ == "__main__":
    download_models()
```

**方法3: wgetコマンド（Linux/macOS）**

```bash
mkdir -p models
cd models

wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**方法4: PowerShell（Windows）**

```powershell
# ディレクトリ作成
New-Item -ItemType Directory -Force -Path models

# PoseLandmarker Full
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models\pose_landmarker_full.task"

# FaceLandmarker
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "models\face_landmarker.task"
```

### モデルファイルの読み込み

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# 相対パス
base_options = python.BaseOptions(
    model_asset_path='models/pose_landmarker_full.task'
)

# 絶対パス
model_path = os.path.abspath('models/pose_landmarker_full.task')
base_options = python.BaseOptions(
    model_asset_path=model_path
)

# バイトデータとして読み込み
with open('models/pose_landmarker_full.task', 'rb') as f:
    model_data = f.read()

base_options = python.BaseOptions(
    model_asset_buffer=model_data
)
```

---

## 総合比較表

| 比較項目 | Legacy API | Tasks API |
|----------|------------|-----------|
| **安定性** | 成熟、安定 | 新しいが活発に開発中 |
| **モデル管理** | 自動 | 手動ダウンロード必要 |
| **ブレンドシェイプ** | 非対応 | 対応（52種類） |
| **ワールド座標** | 限定的 | 対応（Pose） |
| **LIVE_STREAMモード** | なし（同期のみ） | あり（非同期コールバック） |
| **GPU制御** | 自動 | 明示的に指定可能 |
| **パフォーマンス** | 良好 | やや高速 |
| **将来性** | 非推奨に向かう可能性 | 推奨される新API |

---

## 推奨事項

### Legacy APIを選ぶべき場合
- 既存プロジェクトの保守
- シンプルな実装が必要な場合
- モデル管理の手間を省きたい場合
- ブレンドシェイプが不要な場合

### Tasks APIを選ぶべき場合
- 新規プロジェクト
- ブレンドシェイプが必要な場合（表情認識、アバター制御など）
- 非同期処理でレイテンシを最小化したい場合
- ワールド座標が必要な場合
- 明示的なGPU制御が必要な場合

---

## 参考リンク

- MediaPipe公式ドキュメント: https://developers.google.com/mediapipe
- MediaPipe GitHub: https://github.com/google/mediapipe
- PoseLandmarker: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- FaceLandmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

---

*最終更新: 2026年3月*
