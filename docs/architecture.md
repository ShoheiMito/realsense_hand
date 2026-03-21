# アーキテクチャ詳細設計

## 概要

RealSense L515 を使用した3Dポーズ推定＋表情認識のリアルタイムシステム。
3スレッド Producer-Consumer パターンにより、30fps 以上のリアルタイム処理を実現する。

---

## データフロー図

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Thread 1: Camera                            │
│                                                                     │
│  RealSense L515                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐              │
│  │ Color    │    │ Depth    │    │ rs.align(color)  │              │
│  │ 640x480  │───▶│ 640x480  │───▶│                  │              │
│  │ BGR8@30  │    │ Z16@30   │    └────────┬─────────┘              │
│  └──────────┘    └──────────┘             │                        │
│                                           ▼                        │
│                              ┌──────────────────────┐              │
│                              │ Depth Filter Chain   │              │
│                              │ spatial → temporal   │              │
│                              │ → hole_filling       │              │
│                              └──────────┬───────────┘              │
│                                         │                          │
│                                         ▼                          │
│                              ┌──────────────────────┐              │
│                              │ FrameData            │              │
│                              │ (color, depth,       │              │
│                              │  intrinsics)         │              │
│                              └──────────┬───────────┘              │
└─────────────────────────────────────────┼───────────────────────────┘
                                          │
                                  ┌───────▼───────┐
                                  │  frame_queue  │
                                  │  maxsize=2    │
                                  │  (古いフレーム │
                                  │   を破棄)     │
                                  └───────┬───────┘
                                          │
┌─────────────────────────────────────────┼───────────────────────────┐
│                        Thread 2: Processor                         │
│                                         │                          │
│                                         ▼                          │
│                              ┌──────────────────────┐              │
│                              │ MediaPipe Pose       │              │
│                              │ (Tasks API)          │              │
│                              │ 33 landmarks         │              │
│                              └──────────┬───────────┘              │
│                                         │                          │
│                              ┌──────────▼───────────┐              │
│                              │ 3D Deprojection      │              │
│                              │ pixel→world coords   │              │
│                              │ + median fallback    │              │
│                              └──────────┬───────────┘              │
│                                         │                          │
│                              ┌──────────▼───────────┐              │
│                              │ One Euro Filter      │              │
│                              │ (temporal smoothing)  │              │
│                              └──────────┬───────────┘              │
│                                         │                          │
│                     ┌───────────────────┼───────────┐              │
│                     │ 毎フレーム        │ N フレーム毎              │
│                     ▼                   ▼           │              │
│          ┌──────────────┐    ┌──────────────────┐   │              │
│          │ PoseResult   │    │ FaceLandmarker   │   │              │
│          │ (3D coords)  │    │ (blendshapes)    │   │              │
│          └──────┬───────┘    └────────┬─────────┘   │              │
│                 │                     │             │              │
│                 └─────────┬───────────┘             │              │
│                           ▼                          │              │
│                ┌─────────────────────┐              │              │
│                │ ProcessingResult    │              │              │
│                │ (pose_3d, emotion,  │              │              │
│                │  fps_info)          │              │              │
│                └─────────┬───────────┘              │              │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                   ┌───────▼───────┐
                   │ result_queue  │
                   │  maxsize=2    │
                   └───────┬───────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────┐
│                 Thread 3: Main (Visualizer)                        │
│                          │                                          │
│                          ▼                                          │
│               ┌─────────────────────┐                              │
│               │ Skeleton Drawing    │                              │
│               │ (OpenPose-style     │                              │
│               │  colorful overlay)  │                              │
│               └─────────┬───────────┘                              │
│                         │                                          │
│               ┌─────────▼───────────┐                              │
│               │ Emotion Label       │                              │
│               │ + FPS display       │                              │
│               └─────────┬───────────┘                              │
│                         │                                          │
│               ┌─────────▼───────────┐                              │
│               │ cv2.imshow()        │                              │
│               │ + キー入力処理      │                              │
│               └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## モジュール設計

### 1. config.py — 設定・定数

**責務:** システム全体の設定値を一元管理する。

```python
# --- 既存（変更なし） ---
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 30

FRAME_QUEUE_SIZE: int = 2
RESULT_QUEUE_SIZE: int = 2

SPATIAL_FILTER_MAGNITUDE: int = 2
SPATIAL_FILTER_ALPHA: float = 0.5
SPATIAL_FILTER_DELTA: int = 20
TEMPORAL_FILTER_ALPHA: float = 0.4
TEMPORAL_FILTER_DELTA: int = 20
HOLE_FILLING_MODE: int = 1

DEPTH_FALLBACK_KERNEL_SIZE: int = 5
EXPRESSION_SKIP_FRAMES: int = 3

@dataclass
class OneEuroFilterParams:
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

# --- 追加 ---

# MediaPipe Pose 設定
POSE_MODEL_PATH: str = "models/pose_landmarker_full.task"
POSE_MIN_DETECTION_CONFIDENCE: float = 0.5
POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
POSE_NUM_POSES: int = 1

# MediaPipe FaceLandmarker 設定
FACE_MODEL_PATH: str = "models/face_landmarker.task"
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

# 深度有効範囲 (メートル)
DEPTH_MIN_DISTANCE: float = 0.25
DEPTH_MAX_DISTANCE: float = 4.0

# 可視化設定
SKELETON_LINE_THICKNESS: int = 3
SKELETON_CIRCLE_RADIUS: int = 5
WINDOW_NAME: str = "RealSense 3D Pose"
```

**依存関係:** なし（他のすべてのモジュールから参照される）

---

### 2. camera.py — RealSense カメラ管理（Thread 1）

**責務:** RealSense L515 の初期化、フレーム取得、深度フィルタリング、フレームキューへのプッシュ。

```python
@dataclass
class FrameData:
    """カメラスレッドから処理スレッドへ渡すデータ。"""
    color_image: np.ndarray          # BGR (H, W, 3) uint8
    depth_image: np.ndarray          # (H, W) uint16 (生の深度値)
    depth_frame: rs.depth_frame      # get_distance() 用
    intrinsics: rs.intrinsics        # デプロジェクション用
    timestamp: float                 # time.monotonic()


class RealsenseCamera:
    """RealSense L515 カメラの管理とフレーム取得。"""

    def __init__(self) -> None:
        """パイプライン、フィルタ、アライメントを初期化。"""
        ...

    def start(self) -> None:
        """ストリーミングを開始し、最初の数フレームを破棄。"""
        ...

    def get_frame(self) -> FrameData | None:
        """アライン + フィルタ適用済みフレームを1つ取得。

        Returns:
            FrameData: 成功時
            None: フレーム取得失敗時
        """
        ...

    def stop(self) -> None:
        """パイプラインを停止。"""
        ...


def camera_thread(
    frame_queue: queue.Queue[FrameData],
    stop_event: threading.Event,
) -> None:
    """カメラスレッドのエントリーポイント。

    Args:
        frame_queue: FrameData を格納するキュー (maxsize=2)
        stop_event: 停止シグナル
    """
    ...
```

**内部実装のポイント:**
- フィルタオブジェクト（`spatial_filter`, `temporal_filter`, `hole_filling_filter`）は `__init__` で1回だけ生成し再利用
- `rs.align(rs.stream.color)` でカラー座標系にアライン
- `intrinsics` はアライン後の深度ストリームプロファイルから取得
- キューがフルの場合は `get_nowait()` で古いフレームを破棄してから `put_nowait()`
- L515 固有: 赤外線ストリームはインデックス 0 のみ、ディスパリティ変換は不要

**依存関係:** `config.py`

---

### 3. depth_utils.py — 深度処理ユーティリティ

**責務:** 2D→3D デプロジェクション、深度欠損のフォールバック処理、One Euro フィルタ。

```python
def deproject_pixel_to_point(
    intrinsics: rs.intrinsics,
    pixel: tuple[int, int],
    depth_frame: rs.depth_frame,
    depth_image: np.ndarray,
    fallback_kernel: int = 5,
) -> tuple[float, float, float] | None:
    """ピクセル座標を3Dワールド座標に変換。

    深度値が0（欠損）の場合、fallback_kernel x fallback_kernel 近傍の
    メディアン値でフォールバックする。

    Args:
        intrinsics: カメラ内部パラメータ
        pixel: (x, y) ピクセル座標
        depth_frame: RealSense 深度フレーム
        depth_image: 深度画像の NumPy 配列
        fallback_kernel: フォールバック近傍サイズ

    Returns:
        (x, y, z) メートル単位の3D座標。取得不能な場合は None。
    """
    ...


def deproject_landmarks(
    intrinsics: rs.intrinsics,
    landmarks: list[tuple[float, float]],
    depth_frame: rs.depth_frame,
    depth_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float, float] | None]:
    """複数のランドマークを一括で3D座標に変換。

    Args:
        intrinsics: カメラ内部パラメータ
        landmarks: 正規化座標 [(nx, ny), ...] のリスト
        depth_frame: RealSense 深度フレーム
        depth_image: 深度画像の NumPy 配列
        image_width: 画像幅
        image_height: 画像高さ

    Returns:
        3D座標のリスト。各要素は (x, y, z) または None。
    """
    ...


def get_median_depth(
    depth_image: np.ndarray,
    x: int,
    y: int,
    kernel_size: int = 5,
) -> float:
    """指定ピクセル周辺のメディアン深度値を取得。

    Args:
        depth_image: 深度画像 (H, W) uint16
        x, y: 中心ピクセル座標
        kernel_size: 近傍サイズ（奇数）

    Returns:
        メディアン深度値。有効な値がない場合は 0.0。
    """
    ...


class OneEuroFilter:
    """1D One Euro Filter for temporal smoothing.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, params: OneEuroFilterParams) -> None: ...

    def __call__(self, x: float, timestamp: float) -> float:
        """フィルタを適用して平滑化された値を返す。"""
        ...


class KeypointSmoother:
    """33 個の 3D キーポイントに対する One Euro Filter の管理。"""

    def __init__(self, num_keypoints: int = 33) -> None: ...

    def smooth(
        self,
        keypoints_3d: list[tuple[float, float, float] | None],
        timestamp: float,
    ) -> list[tuple[float, float, float] | None]:
        """全キーポイントにスムージングを適用。

        Args:
            keypoints_3d: 生の3D座標リスト
            timestamp: 現在のタイムスタンプ（秒）

        Returns:
            スムージング済みの3D座標リスト
        """
        ...
```

**依存関係:** `config.py`

---

### 4. expression.py — 表情認識モジュール

**責務:** MediaPipe FaceLandmarker からブレンドシェイプを取得し、感情ラベルにマッピングする。

```python
# ブレンドシェイプ → 感情のマッピングルール
EMOTION_RULES: dict[str, list[tuple[str, float]]] = {
    "happy":    [("mouthSmileLeft", 0.4), ("mouthSmileRight", 0.4)],
    "surprise": [("eyeWideLeft", 0.5), ("eyeWideRight", 0.5), ("jawOpen", 0.3)],
    "angry":    [("browDownLeft", 0.5), ("browDownRight", 0.5)],
    "sad":      [("mouthFrownLeft", 0.4), ("mouthFrownRight", 0.4)],
    "neutral":  [],
}


@dataclass
class ExpressionResult:
    """表情認識の結果。"""
    emotion: str                           # "happy", "surprise", "angry", "sad", "neutral"
    confidence: float                      # 0.0 ~ 1.0
    blendshapes: dict[str, float]          # 全ブレンドシェイプ値


class ExpressionRecognizer:
    """MediaPipe FaceLandmarker を使用した表情認識。"""

    def __init__(self) -> None:
        """FaceLandmarker を IMAGE モードで初期化。"""
        ...

    def recognize(self, rgb_image: np.ndarray) -> ExpressionResult | None:
        """RGB 画像から表情を認識。

        Args:
            rgb_image: RGB 画像 (H, W, 3) uint8

        Returns:
            ExpressionResult: 顔が検出された場合
            None: 顔が検出されなかった場合
        """
        ...

    def close(self) -> None:
        """リソースを解放。"""
        ...
```

**設計判断:**
- FaceLandmarker は IMAGE モードを使用（LIVE_STREAM のコールバックはスレッド間で複雑になるため）
- N フレームに1回（`EXPRESSION_SKIP_FRAMES`）のみ実行し、CPU 負荷を軽減
- ブレンドシェイプ閾値ベースの感情分類（軽量、ルール調整が容易）

**依存関係:** `config.py`

---

### 5. processor.py — ポーズ推定＋統合処理（Thread 2）

**責務:** カメラフレームを受け取り、ポーズ推定・3Dデプロジェクション・表情認識を実行し、結果をキューに格納。

```python
@dataclass
class PoseKeypoint3D:
    """1つの3Dキーポイント。"""
    x: float          # メートル
    y: float          # メートル
    z: float          # メートル
    visibility: float # 0.0 ~ 1.0
    name: str         # ランドマーク名


@dataclass
class ProcessingResult:
    """処理スレッドから可視化スレッドへ渡すデータ。"""
    color_image: np.ndarray                        # BGR 描画用
    landmarks_2d: list[tuple[int, int]] | None     # ピクセル座標 (描画用)
    keypoints_3d: list[PoseKeypoint3D] | None      # 3D座標
    expression: ExpressionResult | None            # 表情認識結果
    processing_fps: float                          # 処理スレッドの FPS
    timestamp: float


class PoseProcessor:
    """MediaPipe Pose + 深度統合 + 表情認識の統合処理。"""

    def __init__(self) -> None:
        """PoseLandmarker, ExpressionRecognizer, KeypointSmoother を初期化。"""
        ...

    def process_frame(self, frame_data: FrameData) -> ProcessingResult:
        """1フレームを処理。

        Args:
            frame_data: カメラスレッドからのフレームデータ

        Returns:
            ProcessingResult: 処理結果
        """
        ...

    def close(self) -> None:
        """全リソースを解放。"""
        ...


def processing_thread(
    frame_queue: queue.Queue[FrameData],
    result_queue: queue.Queue[ProcessingResult],
    stop_event: threading.Event,
) -> None:
    """処理スレッドのエントリーポイント。

    Args:
        frame_queue: 入力フレームキュー
        result_queue: 出力結果キュー
        stop_event: 停止シグナル
    """
    ...
```

**内部実装のポイント:**
- `image.flags.writeable = False` を MediaPipe 推論前に設定（メモリ最適化）
- PoseLandmarker は IMAGE モード（同期）を使用
- 表情認識は `frame_count % EXPRESSION_SKIP_FRAMES == 0` の時のみ実行
- 前回の `ExpressionResult` を保持し、スキップフレームでは前回結果を再利用
- キューへの put/get は camera_thread と同じ古フレーム破棄パターン

**依存関係:** `config.py`, `camera.py`(FrameData), `depth_utils.py`, `expression.py`

---

### 6. visualizer.py — 骨格描画・表示（Thread 3 / Main Thread）

**責務:** 処理結果を受け取り、カラフルな骨格を描画し、`cv2.imshow` で表示する。

```python
# OpenPose スタイルの骨格接続定義と色
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # 頭部
    (0, 4), (4, 5), (5, 6), (6, 8),       # 頭部
    (9, 10),                                # 口
    (11, 12),                               # 肩
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),  # 左腕
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),  # 右腕
    (11, 23), (12, 24), (23, 24),           # 胴体
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),  # 左脚
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),  # 右脚
]

# 部位ごとの色 (BGR)
LIMB_COLORS: dict[str, tuple[int, int, int]] = {
    "head":      (0, 255, 255),    # 黄色
    "left_arm":  (0, 128, 255),    # オレンジ
    "right_arm": (0, 255, 0),      # 緑
    "torso":     (255, 255, 0),    # シアン
    "left_leg":  (255, 0, 128),    # ピンク
    "right_leg": (255, 0, 0),      # 青
}


class Visualizer:
    """骨格描画と画面表示。"""

    def __init__(self) -> None: ...

    def draw(self, result: ProcessingResult) -> np.ndarray:
        """処理結果をカラー画像に描画。

        Args:
            result: ProcessingResult

        Returns:
            描画済みの画像 (BGR)
        """
        ...

    def _draw_skeleton(
        self,
        image: np.ndarray,
        landmarks_2d: list[tuple[int, int]],
        keypoints_3d: list[PoseKeypoint3D] | None,
    ) -> None:
        """OpenPose スタイルのカラフルな骨格を描画。"""
        ...

    def _draw_emotion_label(
        self,
        image: np.ndarray,
        expression: ExpressionResult | None,
    ) -> None:
        """感情ラベルを画面上部に表示。"""
        ...

    def _draw_fps(self, image: np.ndarray, fps: float) -> None:
        """FPS を画面左上に表示。"""
        ...


def run_visualizer(
    result_queue: queue.Queue[ProcessingResult],
    stop_event: threading.Event,
) -> None:
    """メインスレッドで実行する可視化ループ。

    cv2.imshow / cv2.waitKey はメインスレッドで呼ぶ必要があるため、
    この関数はメインスレッドから直接呼び出す。

    Args:
        result_queue: 処理結果キュー
        stop_event: 停止シグナル（'q' キーで set）
    """
    ...
```

**依存関係:** `config.py`, `processor.py`(ProcessingResult, PoseKeypoint3D)

---

### 7. main.py — エントリーポイント

**責務:** スレッドの起動・停止と全体のライフサイクル管理。

```python
def main() -> None:
    """アプリケーションのエントリーポイント。

    1. frame_queue, result_queue を作成
    2. stop_event を作成
    3. camera_thread, processing_thread を daemon スレッドとして起動
    4. run_visualizer をメインスレッドで実行
    5. 終了時に stop_event.set() → スレッド join → リソース解放
    """
    ...


if __name__ == "__main__":
    main()
```

**依存関係:** `config.py`, `camera.py`, `processor.py`, `visualizer.py`

---

## モジュール依存関係図

```
          config.py
         ╱    │    ╲
        ╱     │     ╲
       ▼      ▼      ▼
  camera.py  depth_utils.py  expression.py
       │      ╱     ╲          │
       │     ╱       ╲         │
       ▼    ▼         ▼       ▼
      processor.py ◀──────────┘
           │
           ▼
      visualizer.py
           │
           ▼
        main.py
```

---

## エラーハンドリング戦略

### 1. カメラ接続エラー

```
camera.py: start() で pipeline.start() 失敗
  → RuntimeError をキャッチ
  → ログ出力 + stop_event.set() でシステム全体を停止
  → USB 2.0 接続の場合は警告メッセージを表示
```

### 2. フレーム取得タイムアウト

```
camera.py: wait_for_frames(timeout_ms=5000) で RuntimeError
  → 3回リトライ
  → 3回失敗 → stop_event.set()
```

### 3. 深度欠損

```
depth_utils.py: deproject_pixel_to_point() で depth == 0
  → 5x5 近傍のメディアンでフォールバック
  → メディアンも 0 → None を返す
  → 呼び出し側で None チェック → 前フレームの値を維持
```

### 4. MediaPipe 推論失敗

```
processor.py: results.pose_landmarks が空
  → landmarks_2d=None, keypoints_3d=None を返す
  → visualizer は骨格描画をスキップ（カラー画像のみ表示）
```

### 5. キューフル

```
camera_thread / processing_thread: queue.Full
  → get_nowait() で古い要素を1つ破棄
  → put_nowait() で新しい要素を格納
  → ログ出力は行わない（高頻度で発生しうるため）
```

### 6. グレースフルシャットダウン

```
main.py:
  1. 'q' キー押下 or KeyboardInterrupt
  2. stop_event.set()
  3. camera_thread.join(timeout=2.0)
  4. processing_thread.join(timeout=2.0)
  5. RealsenseCamera.stop()
  6. PoseProcessor.close()
  7. cv2.destroyAllWindows()
```

---

## パフォーマンス最適化

### 1. MediaPipe 推論の最適化
- `image.flags.writeable = False` で不要なメモリコピーを回避
- PoseLandmarker の `model_complexity` は Full (1) をデフォルトに。30fps を下回る場合は Lite (0) にフォールバック

### 2. 深度フィルタの最適化
- フィルタオブジェクトは `__init__` で1回だけ生成（毎フレーム再生成しない）
- L515 は LiDAR のためディスパリティ変換は不要（ステレオカメラと異なる）

### 3. 表情認識の間引き
- `EXPRESSION_SKIP_FRAMES = 3`（3フレームに1回実行）
- 人間の知覚的には差がなく、CPU 負荷を約 33% 軽減

### 4. キューによるレイテンシ制御
- `maxsize=2` で古いフレームを破棄
- 処理スレッドが遅延してもキューに古いフレームが蓄積しない

### 5. NumPy ベクトル化
- `depth_image` は NumPy 配列として扱い、ピクセル値の境界チェックとメディアン計算をベクトル化

### 6. One Euro フィルタ
- 3Dキーポイントのジッター除去
- 動きが速い時は追従性を優先（beta パラメータで制御）
- 33 keypoints × 3 axes = 99 個のフィルタインスタンス

### 7. 解像度の統一
- カラー・深度ともに 640x480 に統一
- `rs.align` のコスト最小化（同一解像度間のアライメント: 約 1-2ms）

---

## データ型まとめ

| 型名 | 定義場所 | 用途 |
|---|---|---|
| `FrameData` | camera.py | カメラ→処理スレッド間のフレームデータ |
| `PoseKeypoint3D` | processor.py | 1つの3Dキーポイント |
| `ProcessingResult` | processor.py | 処理→可視化スレッド間の結果データ |
| `ExpressionResult` | expression.py | 表情認識の結果 |
| `OneEuroFilterParams` | config.py | One Euro フィルタのパラメータ |
