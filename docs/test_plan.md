# テスト計画

docs/architecture.md の設計に基づくテスト戦略。

---

## テスト実行コマンド

```bash
# 全テスト実行
pytest tests/ -v

# モジュール別
pytest tests/test_depth_utils.py -v
pytest tests/test_expression.py -v
pytest tests/test_config.py -v
pytest tests/test_camera.py -v
pytest tests/test_processor.py -v

# カバレッジ
pytest --cov=src tests/
```

---

## 1. 単体テスト（RealSense なしで実行可能）

### 1.1 test_config.py

| テストケース | 内容 | 期待結果 |
|---|---|---|
| `test_camera_resolution_valid` | `CAMERA_WIDTH`, `CAMERA_HEIGHT` が正の整数 | アサーション通過 |
| `test_camera_fps_valid` | `CAMERA_FPS` が 15, 30, 60 のいずれか | アサーション通過 |
| `test_queue_sizes_positive` | `FRAME_QUEUE_SIZE`, `RESULT_QUEUE_SIZE` が 1 以上 | アサーション通過 |
| `test_depth_filter_params_range` | Spatial/Temporal フィルタパラメータが有効範囲内 | アサーション通過 |
| `test_depth_fallback_kernel_odd` | `DEPTH_FALLBACK_KERNEL_SIZE` が奇数 | アサーション通過 |
| `test_expression_skip_frames_positive` | `EXPRESSION_SKIP_FRAMES` が 1 以上 | アサーション通過 |
| `test_one_euro_params_defaults` | `OneEuroFilterParams()` のデフォルト値が妥当 | `min_cutoff=1.0, beta=0.007, d_cutoff=1.0` |
| `test_depth_distance_range` | `DEPTH_MIN_DISTANCE < DEPTH_MAX_DISTANCE` | アサーション通過 |
| `test_confidence_thresholds_range` | 信頼度閾値が 0.0〜1.0 の範囲 | アサーション通過 |

---

### 1.2 test_depth_utils.py

#### get_median_depth

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_median_depth_center_pixel` | 均一な深度画像 (全ピクセル=1000)、中心座標 | `1000.0` |
| `test_median_depth_with_zeros` | 中心が0、周囲が1000の深度画像 | 0以外の中央値（1000に近い値） |
| `test_median_depth_all_zeros` | 全ピクセル0の深度画像 | `0.0` |
| `test_median_depth_edge_pixel` | 画像端の座標 (0, 0) | クリッピングされた領域のメディアン |
| `test_median_depth_corner` | 右下角の座標 (W-1, H-1) | 境界外アクセスなし |
| `test_median_depth_kernel_3` | kernel_size=3 | 3x3 近傍のメディアン |
| `test_median_depth_kernel_5` | kernel_size=5 | 5x5 近傍のメディアン |

#### deproject_pixel_to_point

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_deproject_valid_depth` | 有効な深度値 (1.0m) のピクセル | `(x, y, z)` タプル、z ≈ 1.0 |
| `test_deproject_zero_depth_fallback` | 深度=0、周囲に有効値あり | メディアンフォールバックで非None |
| `test_deproject_zero_depth_no_fallback` | 深度=0、周囲も全て0 | `None` |
| `test_deproject_out_of_range` | `DEPTH_MAX_DISTANCE` を超える深度 | `None` |
| `test_deproject_negative_coords` | 負のピクセル座標 | `None`（IndexError を発生させない） |
| `test_deproject_coords_beyond_image` | 画像サイズを超える座標 | `None` |

**テスト用モック:** `rs.intrinsics` のモックオブジェクト（`ppx`, `ppy`, `fx`, `fy`, `width`, `height` を設定）

```python
@pytest.fixture
def mock_intrinsics():
    """テスト用のカメラ内部パラメータ。"""
    intrinsics = MagicMock()
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = 320.0
    intrinsics.ppy = 240.0
    intrinsics.fx = 600.0
    intrinsics.fy = 600.0
    intrinsics.model = MagicMock()
    intrinsics.coeffs = [0.0] * 5
    return intrinsics
```

#### deproject_landmarks

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_deproject_landmarks_basic` | 正規化座標 [(0.5, 0.5)] | 1要素のリスト |
| `test_deproject_landmarks_multiple` | 33個の正規化座標 | 33要素のリスト |
| `test_deproject_landmarks_empty` | 空リスト | 空リスト |
| `test_deproject_landmarks_mixed_validity` | 有効/無効な深度が混在 | 対応する要素が (x,y,z) または None |

#### OneEuroFilter

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_one_euro_first_value` | 最初の値 | フィルタなしでそのまま返す |
| `test_one_euro_constant_input` | 同じ値を連続入力 | 入力値と同じ値を返す |
| `test_one_euro_smoothing` | ノイズ付きの値列 | 入力より分散が小さい出力 |
| `test_one_euro_step_response` | 急激な値変化 | 徐々に追従（オーバーシュートなし） |
| `test_one_euro_high_beta_fast_tracking` | beta=1.0, 急激な変化 | 素早く追従 |

#### KeypointSmoother

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_smoother_initialization` | `num_keypoints=33` | 99個のフィルタ（33×3軸） |
| `test_smoother_none_keypoints` | 一部が None のリスト | None はそのまま None |
| `test_smoother_consistent_output` | 同じ座標を連続入力 | 安定した出力 |

---

### 1.3 test_expression.py

#### 感情マッピングロジック

| テストケース | 入力ブレンドシェイプ | 期待結果 |
|---|---|---|
| `test_happy_detection` | `mouthSmileLeft=0.7, mouthSmileRight=0.7` | `emotion="happy"` |
| `test_surprise_detection` | `eyeWideLeft=0.8, eyeWideRight=0.8, jawOpen=0.6` | `emotion="surprise"` |
| `test_angry_detection` | `browDownLeft=0.7, browDownRight=0.7` | `emotion="angry"` |
| `test_sad_detection` | `mouthFrownLeft=0.6, mouthFrownRight=0.6` | `emotion="sad"` |
| `test_neutral_detection` | 全ブレンドシェイプが閾値未満 | `emotion="neutral"` |
| `test_ambiguous_emotion` | 複数の感情条件を同時に満たす | 最も強い感情が選択される |
| `test_empty_blendshapes` | 空のブレンドシェイプ辞書 | `emotion="neutral"` |

#### ExpressionRecognizer（モック使用）

| テストケース | 入力 | 期待結果 |
|---|---|---|
| `test_recognizer_no_face` | 顔のない画像（モック結果） | `None` |
| `test_recognizer_returns_result` | 正常なブレンドシェイプ（モック結果） | `ExpressionResult` インスタンス |
| `test_recognizer_result_fields` | 正常結果 | `emotion`, `confidence`, `blendshapes` が全て存在 |

---

## 2. 統合テスト（モック使用）

### 2.1 test_camera.py

**前提:** `pyrealsense2` をモックし、実機なしでテスト。

```python
@pytest.fixture
def mock_pipeline(mocker):
    """pyrealsense2.pipeline のモック。"""
    mock_rs = mocker.patch("src.camera.rs")

    # モックフレームの作成
    mock_depth_frame = MagicMock()
    mock_depth_frame.get_data.return_value = np.zeros((480, 640), dtype=np.uint16)

    mock_color_frame = MagicMock()
    mock_color_frame.get_data.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_frameset = MagicMock()
    mock_frameset.get_depth_frame.return_value = mock_depth_frame
    mock_frameset.get_color_frame.return_value = mock_color_frame

    mock_pipeline = MagicMock()
    mock_pipeline.wait_for_frames.return_value = mock_frameset

    mock_rs.pipeline.return_value = mock_pipeline
    return mock_rs
```

| テストケース | 内容 | 期待結果 |
|---|---|---|
| `test_camera_start_stop` | `start()` → `stop()` | pipeline.start/stop が呼ばれる |
| `test_camera_get_frame_returns_framedata` | `get_frame()` | `FrameData` インスタンスを返す |
| `test_camera_get_frame_color_shape` | フレームのカラー画像形状 | `(480, 640, 3)` |
| `test_camera_get_frame_depth_shape` | フレームの深度画像形状 | `(480, 640)` |
| `test_camera_get_frame_has_intrinsics` | フレームに intrinsics が含まれる | 非 None |
| `test_camera_get_frame_has_timestamp` | フレームにタイムスタンプが含まれる | `float` 型 |
| `test_camera_filters_initialized_once` | フィルタオブジェクト | `__init__` で1回だけ生成 |
| `test_camera_alignment_to_color` | アライメント方向 | `rs.stream.color` に対してアライン |
| `test_camera_filter_chain_order` | フィルタ適用順序 | spatial → temporal → hole_filling |
| `test_camera_timeout_retry` | `wait_for_frames` が RuntimeError | リトライ後に None を返す |
| `test_camera_no_frame_returns_none` | depth_frame=None | `None` を返す |

#### camera_thread 統合テスト

| テストケース | 内容 | 期待結果 |
|---|---|---|
| `test_camera_thread_pushes_to_queue` | スレッド起動 → キュー確認 | キューに FrameData が入る |
| `test_camera_thread_stops_on_event` | `stop_event.set()` | スレッドが終了する |
| `test_camera_thread_drops_old_frames` | キューフル時の動作 | 古いフレームが破棄される |

---

### 2.2 test_processor.py

**前提:** MediaPipe と深度処理をモックし、統合動作をテスト。

```python
@pytest.fixture
def mock_frame_data():
    """テスト用の FrameData。"""
    return FrameData(
        color_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        depth_image=np.full((480, 640), 1000, dtype=np.uint16),
        depth_frame=MagicMock(),
        intrinsics=MagicMock(),
        timestamp=time.monotonic(),
    )
```

| テストケース | 内容 | 期待結果 |
|---|---|---|
| `test_processor_returns_result` | 正常フレーム処理 | `ProcessingResult` インスタンス |
| `test_processor_result_has_color` | 結果のカラー画像 | `(480, 640, 3)` の ndarray |
| `test_processor_no_pose_detected` | ポーズ未検出（モック） | `landmarks_2d=None, keypoints_3d=None` |
| `test_processor_pose_landmarks_count` | 33 ランドマーク検出（モック） | `len(landmarks_2d) == 33` |
| `test_processor_3d_coords_are_meters` | 3D座標 | 値が妥当な範囲（-5.0〜5.0m） |
| `test_processor_expression_skip_frames` | N フレーム連続処理 | 表情認識は N フレームに1回 |
| `test_processor_expression_cached` | スキップフレーム | 前回の表情結果を再利用 |
| `test_processor_writeable_flag` | MediaPipe 呼び出し時 | `image.flags.writeable = False` が設定される |
| `test_processor_fps_calculation` | 連続処理 | `processing_fps > 0` |
| `test_processor_close_releases_resources` | `close()` 呼び出し | 全リソースが解放される |

#### processing_thread 統合テスト

| テストケース | 内容 | 期待結果 |
|---|---|---|
| `test_processing_thread_consumes_queue` | frame_queue にデータ投入 | result_queue にデータが出る |
| `test_processing_thread_stops_on_event` | `stop_event.set()` | スレッドが終了する |

---

## 3. E2E テスト（実機必要、手動）

### 3.1 パフォーマンステスト

| テスト項目 | 手順 | 合格基準 |
|---|---|---|
| FPS 計測 | `python -m src.main` を実行し、画面左上の FPS 表示を確認 | 30fps 以上 |
| レイテンシ確認 | 手を振って反応の遅延を目視確認 | 知覚可能な遅延なし（~100ms以下） |
| 長時間安定性 | 5分間連続実行 | FPS の低下なし、メモリリークなし |
| CPU 使用率 | タスクマネージャーで確認 | 80% 以下 |

### 3.2 3D 座標の妥当性確認

| テスト項目 | 手順 | 合格基準 |
|---|---|---|
| 深度精度 | カメラから 1m の位置に立ち、鼻のキーポイントの z 座標を確認 | 0.9〜1.1m |
| 左右対称性 | 正面を向いて両肩の x 座標を比較 | 左右でほぼ対称 |
| 身長推定 | 頭頂から足首までの y 座標差を実測値と比較 | 誤差 10% 以内 |
| 深度欠損フォールバック | 黒い服を着て深度欠損を意図的に発生 | フォールバック値で補完される |
| 範囲外フィルタリング | カメラから 5m 以上離れる | 座標が None になる |

### 3.3 表情認識の確認

| テスト項目 | 手順 | 合格基準 |
|---|---|---|
| 笑顔検出 | 笑顔を作る | `happy` ラベル表示 |
| 驚き検出 | 目と口を大きく開く | `surprise` ラベル表示 |
| ニュートラル | 無表情 | `neutral` ラベル表示 |
| 表情遷移 | 笑顔 → 無表情 → 驚き | スムーズに遷移 |

### 3.4 異常系テスト

| テスト項目 | 手順 | 合格基準 |
|---|---|---|
| カメラ未接続起動 | L515 を外した状態で起動 | エラーメッセージ表示後に終了 |
| USB 2.0 接続 | USB 2.0 ポートに接続 | 警告メッセージ表示 |
| フレーム外の人物 | カメラの視野外に出る | クラッシュせず、骨格描画が消える |
| 複数人検出 | 2人がカメラに映る | 1人のみ検出（`num_poses=1`） |
| 'q' キー終了 | 実行中に 'q' キー押下 | グレースフルシャットダウン |

---

## テストユーティリティ

### conftest.py に配置する共通フィクスチャ

```python
# tests/conftest.py
import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """テスト用の 640x480 BGR 画像。"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_image() -> np.ndarray:
    """テスト用の 640x480 深度画像（全ピクセル 1000mm）。"""
    return np.full((480, 640), 1000, dtype=np.uint16)


@pytest.fixture
def zero_depth_image() -> np.ndarray:
    """テスト用の全ゼロ深度画像。"""
    return np.zeros((480, 640), dtype=np.uint16)


@pytest.fixture
def mock_intrinsics() -> MagicMock:
    """テスト用のカメラ内部パラメータ。"""
    intrinsics = MagicMock()
    intrinsics.width = 640
    intrinsics.height = 480
    intrinsics.ppx = 320.0
    intrinsics.ppy = 240.0
    intrinsics.fx = 600.0
    intrinsics.fy = 600.0
    intrinsics.model = MagicMock()
    intrinsics.coeffs = [0.0] * 5
    return intrinsics
```
