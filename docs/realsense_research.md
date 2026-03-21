# Intel RealSense L515 pyrealsense2 技術調査レポート

本ドキュメントは、Intel RealSense L515 LiDARカメラとpyrealsense2ライブラリの技術仕様について調査した結果をまとめたものです。

---

## 目次

1. [L515で利用可能なストリームタイプと解像度の組み合わせ](#1-l515で利用可能なストリームタイプと解像度の組み合わせ)
2. [rs.align()の仕様とパフォーマンスへの影響](#2-rsalignの仕様とパフォーマンスへの影響)
3. [rs2_deproject_pixel_to_point()の入出力仕様](#3-rs2_deproject_pixel_to_pointの入出力仕様)
4. [深度フィルターパラメータの詳細](#4-深度フィルターパラメータの詳細)
5. [pipeline.wait_for_frames()のタイムアウト動作](#5-pipelinewait_for_framesのタイムアウト動作)

---

## 1. L515で利用可能なストリームタイプと解像度の組み合わせ

### L515の特徴

L515はIntel RealSenseシリーズの**LiDAR（Light Detection and Ranging）カメラ**です。従来のD400シリーズがステレオビジョン方式を採用しているのに対し、L515は**固体LiDAR技術**を使用しており、以下の特徴があります：

- 高精度な深度測定（0.25m〜9mの範囲）
- 低消費電力（約3.5W）
- コンパクトな設計
- 室内環境での高い精度

### ストリームタイプ一覧

| ストリームタイプ | 説明 |
|-----------------|------|
| `rs.stream.depth` | 深度ストリーム（LiDARセンサー） |
| `rs.stream.infrared` | 赤外線ストリーム |
| `rs.stream.color` | RGBカラーストリーム |
| `rs.stream.confidence` | 深度信頼度ストリーム |
| `rs.stream.gyro` | ジャイロスコープデータ（IMU） |
| `rs.stream.accel` | 加速度計データ（IMU） |

### 解像度とフレームレートの組み合わせ

#### 深度ストリーム (Depth)

| 解像度 | フレームレート (fps) | 備考 |
|--------|---------------------|------|
| 1024 x 768 | 30 | 最高解像度 |
| 640 x 480 | 30 | 標準解像度 |
| 320 x 240 | 30 | 低解像度（高速処理向け） |

#### カラーストリーム (Color)

| 解像度 | フレームレート (fps) | 備考 |
|--------|---------------------|------|
| 1920 x 1080 | 30 | フルHD |
| 1280 x 720 | 30, 60 | HD（推奨） |
| 960 x 540 | 30, 60 | qHD |
| 640 x 480 | 30, 60 | VGA |
| 640 x 360 | 30, 60 | - |
| 424 x 240 | 30, 60 | 低解像度 |
| 320 x 240 | 30, 60 | QVGA |
| 320 x 180 | 30, 60 | 最小解像度 |

#### 赤外線ストリーム (Infrared)

| 解像度 | フレームレート (fps) |
|--------|---------------------|
| 1024 x 768 | 30 |
| 640 x 480 | 30 |

#### 信頼度ストリーム (Confidence)

| 解像度 | フレームレート (fps) |
|--------|---------------------|
| 1024 x 768 | 30 |
| 640 x 480 | 30 |

### コード例：ストリーム設定

```python
import pyrealsense2 as rs

# パイプラインとコンフィグの初期化
pipeline = rs.pipeline()
config = rs.config()

# L515用ストリーム設定
# 深度ストリーム: 640x480 @ 30fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# カラーストリーム: 1280x720 @ 30fps
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 信頼度ストリーム（オプション）
config.enable_stream(rs.stream.confidence, 640, 480, rs.format.raw8, 30)

# IMUストリーム（オプション）
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

# パイプライン開始
profile = pipeline.start(config)

# デバイス情報の取得
device = profile.get_device()
print(f"デバイス名: {device.get_info(rs.camera_info.name)}")
print(f"シリアル番号: {device.get_info(rs.camera_info.serial_number)}")
print(f"ファームウェア: {device.get_info(rs.camera_info.firmware_version)}")
```

### 利用可能なストリームの動的確認

```python
import pyrealsense2 as rs

# コンテキストからデバイスを取得
ctx = rs.context()
devices = ctx.query_devices()

for device in devices:
    print(f"\n=== {device.get_info(rs.camera_info.name)} ===")

    for sensor in device.sensors:
        print(f"\nセンサー: {sensor.get_info(rs.camera_info.name)}")

        for profile in sensor.get_stream_profiles():
            if isinstance(profile, rs.video_stream_profile):
                print(f"  ストリーム: {profile.stream_type()}, "
                      f"解像度: {profile.width()}x{profile.height()}, "
                      f"FPS: {profile.fps()}, "
                      f"フォーマット: {profile.format()}")
```

---

## 2. rs.align()の仕様とパフォーマンスへの影響

### 概要

`rs.align()`は、異なるセンサーからのストリームを空間的に位置合わせ（アライメント）するための機能です。カラー画像と深度画像のピクセル座標を一致させるために使用されます。

### アライメントの方向

```python
import pyrealsense2 as rs

# 深度フレームをカラーフレームに合わせる（推奨）
align_to_color = rs.align(rs.stream.color)

# カラーフレームを深度フレームに合わせる
align_to_depth = rs.align(rs.stream.depth)
```

### 動作原理

#### 1. Depth to Color アライメント（推奨）

```
深度フレーム (640x480) → 変換 → アライメント済み深度 (1280x720)
                                  ↓
カラーフレーム (1280x720) ←←←←← ピクセル対応
```

- 深度データがカラー画像の解像度に再投影される
- カラー画像の各ピクセルに対応する深度値が得られる
- RGBDアプリケーション（3D再構成、オブジェクト検出等）に最適

#### 2. Color to Depth アライメント

```
カラーフレーム (1280x720) → 変換 → アライメント済みカラー (640x480)
                                    ↓
深度フレーム (640x480) ←←←←←←←←←← ピクセル対応
```

- カラーデータが深度画像の解像度に再投影される
- 深度ベースの処理に適している

### コード例：基本的なアライメント処理

```python
import pyrealsense2 as rs
import numpy as np

# パイプライン設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# アライメントオブジェクトの作成
align = rs.align(rs.stream.color)

pipeline.start(config)

try:
    while True:
        # フレーム取得
        frames = pipeline.wait_for_frames()

        # アライメント実行
        aligned_frames = align.process(frames)

        # アライメント済みフレームの取得
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # NumPy配列に変換
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_imageとcolor_imageは同じ解像度（1280x720）
        # 各ピクセル座標が対応している

finally:
    pipeline.stop()
```

### パフォーマンスへの影響

#### 処理時間の目安

| 設定 | 処理時間 (参考値) | CPU使用率への影響 |
|------|------------------|------------------|
| 640x480 → 640x480 | 約1-2ms | 低 |
| 640x480 → 1280x720 | 約3-5ms | 中 |
| 1024x768 → 1920x1080 | 約8-15ms | 高 |

#### パフォーマンス最適化のヒント

1. **解像度の選択**
   - アライメント後の解像度が大きいほど処理負荷が増加
   - 必要に応じて低解像度を選択

```python
# パフォーマンス重視の設定例
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 同じ解像度
```

2. **処理頻度の制御**
   - 毎フレームではなく、必要な時のみアライメントを実行

```python
frame_count = 0
align_interval = 3  # 3フレームに1回アライメント

while True:
    frames = pipeline.wait_for_frames()
    frame_count += 1

    if frame_count % align_interval == 0:
        aligned_frames = align.process(frames)
        # アライメント済みフレームを使用した処理
    else:
        # アライメントなしの処理
        pass
```

3. **ハードウェアアクセラレーションの活用**
   - L515はデバイス内でのハードウェアアライメントに対応
   - `rs.sensor`のオプションで有効化可能

```python
# センサーオプションの確認と設定
depth_sensor = profile.get_device().first_depth_sensor()

# 利用可能なオプションを確認
for option in depth_sensor.get_supported_options():
    print(f"{option}: {depth_sensor.get_option(option)}")
```

### 注意事項

- アライメント処理では、視野角の違いにより一部のピクセルで深度値が欠損する場合がある
- L515の深度センサーとカラーセンサーは物理的に離れているため、近距離でオクルージョンが発生しやすい
- メモリ使用量：アライメント後のフレームは追加のメモリを消費する

---

## 3. rs2_deproject_pixel_to_point()の入出力仕様

### 概要

`rs2_deproject_pixel_to_point()`は、2Dピクセル座標と深度値から3D空間座標を計算する関数です。カメラの内部パラメータ（intrinsics）を使用してデプロジェクション（逆投影）を行います。

### 関数シグネチャ

```python
# Python (pyrealsense2)
point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
```

```c
// C言語 (librealsense2)
void rs2_deproject_pixel_to_point(
    float point[3],              // 出力: 3D座標 [x, y, z]
    const rs2_intrinsics* intrin, // 入力: カメラ内部パラメータ
    const float pixel[2],         // 入力: ピクセル座標 [u, v]
    float depth                   // 入力: 深度値（メートル）
);
```

### 入力パラメータ

#### 1. intrinsics (カメラ内部パラメータ)

```python
# intrinsicsの取得方法
profile = pipeline.start(config)
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# intrinsicsの内容
print(f"幅: {intrinsics.width}")          # 画像幅（ピクセル）
print(f"高さ: {intrinsics.height}")        # 画像高さ（ピクセル）
print(f"主点X (ppx): {intrinsics.ppx}")   # 主点のX座標
print(f"主点Y (ppy): {intrinsics.ppy}")   # 主点のY座標
print(f"焦点距離X (fx): {intrinsics.fx}") # X方向の焦点距離
print(f"焦点距離Y (fy): {intrinsics.fy}") # Y方向の焦点距離
print(f"歪みモデル: {intrinsics.model}")  # 歪み補正モデル
print(f"歪み係数: {intrinsics.coeffs}")   # 歪み係数 [k1, k2, p1, p2, k3]
```

#### 2. pixel (ピクセル座標)

- 形式: `[u, v]` または `(u, v)`
- `u`: 画像の水平方向座標（左から右、0〜width-1）
- `v`: 画像の垂直方向座標（上から下、0〜height-1）

#### 3. depth (深度値)

- 単位: **メートル**
- 深度フレームからの値を深度スケールで変換する必要がある

```python
# 深度スケールの取得
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度スケール: {depth_scale}")  # L515では通常 0.00025 または 0.0001

# 深度値の変換
raw_depth = depth_frame.get_distance(x, y)  # メートル単位で取得
# または
raw_depth_value = depth_image[y, x]  # 生の深度値（整数）
depth_in_meters = raw_depth_value * depth_scale
```

### 出力

- 形式: `[x, y, z]` （3要素のリストまたはタプル）
- 座標系: カメラ座標系
  - X: 右方向が正
  - Y: 下方向が正
  - Z: カメラから離れる方向（前方）が正
- 単位: メートル

### 数学的原理

デプロジェクションは以下の式で計算されます：

```
x = (u - ppx) / fx * depth
y = (v - ppy) / fy * depth
z = depth
```

歪み補正が適用される場合は、まずピクセル座標の歪み補正が行われます。

### コード例：完全な使用例

```python
import pyrealsense2 as rs
import numpy as np

def pixel_to_point_example():
    """ピクセル座標から3D座標への変換例"""

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 内部パラメータの取得
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    # 深度スケールの取得
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if depth_frame:
            # 例: 画像中心のピクセルを3D座標に変換
            center_x = intrinsics.width // 2
            center_y = intrinsics.height // 2

            # 深度値の取得（メートル単位）
            depth_value = depth_frame.get_distance(center_x, center_y)

            if depth_value > 0:  # 有効な深度値の場合
                # 3D座標の計算
                pixel = [center_x, center_y]
                point_3d = rs.rs2_deproject_pixel_to_point(
                    intrinsics, pixel, depth_value
                )

                print(f"ピクセル座標: ({center_x}, {center_y})")
                print(f"深度値: {depth_value:.3f} m")
                print(f"3D座標: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f} m")

    finally:
        pipeline.stop()

# 複数ピクセルの一括変換（効率的な方法）
def batch_deproject(depth_frame, intrinsics):
    """深度フレーム全体をポイントクラウドに変換"""

    depth_image = np.asanyarray(depth_frame.get_data())

    # ピクセル座標のグリッドを作成
    height, width = depth_image.shape
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 深度値（メートル単位）
    z = depth_image * depth_frame.get_units()

    # 3D座標の計算
    x = (u - intrinsics.ppx) / intrinsics.fx * z
    y = (v - intrinsics.ppy) / intrinsics.fy * z

    # ポイントクラウドとして返す (N x 3)
    valid_mask = z > 0
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)

    return points
```

### PointCloudクラスを使用した方法（推奨）

```python
import pyrealsense2 as rs
import numpy as np

def generate_pointcloud():
    """ポイントクラウドの生成（組み込み機能使用）"""

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # ポイントクラウドオブジェクト
    pc = rs.pointcloud()

    # カラーフレームへのアライメント
    align = rs.align(rs.stream.color)

    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if depth_frame and color_frame:
            # テクスチャマッピング用にカラーフレームを設定
            pc.map_to(color_frame)

            # ポイントクラウドの計算
            points = pc.calculate(depth_frame)

            # 頂点データの取得
            vertices = np.asanyarray(points.get_vertices()).view(np.float32)
            vertices = vertices.reshape(-1, 3)  # (N, 3) の形状

            # テクスチャ座標の取得
            tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32)
            tex_coords = tex_coords.reshape(-1, 2)  # (N, 2) の形状

            return vertices, tex_coords

    finally:
        pipeline.stop()
```

---

## 4. 深度フィルターパラメータの詳細

### 概要

pyrealsense2には、深度データの品質を向上させるための複数のポストプロセッシングフィルターが用意されています。L515（LiDARカメラ）でも使用可能ですが、ステレオカメラとは異なる特性を考慮する必要があります。

### フィルターの種類と処理順序

推奨される処理順序：

```
深度フレーム
    ↓
① Decimation Filter（オプション）
    ↓
② Depth to Disparity Transform
    ↓
③ Spatial Filter
    ↓
④ Temporal Filter
    ↓
⑤ Disparity to Depth Transform
    ↓
⑥ Hole Filling Filter
    ↓
フィルタリング済み深度フレーム
```

### 4.1 Spatial Filter（空間フィルター）

#### 概要
エッジを保持しながら深度画像のノイズを低減する空間平滑化フィルターです。

#### パラメータ

| パラメータ | 説明 | 範囲 | デフォルト | L515推奨値 |
|-----------|------|------|-----------|-----------|
| `filter_magnitude` | フィルター反復回数 | 1-5 | 2 | 2-3 |
| `filter_smooth_alpha` | 平滑化の強さ（α値） | 0.25-1.0 | 0.5 | 0.4-0.6 |
| `filter_smooth_delta` | エッジ保持の閾値 | 1-50 | 20 | 15-25 |
| `holes_fill` | 穴埋めモード | 0-5 | 0 | 1-2 |

#### パラメータの詳細説明

- **filter_magnitude**: 値が大きいほど平滑化が強くなるが、処理時間が増加
- **filter_smooth_alpha**: 値が小さいほどエッジが保持される、値が大きいほど平滑化が強い
- **filter_smooth_delta**: 深度差がこの値より大きいピクセルはエッジとして扱われ、平滑化されない
- **holes_fill**: 穴埋め時に使用するピクセル数（0=なし）

#### コード例

```python
import pyrealsense2 as rs

# Spatial Filterの設定
spatial_filter = rs.spatial_filter()

# パラメータの設定
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
spatial_filter.set_option(rs.option.holes_fill, 2)

# フィルターの適用
filtered_depth = spatial_filter.process(depth_frame)
```

### 4.2 Temporal Filter（時間フィルター）

#### 概要
複数フレーム間で深度値を平滑化することでノイズを低減します。動きの少ないシーンで特に効果的です。

#### パラメータ

| パラメータ | 説明 | 範囲 | デフォルト | L515推奨値 |
|-----------|------|------|-----------|-----------|
| `filter_smooth_alpha` | 現在フレームの重み | 0.0-1.0 | 0.4 | 0.3-0.5 |
| `filter_smooth_delta` | 深度変化の閾値 | 1-100 | 20 | 20-40 |
| `holes_fill` | 穴埋め動作モード | 0-8 | 3 | 3-4 |

#### パラメータの詳細説明

- **filter_smooth_alpha**: 時間的平滑化の強さ
  - 0に近い: 過去フレームの影響が強い（より滑らか、レイテンシ増加）
  - 1に近い: 現在フレームの影響が強い（変化に敏感）

- **filter_smooth_delta**: この閾値を超える深度変化は動きとして扱われ、平滑化されない

- **holes_fill**: 穴埋めのための持続性モード
  - 0: 穴埋めなし
  - 1-8: 数字が大きいほど積極的に過去の値で穴を埋める

#### コード例

```python
import pyrealsense2 as rs

# Temporal Filterの設定
temporal_filter = rs.temporal_filter()

# パラメータの設定
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
temporal_filter.set_option(rs.option.holes_fill, 3)

# フィルターの適用（時系列データが必要）
filtered_depth = temporal_filter.process(depth_frame)
```

### 4.3 Hole Filling Filter（穴埋めフィルター）

#### 概要
深度値が欠損している領域（穴）を周囲のピクセル値で埋めるフィルターです。

#### パラメータ

| パラメータ | 説明 | 値 | 説明 |
|-----------|------|-----|------|
| `holes_fill` | 穴埋めモード | 0 | 穴埋めを行わない |
| | | 1 | 最も近い有効な深度値で埋める（farest from around） |
| | | 2 | 周囲から最も遠い深度値で埋める（nearest from around） |

#### モードの詳細

- **モード0 (fill_from_left)**: 左側の有効ピクセルの値を使用
- **モード1 (farest_from_around)**: 周囲のピクセルの中で最も遠い（大きい）深度値を使用
- **モード2 (nearest_from_around)**: 周囲のピクセルの中で最も近い（小さい）深度値を使用

#### コード例

```python
import pyrealsense2 as rs

# Hole Filling Filterの設定
hole_filling_filter = rs.hole_filling_filter()

# モードの設定（0, 1, 2のいずれか）
hole_filling_filter.set_option(rs.option.holes_fill, 1)

# フィルターの適用
filled_depth = hole_filling_filter.process(depth_frame)
```

### 4.4 完全なフィルターパイプラインの例

```python
import pyrealsense2 as rs
import numpy as np

class DepthFilterPipeline:
    """L515用深度フィルターパイプライン"""

    def __init__(self):
        # フィルターの初期化
        self.decimation = rs.decimation_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.disparity_to_depth = rs.disparity_transform(False)
        self.hole_filling = rs.hole_filling_filter()

        # L515用推奨パラメータの設定
        self._configure_for_l515()

    def _configure_for_l515(self):
        """L515に最適化されたパラメータ設定"""

        # Decimation Filter（解像度を下げる場合のみ使用）
        self.decimation.set_option(rs.option.filter_magnitude, 2)

        # Spatial Filter
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 2)

        # Temporal Filter
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal.set_option(rs.option.holes_fill, 3)

        # Hole Filling Filter
        self.hole_filling.set_option(rs.option.holes_fill, 1)

    def process(self, depth_frame, use_decimation=False):
        """
        深度フレームにフィルターを適用

        Args:
            depth_frame: 入力深度フレーム
            use_decimation: 解像度を下げる場合はTrue

        Returns:
            フィルタリング済み深度フレーム
        """
        frame = depth_frame

        # 1. Decimation（オプション）
        if use_decimation:
            frame = self.decimation.process(frame)

        # 2. Depth to Disparity
        frame = self.depth_to_disparity.process(frame)

        # 3. Spatial Filter
        frame = self.spatial.process(frame)

        # 4. Temporal Filter
        frame = self.temporal.process(frame)

        # 5. Disparity to Depth
        frame = self.disparity_to_depth.process(frame)

        # 6. Hole Filling
        frame = self.hole_filling.process(frame)

        return frame


# 使用例
def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    filter_pipeline = DepthFilterPipeline()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if depth_frame:
                # フィルター適用
                filtered_depth = filter_pipeline.process(depth_frame)

                # NumPy配列に変換
                depth_image = np.asanyarray(filtered_depth.get_data())

                # 処理を続行...

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
```

### L515特有の注意事項

L515はLiDARカメラであるため、ステレオカメラ（D400シリーズ）とは異なる特性があります：

1. **ノイズ特性**: LiDARは一般的にステレオカメラよりノイズが少ないため、フィルターを控えめに設定できる場合がある

2. **エッジ精度**: LiDARはエッジ部分での精度が高いため、Spatial Filterのdelta値を適切に設定することが重要

3. **処理負荷**: L515の深度データは既に高品質なため、過度なフィルタリングは処理負荷の増加につながる可能性がある

4. **推奨設定**:
   - 静的なシーン: Temporal Filterを積極的に使用
   - 動的なシーン: Spatial Filterを中心に、Temporalは控えめに

---

## 5. pipeline.wait_for_frames()のタイムアウト動作

### 概要

`pipeline.wait_for_frames()`は、デバイスからフレームセットを同期的に取得するブロッキング関数です。フレームが到着するまで待機し、タイムアウトが発生した場合は例外をスローします。

### 関数シグネチャ

```python
# Python
frames = pipeline.wait_for_frames(timeout_ms=5000)
```

```cpp
// C++
rs2::frameset wait_for_frames(unsigned int timeout_ms = 5000) const;
```

### パラメータ

| パラメータ | 型 | デフォルト値 | 説明 |
|-----------|-----|------------|------|
| `timeout_ms` | int | 5000 | タイムアウト時間（ミリ秒） |

### 戻り値

- `rs.composite_frame` (frameset): 複数のフレームを含むフレームセット
- 設定したストリームのフレームが同期されて返される

### タイムアウト動作

#### タイムアウト発生時

タイムアウト時間内にフレームが到着しない場合、`RuntimeError`例外がスローされます。

```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    # 1秒（1000ms）でタイムアウト
    frames = pipeline.wait_for_frames(timeout_ms=1000)
except RuntimeError as e:
    print(f"タイムアウトエラー: {e}")
    # エラーメッセージ例: "Frame didn't arrive within 1000"
```

### タイムアウト設定の考慮事項

#### 1. フレームレートとの関係

| フレームレート | 1フレームの間隔 | 推奨タイムアウト |
|---------------|----------------|-----------------|
| 30 fps | 約33ms | 100-200ms以上 |
| 60 fps | 約17ms | 50-100ms以上 |
| 15 fps | 約67ms | 200-500ms以上 |

#### 2. 一般的な推奨値

```python
# 標準的なアプリケーション
frames = pipeline.wait_for_frames(timeout_ms=5000)  # デフォルト

# リアルタイム性が重要なアプリケーション
frames = pipeline.wait_for_frames(timeout_ms=1000)

# 初期化直後（デバイスの安定化を待つ）
frames = pipeline.wait_for_frames(timeout_ms=10000)
```

### エラーハンドリングのベストプラクティス

```python
import pyrealsense2 as rs
import time

class RealSenseCapture:
    """堅牢なフレーム取得クラス"""

    def __init__(self, timeout_ms=5000, max_retries=3):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.is_running = False

    def start(self):
        """パイプラインを開始"""
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(self.config)
        self.is_running = True

        # 最初の数フレームを破棄（デバイスの安定化）
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
            except RuntimeError:
                pass

    def get_frames(self):
        """
        フレームを取得（リトライ機能付き）

        Returns:
            frameset: 成功時
            None: 失敗時
        """
        if not self.is_running:
            return None

        for attempt in range(self.max_retries):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                return frames

            except RuntimeError as e:
                print(f"フレーム取得失敗 (試行 {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # 短い待機
                    continue
                else:
                    # 最後の試行も失敗した場合
                    return None

        return None

    def stop(self):
        """パイプラインを停止"""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False


# 使用例
def main():
    capture = RealSenseCapture(timeout_ms=2000, max_retries=3)

    try:
        capture.start()
        print("キャプチャ開始")

        while True:
            frames = capture.get_frames()

            if frames is None:
                print("フレーム取得に失敗しました。再接続を試みます...")
                capture.stop()
                time.sleep(1)
                capture.start()
                continue

            # フレーム処理
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame and color_frame:
                # 処理を続行...
                pass

    except KeyboardInterrupt:
        print("終了します")

    finally:
        capture.stop()


if __name__ == "__main__":
    main()
```

### 非ブロッキング代替手段: poll_for_frames()

`wait_for_frames()`の代わりに、非ブロッキングの`poll_for_frames()`を使用することもできます。

```python
import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        # 非ブロッキングでフレームを取得
        frames = pipeline.poll_for_frames()

        if frames:
            # フレームが利用可能
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # 処理を実行
                pass
        else:
            # フレームがまだ準備できていない
            # 他の処理を行うか、短い待機
            time.sleep(0.001)  # 1ms待機
            continue

finally:
    pipeline.stop()
```

### コールバックベースの取得方法

```python
import pyrealsense2 as rs
import threading
import queue

class AsyncFrameCapture:
    """非同期フレーム取得クラス"""

    def __init__(self, queue_size=2):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()

    def _frame_callback(self, frame):
        """コールバック関数"""
        try:
            # キューがフルの場合は古いフレームを破棄
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # フレームを破棄

    def start(self):
        """非同期キャプチャを開始"""
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # コールバックモードで開始
        self.pipeline.start(self.config, self._frame_callback)

    def get_frame(self, timeout=1.0):
        """
        フレームを取得

        Args:
            timeout: タイムアウト時間（秒）

        Returns:
            frameset または None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """キャプチャを停止"""
        self._stop_event.set()
        self.pipeline.stop()


# 使用例
capture = AsyncFrameCapture()
capture.start()

try:
    while True:
        frames = capture.get_frame(timeout=1.0)
        if frames:
            # フレーム処理
            pass
finally:
    capture.stop()
```

---

## 参考リンク

- [Intel RealSense SDK 2.0 公式ドキュメント](https://dev.intelrealsense.com/docs)
- [librealsense GitHub リポジトリ](https://github.com/IntelRealSense/librealsense)
- [pyrealsense2 API ドキュメント](https://intelrealsense.github.io/librealsense/python_docs/)
- [L515 製品ページ](https://www.intel.com/content/www/us/en/products/sku/201855/intel-realsense-lidar-camera-l515/specifications.html)

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2024-01-XX | 1.0 | 初版作成 |

---

*本ドキュメントはIntel RealSense L515とpyrealsense2ライブラリの技術調査結果をまとめたものです。実際の動作はファームウェアバージョンやSDKバージョンにより異なる場合があります。*
