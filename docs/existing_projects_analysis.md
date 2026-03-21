# 既存プロジェクト調査分析

RealSense + ポーズ推定に関する3つのGitHubプロジェクトを調査し、本プロジェクトの設計に活かすべきパターンとアンチパターンを整理した。

---

## 1. cansik/realsense-pose-detector

**リポジトリ構成:** 単一ファイル（`pose.py`）+ バッチファイル（`runpose.bat`）
**対象カメラ:** RealSense（カラーストリームのみ使用）
**ライセンス:** MIT

### 使用しているポーズ推定モデル

**MediaPipe BlazePose** を使用。`mp.solutions.pose.Pose()` で直接インスタンス化。

```python
pose = mp_pose.Pose(
    smooth_landmarks=args.no_smooth_landmarks,
    static_image_mode=args.static_image_mode,
    model_complexity=args.model_complexity,  # 0=Light, 1=Full, 2=Heavy
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence)
```

`argparse` でモデル複雑度・信頼度閾値を外部設定可能にしている。

### 深度データの統合方法

**深度データは使用していない。** RealSenseからカラーストリーム（`rs.stream.color`）のみを取得し、深度ストリームは有効化されていない。MediaPipeの `landmark.z`（相対的な深度推定値）をそのまま OSC で送信している。

```python
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
# depth stream は未設定
```

### スレッディングの有無とアーキテクチャ

**スレッディングなし。** 単一スレッドのシーケンシャルループ：

```
while True:
    フレーム取得 → BGR→RGB変換 → MediaPipe推論 → OSC送信 → 描画 → imshow
```

OSC（Open Sound Control）プロトコルで外部アプリケーション（TouchDesigner等）にランドマークデータを送信するアーキテクチャが特徴的。

### 参考にすべき実装パターン

**(a) `image.flags.writeable = False` の最適化**
```python
image.flags.writeable = False
results = pose.process(image)
image.flags.writeable = True
```
MediaPipe公式推奨パターン。不要なメモリコピーを回避し、パフォーマンスを向上させる。**本プロジェクトの `processor.py` で採用すべき。**

**(b) `argparse` による柔軟な設定**
モデル複雑度、解像度、FPS、信頼度閾値をコマンドライン引数で設定可能。本プロジェクトでは `config.py` で同等の設定管理を行う。

**(c) `try/finally` による確実なリソース解放**
```python
try:
    while True:
        # メインループ
finally:
    pose.close()
    pipeline.stop()
```

**(d) OSCによるデータ外部送信パターン**
将来的にUnity/TouchDesigner等と連携する場合に参考になるアーキテクチャ。

### 避けるべきアンチパターン

- **深度データ未使用**: RealSenseの最大の利点であるハードウェア深度を活用していない
- **単一ファイル構成**: テスタビリティ・保守性が低い
- **`cv2.flip(image, 1)`**: セルフィービュー用の左右反転を推論前に適用しており、座標系が反転する。ポーズ推定の用途（ロボティクス等）では不適切

---

## 2. SiaMahmoudi/MediaPipe-pose-estimation-using-intel-realsense-debth-camera

**リポジトリ構成:** `realsense_camera.py`（カメラクラス）+ `MediaPipe_Pose_detection_using_depth_camera.py`（メイン処理）+ `main.pdf`（ドキュメント）
**対象カメラ:** RealSense D435
**用途:** 四足歩行ロボットの知覚制御

### 使用しているポーズ推定モデル

**MediaPipe Pose（BlazePose）** を使用。33個のランドマークを検出し、姿勢分類（Hey, Ready, Go シーケンス）とジェスチャーコマンド生成に利用。

### 深度データの統合方法

RealSense D435からカラー+深度の両ストリームを取得し、アライメント後に深度値を統合。

```python
# realsense_camera.py
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# アライメント
align_to = rs.stream.color
self.align = rs.align(align_to)
```

**深度フィルタリング:**
- `rs.spatial_filter()` でホール充填（`holes_fill=3`）
- `rs.hole_filling_filter()` で追加のホール充填

```python
spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)
filtered_depth = spatial.process(depth_frame)

hole_filling = rs.hole_filling_filter()
filled_depth = hole_filling.process(filtered_depth)
```

**深度値の取得:** ランドマークのピクセル座標から `depth_image[y, x]` で直接深度値を参照。速度・加速度の算出にも深度差分を利用。

### スレッディングの有無とアーキテクチャ

**スレッディングなし。** 単一スレッドで以下を直列実行：
1. フレーム取得 + アライメント + フィルタ適用
2. MediaPipe推論
3. 深度値取得 + 速度/加速度算出
4. ジェスチャー分類
5. 描画 + 表示

### 参考にすべき実装パターン

**(a) `RealsenseCamera` クラスによるカメラ抽象化**
カメラの初期化・フレーム取得・リソース解放をクラスにカプセル化。本プロジェクトの `camera.py` と同じアプローチ。

**(b) 深度フィルタの適用パターン**
`spatial_filter` → `hole_filling_filter` のチェーン。ただし本プロジェクトでは `temporal_filter` も追加し、CLAUDE.mdの仕様（spatial → temporal → hole_filling）に従う。

**(c) 速度・加速度の時間差分計算**
フレーム間の深度差分から動きの速度を算出し、コマンド分類（GO FAST / GO SLOW）に利用。本プロジェクトの時系列データ活用の参考になる。

### 避けるべきアンチパターン

- **解像度 1280x720**: D435では問題ないが、L515では640x480が推奨（CLAUDE.md参照）。高解像度は処理負荷増大の原因
- **フィルタオブジェクトの毎フレーム再生成**: `get_frame_stream()` 内で毎回 `rs.spatial_filter()` と `rs.hole_filling_filter()` を新規生成している。初期化時に1回だけ生成し再利用すべき
- **`rs.colorizer()` の毎フレーム再生成**: 同様にコストのかかるオブジェクトを毎フレーム生成している
- **temporal_filter 未使用**: フレーム間の時間的一貫性を確保するフィルタが欠如
- **深度欠損のフォールバックなし**: `depth_image[y, x]` が0の場合の処理が未実装
- **`rs2_deproject_pixel_to_point` 未使用**: ピクセル座標→3Dワールド座標の変換を行っておらず、正確な3D位置が得られない

---

## 3. Razg93/Skeleton-Tracking-using-RealSense-depth-camera

**リポジトリ構成:** `main.py`（エントリーポイント + PoseDetector クラス）+ `realsense_camera.py`（カメラクラス）
**対象カメラ:** RealSense D435
**コミット数:** 10

### 使用しているポーズ推定モデル

**MediaPipe Pose（BlazePose）** を使用。`PoseDetector` クラスでラップし、信頼度閾値を設定可能にしている。

```python
class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.8, trackCon=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
```

### 深度データの統合方法

SiaMahmoudi プロジェクトと同じ `RealsenseCamera` クラスパターンを使用（コードがほぼ同一）。

**深度値の取得方法:**
```python
# ランドマークのピクセル座標から直接深度画像を参照
depth_to_object = depth_image[y, x]
text = "Depth: {} cm".format(depth_to_object / 10)
```

**特徴:**
- 特定のランドマーク（`object_to_track = [0,1,3,4,6,9,10]`：顔周辺のキーポイント）のみ深度を取得
- 深度値を mm → cm に変換して表示
- `rs2_deproject_pixel_to_point` は未使用（3D座標変換なし）

**深度フィルタリング:** SiaMahmoudi と同一のフィルタ構成（spatial + hole_filling）。

### スレッディングの有無とアーキテクチャ

**スレッディングなし。** 単一スレッドのシーケンシャルループ。

### 参考にすべき実装パターン

**(a) `PoseDetector` クラスによるポーズ推定の抽象化**
```python
detector = PoseDetector()
frame = detector.findPose(frame)
lmList = detector.getPosition(frame)
```
推論と結果取得を分離したインターフェース設計。本プロジェクトの `processor.py` で同様の分離を行うべき。

**(b) FPS計測パターン**
```python
cTime = time.time()
fps = 1 / (cTime - pTime)
pTime = cTime
```
シンプルだが実用的なFPS計測。本プロジェクトでも同様の計測を `visualizer.py` に組み込む。

**(c) 追跡対象ランドマークの選択**
全33ランドマークではなく、用途に応じた特定のランドマークのみを処理する設計。CPU負荷の軽減に有効。

### 避けるべきアンチパターン

- **`check_and_install_libraries()` によるランタイム pip install**: スクリプト実行時にライブラリの有無をチェックし、不足時に `pip install` を実行。セキュリティリスクが高く、requirements.txt + 仮想環境で管理すべき
- **`cv2.destroyAllWindows()` がメインループ外**: ループ終了後ではなくモジュールレベルに配置されており、`main()` が呼ばれなくても実行される
- **深度の直接配列参照**: `depth_image[y, x]` は座標が画像範囲外の場合に IndexError を発生させる。境界チェックが必要
- **フィルタオブジェクトの毎フレーム再生成**: SiaMahmoudi と同じ問題
- **PoseDetector の古い API 使用**: `upBody` パラメータなど、MediaPipe の古い API に依存している可能性がある

---

## 総合比較と本プロジェクトへの示唆

### 比較表

| 観点 | cansik | SiaMahmoudi | Razg93 | 本プロジェクト（設計） |
|---|---|---|---|---|
| ポーズモデル | MediaPipe BlazePose | MediaPipe BlazePose | MediaPipe BlazePose | MediaPipe + 将来RTMPose |
| 深度統合 | なし（カラーのみ） | pixel直接参照 | pixel直接参照 | deproject + メディアンフォールバック |
| 深度フィルタ | なし | spatial + hole_filling | spatial + hole_filling | spatial + temporal + hole_filling |
| スレッディング | なし | なし | なし | 3スレッド Producer-Consumer |
| モジュール構造 | 単一ファイル | 2ファイル | 2ファイル | 6+ モジュール |
| 時間的安定性 | MediaPipe内蔵のみ | なし | なし | One Euro フィルタ |
| 外部連携 | OSC送信 | ロボット制御 | なし | 将来検討 |
| 設定管理 | argparse | ハードコード | ハードコード | config.py |
| 表情認識 | なし | なし | なし | MediaPipe Face |

### 採用すべきパターン（優先度順）

1. **`image.flags.writeable = False`** (cansik) — MediaPipe推論前のメモリ最適化。`processor.py` で実装すべき
2. **`try/finally` によるリソース解放** (cansik) — カメラ・MediaPipeリソースの確実な解放
3. **`RealsenseCamera` クラス抽象化** (SiaMahmoudi/Razg93) — 本プロジェクトの `camera.py` に既に反映
4. **深度フィルタチェーン** (SiaMahmoudi/Razg93) — spatial + hole_filling の基本パターン。本プロジェクトでは temporal も追加
5. **`PoseDetector` クラスによる推論抽象化** (Razg93) — 推論と結果取得のインターフェース分離
6. **OSC外部送信** (cansik) — 将来のUnity/TouchDesigner連携時に参考

### 避けるべきアンチパターン（重要度順）

1. **フィルタオブジェクトの毎フレーム再生成** — 初期化時に1回だけ生成し再利用する
2. **深度欠損の未処理** — 5x5近傍メディアンフォールバック（CLAUDE.md仕様）で対応
3. **`rs2_deproject_pixel_to_point` 未使用** — ピクセル座標→3D座標変換を `depth_utils.py` で実装
4. **temporal_filter 未使用** — フレーム間の時間的一貫性を確保
5. **ランタイム pip install** — requirements.txt + 仮想環境で管理
6. **単一ファイル構成** — モジュール分割（既に設計済み）
7. **境界チェックなしの配列参照** — ランドマーク座標の画像範囲内チェック
8. **スレッディングなし** — 3スレッド構成で推論レイテンシとフレームレートを分離（既に設計済み）

### 結論

調査した3プロジェクトはすべて「MediaPipe + RealSense の最小限の統合例」であり、プロダクション品質には達していない。本プロジェクトの CLAUDE.md に記載されたアーキテクチャ設計（3スレッド構成、深度フィルタチェーン、One Euro フィルタ、表情認識統合）は、これらのプロジェクトの欠点をすべてカバーしている。

特に `image.flags.writeable = False` の最適化と、フィルタオブジェクトの再利用は、実装時に確実に取り入れるべき具体的な改善点である。
