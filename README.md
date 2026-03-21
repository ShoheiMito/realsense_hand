# RealSense L515 3D Pose Estimation + Expression Recognition

Intel RealSense L515（LiDARデプスカメラ）を使用したリアルタイム3Dポーズ推定＋表情認識システム。

## 特徴

- リアルタイム3Dポーズ推定（MediaPipe）
- 表情認識（顔ランドマーク/ブレンドシェイプ）
- 30fps以上のリアルタイム処理
- 3スレッド構成による効率的なパイプライン処理

## 必要条件

- Python 3.10+
- Intel RealSense L515
- USB 3.x接続

## インストール

```bash
pip install -r requirements.txt
```

## 実行

```bash
python -m src.main
```

## テスト

```bash
pytest tests/ -v
```

## 型チェック

```bash
mypy src/ --ignore-missing-imports
```

## リント

```bash
ruff check src/
ruff format src/
```

## アーキテクチャ

3スレッド構成のプロデューサー・コンシューマーパターン:

1. **カメラスレッド**: RealSense L515 → アライン → 深度フィルタ → フレームQueue
2. **処理スレッド**: ポーズ推定 → 3Dデプロジェクション → 表情認識 → 結果Queue
3. **メインスレッド**: 結果読み取り → 骨格描画 + 感情ラベル → cv2.imshow

詳細は [docs/architecture.md](docs/architecture.md) を参照。
