# RealSense L515 3D Pose Estimation + Expression Recognition

## プロジェクト概要
Intel RealSense L515（LiDARデプスカメラ）を使用したリアルタイム3Dポーズ推定＋表情認識システム。
研究・プロトタイプ用途。30fps以上のリアルタイム処理を目標とする。

## 技術スタック
- Python 3.10+
- pyrealsense2（RealSense SDK）
- MediaPipe（ポーズ推定 + 顔ランドマーク/ブレンドシェイプ）
- OpenCV（描画・表示）
- NumPy（数値計算）
- onnxruntime（将来的なRTMPose/HSEmotion統合用）

## アーキテクチャ
3スレッド構成のプロデューサー・コンシューマーパターン:
- スレッド1（カメラ）: RealSense L515 → アライン → 深度フィルタ → フレームQueue
- スレッド2（処理）: ポーズ推定 → 3Dデプロジェクション → 表情認識 → 結果Queue
- スレッド3（メイン）: 結果読み取り → 骨格描画 + 感情ラベル → cv2.imshow

## ディレクトリ構造
```
realsense-pose/
├── CLAUDE.md              # このファイル
├── README.md              # プロジェクト説明
├── requirements.txt       # 依存パッケージ
├── src/
│   ├── __init__.py
│   ├── main.py            # エントリーポイント
│   ├── camera.py          # RealSenseカメラ管理（スレッド1）
│   ├── processor.py       # ポーズ推定＋表情認識（スレッド2）
│   ├── visualizer.py      # 骨格描画・表示（スレッド3）
│   ├── depth_utils.py     # 深度フィルタリング・3Dデプロジェクション
│   ├── expression.py      # 表情認識モジュール
│   └── config.py          # 設定・定数
├── tests/
│   ├── test_camera.py
│   ├── test_depth_utils.py
│   ├── test_processor.py
│   └── test_expression.py
└── docs/
    └── architecture.md    # アーキテクチャ詳細
```

## コーディング規約
- 型ヒントを必ず使用する（Python 3.10+ スタイル）
- docstring は Google スタイル
- 変数名・関数名は snake_case、クラス名は PascalCase
- コメントは日本語OK、docstringは英語
- f-string を優先（.format() は使わない）
- import は isort 準拠（stdlib → third-party → local）

## テスト
- pytest を使用
- テスト実行: `pytest tests/ -v`
- 単体テスト: `pytest tests/test_depth_utils.py -v`
- カバレッジ: `pytest --cov=src tests/`

## 型チェック
- mypy を使用: `mypy src/ --ignore-missing-imports`

## リント
- ruff を使用: `ruff check src/`
- フォーマット: `ruff format src/`

## ビルド・実行
- 実行: `python -m src.main`
- 依存インストール: `pip install -r requirements.txt`

## RealSense L515 固有の注意事項
- 赤外線ストリームはインデックス0のみ（1を指定するとRuntimeError）
- USB 3.x接続を必ず確認（USB 2.0だと深度320×240に制限）
- カラーと深度は同じ解像度で設定（640×480推奨）
- 深度フィルタチェーン: spatial → temporal → hole_filling の順
- ディスパリティ変換は不要（LiDARのため）

## 重要な設計判断
- Queue の maxsize=2 で古いフレームを破棄し、レイテンシ蓄積を防ぐ
- 表情認識は2〜3フレームに1回実行（知覚的に差なし、CPU負荷軽減）
- 3Dキーポイントには One Euro フィルタで時間的スムージング適用
- 深度欠損値は5×5近傍のメディアンでフォールバック

## コンパクション時の注意
コンパクション時は以下を必ず保持すること:
- 変更済みファイルの完全なリスト
- 現在のテスト状態とテストコマンド
- 直近で発見した問題点とその対処方針
- RealSense L515 固有の注意事項
