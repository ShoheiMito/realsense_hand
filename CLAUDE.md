# RealSense L515 Hand Gesture Control

## プロジェクト概要
Intel RealSense L515（LiDARデプスカメラ）の手指トラッキングを使用したPC画面操作システム。
手指ジェスチャーでマウスカーソル移動・クリック・ドラッグ・スクロールを実現する。

## 技術スタック
- Python 3.10+
- pyrealsense2（RealSense SDK）
- MediaPipe HandLandmarker（手指21点ランドマーク検出）
- pynput（マウス制御）
- OpenCV（描画・表示）
- NumPy（数値計算）

## アーキテクチャ
3スレッド構成のプロデューサー・コンシューマーパターン:
- スレッド1（カメラ）: RealSense L515 → アライン → 深度フィルタ → フレームQueue
- スレッド2（処理）: 手指ランドマーク検出 → 3Dデプロジェクション → スムージング → 結果Queue
- スレッド3（メイン）: ジェスチャー認識 → マウス操作 + 手描画 → cv2.imshow

## ディレクトリ構造
```
realsense_hand/
├── CLAUDE.md              # このファイル
├── README.md              # プロジェクト説明
├── requirements.txt       # 依存パッケージ
├── models/                # MediaPipeモデルファイル
│   └── hand_landmarker.task
├── src/
│   ├── __init__.py
│   ├── main.py            # エントリーポイント
│   ├── camera.py          # RealSenseカメラ管理（スレッド1）
│   ├── processor.py       # 手指ランドマーク検出（スレッド2）
│   ├── hand_controller.py # ジェスチャー認識 + マウス制御
│   ├── visualizer.py      # 手描画・コントロールオーバーレイ
│   ├── depth_utils.py     # 深度フィルタリング・3Dデプロジェクション
│   └── config.py          # 設定・定数
└── tests/
    ├── test_hand_controller.py
    ├── test_camera.py
    └── test_depth_utils.py
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
- 単体テスト: `pytest tests/test_hand_controller.py -v`
- カバレッジ: `pytest --cov=src tests/`

## 型チェック
- mypy を使用: `mypy src/ --ignore-missing-imports`

## リント
- ruff を使用: `ruff check src/`
- フォーマット: `ruff format src/`

## ビルド・実行
- 実行: `python -m src.main`
- マウス制御なし（表示のみ）: `python -m src.main --no-control`
- 依存インストール: `pip install -r requirements.txt`

## ジェスチャー操作
| 操作 | ジェスチャー |
|------|------------|
| カーソル移動 | 人差し指を立てる |
| クリック | ピンチ（親指+人差し指）して素早く離す |
| ドラッグ | ピンチしたまま移動 |
| スクロール | 人差し指+中指を立てて上下移動 |

## ランタイムキーボード操作
- `c` — マウス制御ON/OFF切替
- `q` — 終了

## RealSense L515 固有の注意事項
- 赤外線ストリームはインデックス0のみ（1を指定するとRuntimeError）
- USB 3.x接続を必ず確認（USB 2.0だと深度320×240に制限）
- カラーと深度は同じ解像度で設定（640×480推奨）
- 深度フィルタチェーン: spatial → temporal → hole_filling の順
- ディスパリティ変換は不要（LiDARのため）

## 重要な設計判断
- Queue の maxsize=2 で古いフレームを破棄し、レイテンシ蓄積を防ぐ
- 3Dキーポイントには One Euro フィルタで時間的スムージング適用
- 深度欠損値は5×5近傍のメディアンでフォールバック
- ピンチ検出にヒステリシス適用（閾値30px/解除40px）で誤検出防止
- 座標マッピングはカメラ中央70%領域→画面全体（端まで手を伸ばさなくてよい）
- EMA平滑化 + デッドゾーンでカーソルの微振動防止

## コンパクション時の注意
コンパクション時は以下を必ず保持すること:
- 変更済みファイルの完全なリスト
- 現在のテスト状態とテストコマンド
- 直近で発見した問題点とその対処方針
- RealSense L515 固有の注意事項
