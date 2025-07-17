# FullFront (元のREADMEの日本語訳)

FullFrontは、フロントエンド開発ワークフロー全体にわたってマルチモーダル大規模言語モデル（MLLMs）を評価するための包括的なベンチマークです。このプロジェクトは、フロントエンド開発の様々な段階におけるMLLMsのパフォーマンスを測定するためのコード生成、ページ理解、および評価ツールを提供します。

## プロジェクト概要

FullFrontベンチマークは、フロントエンド開発における3つの中核的なタスクを網羅しています：

1. **ウェブページデザイン** - ビジュアル要素を整理・構造化するモデルの能力を評価
2. **ウェブページ認識QA** - ビジュアル組織、要素特性、空間関係に対するモデルの理解を評価
3. **ウェブページコード生成** - ビジュアルデザインを機能的なコードに正確に変換する能力に焦点

## 主な機能

- 主要なマルチモーダルモデル（Claude、OpenAI、Gemini等）の評価をサポート
- 完全なコード生成・評価ワークフローを提供
- 画像類似度とコード品質評価メトリクスを含む
- 評価のためにHTMLを自動的に画像にレンダリング

## インストール

1. このリポジトリをクローンします：

```bash
git clone https://github.com/your-username/FullFront.git
cd FullFront
```

2. 依存関係をインストールします：

```bash
pip install -r requirements.txt
```

## 使用方法

### モデル応答の生成

`generate_response`ディレクトリには、異なるモデルからの応答を生成するスクリプトが含まれています：

1. **APIキーの設定**: 使用するモデルに基づいて、対応するスクリプトでAPIキーを設定します。

2. **生成スクリプトの実行**:

```bash
cd generate_response
python claude_code.py  # Claudeモデルを使用してコードを生成
python openai_code.py  # OpenAIモデルを使用してコードを生成
python gemini_code.py  # Geminiモデルを使用してコードを生成
```

3. **バッチ処理用のシェルスクリプトの使用**:

```bash
bash run_llava_code.sh  # LLaVAモデルのコード生成タスクを実行
bash run_qwen_qa.sh     # QwenモデルのQAタスクを実行
```

生成された結果は`generate_response/results/{model_name}`ディレクトリに保存されます。

### HTMLを画像にレンダリング

`calculate_similarity/render_img.py`を使用して、生成されたHTMLを画像にレンダリングします：

```bash
python calculate_similarity/render_img.py
```

このスクリプトで入力・出力ディレクトリを変更できます：

```python
html_folder = "./path/to/your/html/files"
screenshot_folder = "./path/to/save/screenshots"
```

### 類似度スコアの計算

1. **CLIP類似度**: 生成された画像とターゲット画像間のセマンティック類似度を評価

```bash
python calculate_similarity/clip_score.py
```

2. **コード類似度**: 生成されたコードと標準コード間の構造・内容類似度を評価

```bash
python calculate_similarity/code_score.py
```

3. **Gemini評価**: Geminiモデルを使用して生成されたコンテンツを評価

```bash
python calculate_similarity/gemini_evaluate.py
```

### 結果分析

評価結果は`calculate_similarity/results/`ディレクトリに保存され、以下のメトリクスが含まれます：

- CLIP類似度スコア
- コード構造類似度
- コード内容類似度
