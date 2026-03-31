---
marp: true
theme: default
paginate: true
math: katex
---

# 第11章　マルチモーダル学習と視覚・言語表現の統合

**視覚・言語・3D点群を共通の潜在空間に統合することで、ロボットは言語で指示された世界を「見て理解する」能力を獲得する**

---

## モダリティの異質性が、単純な統合を不可能にする

異なるモダリティは根本的に異なる構造を持つ：

| モダリティ | データ構造 | 典型的な表現 |
|-----------|-----------|------------|
| テキスト | 離散トークン列（系列的） | 埋め込みベクトル |
| 画像 | 連続ピクセルグリッド（空間的） | CNN特徴 / パッチ埋め込み |
| 3D点群 | 不規則な3D点集合（非グリッド） | PointNet埋め込み |
| 力覚センサ | 連続時系列 | 1D CNN / Transformer |

- **中心目標**：異なるモダリティを「共通の潜在空間」に埋め込むこと
- 現代の主流アプローチは**中間融合**：各モダリティの中間表現をクロスアテンションで統合

---

## CLIP：4億ペアの対照学習が視覚と言語を同一空間に写像する

**CLIP（Contrastive Language-Image Pretraining）**（Radford et al., 2021）の設計：

- **画像エンコーダ** $f$：ResNet / ViT → 正規化埋め込み $z_I = f(I) / \|f(I)\|_2$
- **テキストエンコーダ** $g$：Transformer → 正規化埋め込み $z_T = g(T) / \|g(T)\|_2$

対称クロスエントロピー損失（バッチサイズ $N$）を最小化：

$$\mathcal{L}_{\mathrm{CLIP}} = -\frac{1}{2N}\left(\sum_{i=1}^{N}\log \frac{\exp(\mathrm{sim}(z_{I_i}, z_{T_i})/\tau)}{\sum_{j=1}^{N}\exp(\mathrm{sim}(z_{I_i}, z_{T_j})/\tau)} + \sum_{i=1}^{N}\log \frac{\exp(\mathrm{sim}(z_{T_i}, z_{I_i})/\tau)}{\sum_{j=1}^{N}\exp(\mathrm{sim}(z_{T_i}, z_{I_j})/\tau)}\right)$$

$\tau$：学習可能な温度パラメータ、$\mathrm{sim}$：コサイン類似度

---

## CLIPはゼロショットで動作し、固定クラス集合に縛られない

CLIPが実現する能力：

- **Zero-Shot転移**：クラスラベルをテキストとして表現し（「a photo of a [class]」）、$\arg\max_{c} \mathrm{sim}(z_I, z_{T_c})$ でゼロショット分類を実現
  - ImageNetでのゼロショット精度は、ResNet-50の教師あり学習と同等
- **Open-Vocabulary理解**：固定クラス集合に縛られず、任意のテキスト記述に対して画像を評価できる

**SigLIP**（Zhai et al., 2023）：CLIP損失をシグモイドベースに変更
- 負例の影響を分離し、小バッチでの学習効率を改善
- OpenVLA、π0等のロボットモデルで採用

---

## LLaVAはビジョンエンコーダとLLMを線形投影で接続する最小設計のVLM

**LLaVA（Large Language and Vision Assistant）**（Liu et al., 2023）のアーキテクチャ：

1. CLIP ViTで画像を特徴ベクトル列に変換
2. 線形MLP（ビジョン-言語プロジェクタ）で次元を揃える
3. 視覚トークンとテキストトークンを連結してLLMに入力

$$\text{LLMへの入力} = [z_{v,1}, \ldots, z_{v,N_v}, x_1, \ldots, x_T]$$

2段階の学習：
- **Stage 1（Alignment）**：プロジェクタのみ学習（画像キャプション）
- **Stage 2（指示チューニング）**：全パラメータ学習（視覚質問応答）

---

## Flamingoはクロスアテンションで事前学習済みモデルを疎に結合する

**Flamingo**（Alayrac et al., 2022）の設計：事前学習済みLLMとビジョンエンコーダを最小限の新規パラメータで統合

- ビジョンエンコーダとLLMのパラメータを**固定**
- Perceiver Resamplerで視覚特徴を固定数のトークンに圧縮
- LLMの各Transformer層の間にクロスアテンション層を挿入（新規パラメータのみ学習）

現代VLMの主流：

| モデル | 構成 | 用途 |
|--------|------|------|
| PaliGemma（2024） | SigLIP ViT + Gemma | ロボット系（π0等） |
| InternVL2（2024） | InternViT + InternLM | 動的解像度対応、高解像度画像 |

---

## 点群は不規則・順序不変・疎であり、CNNの直接適用が困難

ロボットはLiDAR・深度カメラから3D点群 $\{p_i\}_{i=1}^{N} \subset \mathbb{R}^3$ を取得する。点群の性質：

- **不規則（irregular）**：グリッドではなくランダムな点の集合
- **順序不変（permutation invariant）**：点のインデックスに意味がない
- **疎（sparse）**：空間の大部分が空白

**PointNet**（Qi et al., 2017）は対称関数（max pooling）で順序不変性を保証する：

$$f(\{p_1, \ldots, p_N\}) = g\left(\max_{i=1}^{N}h(p_i)\right)$$

$h$：共有MLP、$\max$：次元ごとの最大値、$g$：タスクヘッド

---

## Point Transformerは局所近傍内の自己注意で点群の幾何構造を学習する

**PointNet++**（Qi et al., 2017b）：局所近傍点のグループ化と階層的特徴抽出により局所的な幾何学構造を学習

**Point Transformer**（Zhao et al., 2021）は局所近傍での自己注意で点群特徴を抽出：

$$y_i = \sum_{x_j \in \mathcal{N}(x_i)} \rho(W_\gamma(\varphi(x_i) - \psi(x_j)) + \delta) \odot (W_\alpha x_j + \delta)$$

$\delta = \theta(p_i - p_j)$：相対位置エンコーディング

大規模モデルとの統合アプローチ：
1. ボクセル化 → CNN → トークン化
2. FPS（最遠点サンプリング）+ PointNet++ → トークン化
3. 深度画像 → ViT（最も実装が簡単）

---

## DINOv2は教師なし事前学習でロボット知覚の汎用特徴を提供する

**DINO**（Caron et al., 2021）：Student-Teacher蒸留フレームワークで局所的なセマンティクスを学習
- 同一画像の異なるクロップを正例として学習
- 教師ネットワークはモメンタム更新（EMA）

**DINOv2**（Oquab et al., 2023）：LVD-142M（厳選された1億4千万枚）で事前学習した汎用ビジョンエンコーダ

ロボット知覚への応用まとめ：

| モデル | 用途 |
|-------|------|
| CLIP / SigLIP | 言語条件付き物体認識、ゴール条件付け |
| DINOv2 | 汎用的なロボット視覚特徴、深度推定 |
| PointNet++ | LiDAR点群での障害物検知、把持計画 |

---

## まとめ：視覚・言語・3D点群の統合技術

| 概念 | 内容 |
|------|------|
| CLIP | 4億ペアの対照学習；テキスト・画像を共通潜在空間に統合 |
| VLM（LLaVA） | ビジョンエンコーダ + プロジェクタ + LLM；視覚言語タスクを統合 |
| クロスアテンション | 異なるモダリティ間の動的な情報統合機構（Flamingo） |
| PointNet / Point Transformer | 不規則な3D点群の順序不変な特徴抽出 |
| DINOv2 | 汎用ビジョン特徴；ロボット知覚の基盤 |

次章では、これらの視覚・言語表現をロボット制御に直接つなぐ**模倣学習と行動表現**を詳述する。
