---
marp: true
theme: default
paginate: true
math: katex
---

# 第6章　系列モデルと自己注意機構
## RNN・LSTM・Transformer

**核心メッセージ：** 系列データの依存関係を効率よくモデル化するため，RNNの限界をLSTMが緩和し，Transformerが根本的に解決した

---

## RNNは「隠れ状態」に系列全体を圧縮する——それが長距離依存の限界を生む

- **再帰型ニューラルネットワーク（RNN）** は，時刻 $t$ の入力 $x_t$ と前時刻の隠れ状態 $h_{t-1}$ から新たな隠れ状態 $h_t$ を計算する

$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$$

- パラメータ $W_h$，$W_x$ は**全時刻で共有**される——系列長に依存しない固定パラメータ数
- 隠れ状態 $h_t$ が過去情報の「記憶」として機能するが，固定次元への圧縮には情報のボトルネックが存在する

---

## BPTTにおける勾配の積が，勾配消失・爆発を引き起こす

- RNNの学習は時間方向に計算グラフを展開した **BPTT（Backpropagation Through Time）** で行われる
- 時刻 $t$ への勾配は以下の積で表される：

$$\frac{\partial \mathcal{L}}{\partial h_t} = \left(\prod_{k=t}^{T-1} W_h^\top \mathrm{diag}(\sigma'(z_{k+1}))\right) \frac{\partial \mathcal{L}}{\partial h_T}$$

- 積の項数は系列長 $T - t$ に比例する
  - $\|W_h\|$ の固有値 $< 1$：**勾配消失**（遠い過去の情報を学習できない）
  - 固有値 $> 1$：**勾配爆発**（学習が発散する）
- 結果として，RNNは実用的な長距離依存（long-range dependency）の学習が困難

---

## LSTMはゲート機構でセル状態を保護し，勾配消失を緩和する

- **LSTM**（Hochreiter & Schmidhuber, 1997）はセル状態 $C_t$（長期記憶）を導入し，3つのゲートで制御する

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{（忘却ゲート）}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{（セル状態更新）}$$

- 更新が**加算**で行われるため，$C_T$ から $C_t$ への勾配がほぼそのまま流れる（ResNetの残差接続と同様の原理）
- **GRU**（Cho et al., 2014）は2つのゲート（リセット・更新）に簡略化したもので，LSTMの約3/4のパラメータ数で同等の性能を達成

---

## LSTMも「逐次計算」と「情報ボトルネック」という本質的限界を持つ

- **逐次計算の制約：** $h_t$ が $h_{t-1}$ に依存するため，時刻方向の並列化が**原理的に不可能**
  - 長い系列の学習に大きな計算時間が必要
- **情報ボトルネック：** 固定次元の $h_t$ にすべての過去情報を圧縮するため，情報が失われる
- **スケーリングの非効率：** GPU の並列性を十分に活用できず，大規模データへの対応が困難

これらの限界を根本的に解決したのが Transformer である

---

## Transformerは注意機構のみで，任意の2トークン間の依存を直接計算する

- Vaswani et al.（2017）「Attention is All You Need」が提案した **Transformer** はRNNを完全に排除した
- 自己注意（Self-Attention）により，任意の2トークン間の依存が**直接**計算される
  - RNNの経路長：$O(T)$（時刻を1つずつ遡る必要がある）
  - Transformerの経路長：$O(1)$（全トークン対を同時に参照）
- **完全な並列化：** 系列全体を同時に処理できるため，GPUの並列性を最大限に活用

---

## スケーリングドット積注意は，クエリとキーの類似度で値を重み付けする

入力から Query $Q$，Key $K$，Value $V$ を線形変換で生成し，以下で注意を計算する：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- $QK^\top \in \mathbb{R}^{T \times T}$：全トークン対間の類似度行列
- $1/\sqrt{d_k}$ スケーリング：次元数 $d_k$ が大きいと内積の分散が増大しソフトマックス後の勾配が消失するため，標準偏差 $\sqrt{d_k}$ で正規化する
- 計算量は $O(T^2 d_k)$（注意行列のサイズが支配的）

---

## マルチヘッド注意は，複数の「注目パターン」を並列に学習する

単一の注意機構では一種類の注目パターンしか表現できない：

$$\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$

- $h$ 個の独立したヘッドが各々異なる「サブスペース」での注目パターンを担当
- 局所的依存・長距離依存・構文・意味など多様な関係を同時に捉えられる
- 各ヘッドの次元 $d_k = d_{\mathrm{model}}/h$ とすることで計算量を維持

---

## 自己注意は位置情報を持たないため，位置エンコーディングが必要

- 自己注意は**順序同変（permutation equivariant）**：トークンの順序を入れ替えると出力も同じ順で入れ替わるだけで，絶対位置を区別できない
- 元論文では三角関数による固定エンコーディングを入力に加算：

$$\mathrm{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{\mathrm{model}}}}\right), \quad \mathrm{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{\mathrm{model}}}}\right)$$

- 現代のモデルでは **RoPE（Rotary Position Embedding）**（Su et al., 2021）が主流：相対位置 $m-n$ のみに依存するため，学習長を超えた系列への外挿性が向上する

---

## TransformerブロックはMSA・FFN・残差・LayerNormの4要素で構成される

```
入力 X
  ↓
[Multi-Head Self-Attention]
  ↓ （+ 残差接続）
[Layer Normalization]
  ↓
[Feed-Forward Network: 2層MLP with GELU]
  ↓ （+ 残差接続）
[Layer Normalization]
  ↓
出力
```

- FFN：$\mathrm{FFN}(x) = W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2$，通常 $d_{\mathrm{FFN}} = 4 d_{\mathrm{model}}$
- **Pre-LN**（注意の前にLayerNorm）が大規模モデルでの学習安定性の観点から現代の標準
- 用途によりエンコーダのみ（BERT系），デコーダのみ（GPT系），エンコーダ・デコーダ（T5系）の3形態がある

---

## ViTは画像をパッチトークンに分割し，Transformerで処理する

- Dosovitskiy et al.（2020）「An Image is Worth 16x16 Words」の **ViT**
  1. 画像 $\mathbb{R}^{H \times W \times C}$ を $N = HW/P^2$ 個の $P \times P$ パッチに分割
  2. 各パッチを平坦化して線形投影し，トークンとして扱う
  3. 学習済み位置埋め込みと分類用 [CLS] トークンを付加
  4. 標準 Transformer エンコーダで処理
- 大規模データ（ImageNet-21k，JFT-300M）ではCNNを凌駕するが，小規模データでは局所性・平行移動不変性という帰納バイアスの欠如が不利に働く
- **DinoV2**，**SigLIP** など，ロボティクス向けVLAのビジョンバックボーンとして標準的に採用されている

---

## まとめ：系列モデルの発展と各手法の特性

| 手法 | 並列化 | 長距離依存 | 位置情報 | 主な用途 |
|------|--------|-----------|---------|---------|
| RNN | 不可 | 弱い（勾配消失） | 暗黙的 | 系列処理（古典的） |
| LSTM/GRU | 不可 | 改善（ゲート機構） | 暗黙的 | 系列処理 |
| Transformer | 完全並列 | 強い（$O(1)$経路長） | 明示的注入が必要 | LLM，VLA，ViT |
| ViT | 完全並列 | 強い | 学習済み埋め込み | 視覚表現 |

- Self-Attention の計算量 $O(T^2)$ は長系列でボトルネックになるが，**FlashAttention** により $O(T)$ のメモリで高速処理が可能
- 次章では「どのような監督信号でモデルを学習するか」という**学習パラダイム**を体系化する
