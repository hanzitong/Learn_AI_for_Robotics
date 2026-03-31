# 第6章　系列モデルと自己注意機構：RNN・LSTM・Transformer

## 概要

CNN は空間的局所性を帰納バイアスとするが，テキスト・音声・時系列センサデータ・ロボットの行動系列といった**順序が意味を持つデータ**を扱うには，系列の依存関係をモデル化する機構が必要である．本章では，系列モデルの発展を辿りながら，RNN・LSTM の構造と限界，そしてそれを超えた**Transformer** のアーキテクチャと**自己注意機構（Self-Attention）**を詳述する．Transformer は LLM（第9章），VLA（第13章），ロボット基盤モデル（第14章）のすべての基盤となる現代最重要のアーキテクチャである．

---

## 6.1　再帰型ニューラルネットワーク（RNN）

### 6.1.1　RNN の基本構造

**再帰型ニューラルネットワーク（Recurrent Neural Network; RNN）**は，時刻 $t$ の入力 $x_t$ と前時刻の**隠れ状態（hidden state）** $h_{t-1}$ から，新たな隠れ状態 $h_t$ を計算する：

$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$$

$$\hat{y}_t = W_y h_t + b_y$$

$W_h$，$W_x$，$W_y$ は全時刻で**共有**されるパラメータであり，系列の長さに関わらず一定のパラメータ数で系列を処理できる．隠れ状態 $h_t$ が系列のこれまでの情報を圧縮した「記憶」として機能する．

### 6.1.2　BPTT（Backpropagation Through Time）

RNN の学習は，計算グラフを時間方向に展開して誤差逆伝播を行う **BPTT（Backpropagation Through Time）**によって行われる．系列長 $T$ に対して，勾配は時刻 $T$ から $1$ へ逆向きに伝播する：

$$\frac{\partial \mathcal{L}}{\partial h_t} = \left(\prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}\right)^\top \frac{\partial \mathcal{L}}{\partial h_T} = \left(\prod_{k=t}^{T-1} W_h^\top \mathrm{diag}(\sigma'(z_{k+1}))\right) \frac{\partial \mathcal{L}}{\partial h_T}$$

（$\frac{\partial h_{k+1}}{\partial h_k} = \mathrm{diag}(\sigma'(z_{k+1})) W_h$ より，転置をとって $W_h^\top \mathrm{diag}(\sigma'(z_{k+1}))$）

この積の項数が $T - t$ と系列長に比例するため，$T$ が大きいと勾配消失（$\|W_h\|$ の固有値 $< 1$）または勾配爆発（固有値 $> 1$）が生じる．

### 6.1.3　長距離依存の問題

RNN は原理上すべての過去の情報を $h_t$ に圧縮するが，実際には遠く離れた時刻の情報は $h_t$ からほぼ失われ，**長距離依存（long-range dependency）**を学習できない．例えば，「The cat that ate the mouse [... 100 tokens ...] **was**」のような長い依存関係を捉えることが困難である．

---

## 6.2　LSTM（Long Short-Term Memory）

### 6.2.1　LSTM の動機

**LSTM**（Hochreiter & Schmidhuber, 1997）は，RNN の長距離依存問題を**ゲート機構**によって解決する．セル状態（cell state）$C_t$ という追加の「長期記憶」を持ち，何を忘れ・記憶し・出力するかをゲートによって制御する．

### 6.2.2　LSTM の数式

入力ゲート $i_t$，忘却ゲート $f_t$，出力ゲート $o_t$，新候補セル $\tilde{C}_t$：

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{（忘却ゲート：何を忘れるか）}$$

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{（入力ゲート：何を追記するか）}$$

$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{（新候補セル）}$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{（セル状態更新：加算により勾配が流れやすい）}$$

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{（出力ゲート）}$$

$$h_t = o_t \odot \tanh(C_t) \quad \text{（隠れ状態）}$$

**勾配消失の緩和：** セル状態 $C_t$ の更新が加算 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ によって行われるため，長い系列でも $C_T$ から $C_t$ への勾配がほぼそのまま流れる（ResNet の残差接続と同様のメカニズム）．

### 6.2.3　GRU（Gated Recurrent Unit）

**GRU**（Cho et al., 2014）は LSTM をリセットゲート $r_t$ と更新ゲート $z_t$ の2ゲートに簡略化したもので，セル状態と隠れ状態を統合している：

$$z_t = \sigma(W_z [h_{t-1}, x_t]), \quad r_t = \sigma(W_r [h_{t-1}, x_t])$$

$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

パラメータ数は LSTM の約 $3/4$ であり，多くのタスクで同等の性能を達成する．

### 6.2.4　LSTM/GRU の限界

LSTM は RNN の長距離依存問題を大幅に改善したが，以下の本質的な限界が残る：

1. **逐次計算：** $h_t$ が $h_{t-1}$ に依存するため，時刻方向の並列化が不可能．長い系列の学習に大きな計算時間が必要
2. **線形的な情報圧縮：** 固定次元の $h_t$ にすべての情報を圧縮する際の情報ボトルネック
3. **スケーリングの困難：** 大規模データと計算資源への対応が GPU 並列性の観点から不効率

---

## 6.3　注意機構（Attention Mechanism）

### 6.3.1　Bahdanau Attention

Bahdanau et al.（2015）はエンコーダ・デコーダの機械翻訳において，デコーダが翻訳の各ステップで**エンコーダの全ステップ**を動的に参照できる**注意機構**を提案した：

エンコーダが各時刻の隠れ状態 $\{h_i^{\mathrm{enc}}\}_{i=1}^{T}$ を出力するとき，デコーダの時刻 $t$ での**注意スコア**は，

$$e_{t,i} = a(s_{t-1}, h_i^{\mathrm{enc}})$$

（$s_{t-1}$：デコーダの前時刻の隠れ状態，$a$：整合性スコア関数）として計算され，ソフトマックスで正規化した注意重みにより**コンテキストベクトル**を生成する：

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}, \quad c_t = \sum_i \alpha_{t,i} h_i^{\mathrm{enc}}$$

この機構は，デコーダが入力系列の「どこに注目すべきか」を動的に決定できることを意味し，翻訳品質を大幅に向上させた．

---

## 6.4　Transformer

### 6.4.1　Transformer の動機

Vaswani et al.（2017）「Attention is All You Need」が提案した **Transformer** は，RNN を完全に排除し，注意機構のみで系列の依存関係をモデル化する．これにより，

- **完全な並列化：** 系列全体を同時に処理できる
- **$O(1)$ 経路長：** 任意の2トークン間の依存が直接の注意で計算される（RNN の $O(T)$ の経路長と対比）
- **スケーラビリティ：** 大規模データと大きな計算資源への効率的な対応

### 6.4.2　スケーリードット積注意（Scaled Dot-Product Attention）

Transformer の基本単位は**スケーリードット積注意**である．入力から Query $Q$，Key $K$，Value $V$ の三つの行列を線形変換で生成する：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

（$X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$：$T$ トークンの特徴行列，$W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$：投影行列）

**注意スコアの計算：**

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- $QK^\top \in \mathbb{R}^{T \times T}$：全トークン対間の類似度行列（注意スコア行列）
- $1/\sqrt{d_k}$ スケーリング：$d_k$ が大きいと内積の分散が大きくなり，ソフトマックス後の勾配が消失するため，標準偏差 $\sqrt{d_k}$ でスケールダウンする
- ソフトマックス後の重みで $V$ を重み付き和：各クエリが「どのキーに注目するか」を決定し，対応する値を取得

計算量は $O(T^2 d_k)$（$T \times T$ の注意行列が支配的）．

### 6.4.3　マルチヘッド注意（Multi-Head Attention）

一つの注意機構では単一の注目パターンしか表現できない．**マルチヘッド注意**は $h$ 個の独立した注意ヘッドを並列に計算し，結果を連結する：

$$\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$

各ヘッドが異なる「注意パターン（サブスペース）」を担当することで，局所的依存・長距離依存・構文・意味など多様な関係を同時に捉えられる．

パラメータ数：$h$ ヘッド，各ヘッドの次元 $d_k = d_{\mathrm{model}}/h$ として，$4 d_{\mathrm{model}}^2$（$Q$，$K$，$V$ 各 $W$ と出力 $W^O$）．

### 6.4.4　位置エンコーディング（Positional Encoding）

自己注意は**順序同変（permutation equivariant）**である：入力トークンの順序を並べ替えると，出力も同じ順序で並べ替わる（出力が入力順に依存するが，絶対的な位置情報は持たない）．これはモデルが「どのトークンが何番目か」を自動的に知ることができないことを意味し，位置情報を別途注入する必要がある．元論文では三角関数を用いた固定エンコーディングを使用：

$$\mathrm{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{\mathrm{model}}}}\right), \quad \mathrm{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{\mathrm{model}}}}\right)$$

現代のモデルでは，訓練可能な**学習済み位置エンコーディング**や，相対位置を扱う **RoPE（Rotary Position Embedding）**（Su et al., 2021；LLaMA 等で採用）が広く使われる：

$$\mathrm{RoPE}(q, k, m, n) = \mathrm{Re}\left[(q_m e^{im\theta}) \overline{(k_n e^{in\theta})}\right] = q_m^\top k_n \cos((m-n)\theta) + \ldots$$

RoPE は相対位置 $m - n$ のみに依存するため，学習長より長い系列への外挿性が向上する．

### 6.4.5　Transformer ブロックの全体構造

標準的な Transformer ブロックは以下の4つのサブレイヤで構成される：

```
Input X
  ↓
[Multi-Head Self-Attention]
  ↓  (+残差接続)
[Layer Normalization]
  ↓
[Feed-Forward Network (FFN): 2層MLP with GELU/ReLU]
  ↓  (+残差接続)
[Layer Normalization]
  ↓
Output
```

**FFN の構造：** $\mathrm{FFN}(x) = W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2$

通常 $d_{\mathrm{FFN}} = 4 d_{\mathrm{model}}$（GPT-3 など）．FFN は各トークンに独立に適用される（位置ごとに同一の重みを使用）．

**Pre-LN vs Post-LN：** 元論文は Post-LN（注意の後に LN）だが，学習の安定性から **Pre-LN**（注意の前に LN）が現代の大規模モデルでは標準．

### 6.4.6　クロスアテンション（Cross-Attention）

**クロスアテンション**は，Query を一つのモダリティ（例：デコーダの現在の状態）から，Key と Value を別のモダリティ（例：エンコーダの出力）から取るアテンションである：

$$\mathrm{CrossAttention}(X_q, X_{kv}) = \mathrm{softmax}\left(\frac{X_q W^Q (X_{kv} W^K)^\top}{\sqrt{d_k}}\right) X_{kv} W^V$$

これはエンコーダ・デコーダ Transformer のデコーダ層や，マルチモーダルモデルでの異なるモダリティ間の相互参照に用いられる．VLA モデル（第13章）では視覚特徴と言語特徴の融合にクロスアテンションが多用される．

### 6.4.7　エンコーダ・デコーダ・エンコーダのみ

用途によって Transformer の使い方が異なる：

- **エンコーダのみ（BERT 系）：** 双方向の自己注意（全トークンが全トークンを参照可能）．テキスト分類・固有表現認識など理解タスク向け
- **デコーダのみ（GPT 系）：** **因果的自己注意（Causal Attention）**（後続トークンをマスク）．テキスト生成・言語モデリング向け
- **エンコーダ・デコーダ（T5 系）：** エンコーダが入力を双方向にエンコードし，デコーダが因果的に生成．翻訳・要約向け

---

## 6.5　Vision Transformer（ViT）

### 6.5.1　ViT の設計

Dosovitskiy et al.（2020）「An Image is Worth 16x16 Words」が提案した **ViT** は，画像を固定サイズのパッチ（例：$16 \times 16$ ピクセル）に分割し，各パッチを「トークン」として Transformer に入力する：

1. 入力画像 $\mathbb{R}^{H \times W \times C}$ を $N = HW / P^2$ 個の $P \times P$ パッチに分割
2. 各パッチを平坦化して線形投影：$\mathbb{R}^{P^2 C} \to \mathbb{R}^{d_{\mathrm{model}}}$
3. 学習済み位置埋め込みを追加
4. 分類用 [CLS] トークンを先頭に付加
5. 標準 Transformer エンコーダで処理

ViT は大規模データセット（ImageNet-21k，JFT-300M）では CNN を凌駕するが，小規模データでは CNN の帰納バイアス（局所性・平行移動不変性）の欠如により性能が劣る．

### 6.5.2　ViT の応用

ViT はロボティクスでの視覚表現に広く採用されており，特に VLA（第13章）のビジョンバックボーンとして標準的である：

- **DinoV2**（Oquab et al., 2023）：自己教師あり事前学習による汎用的な ViT 特徴
- **SigLIP**（Zhai et al., 2023）：画像・テキストの対照学習による視覚・言語整合

---

## 6.6　効率的な注意機構

自己注意の計算量 $O(T^2)$ と GPU メモリ $O(T^2)$ は，長い系列（長文・長い動作系列）の処理を制限する：

- **FlashAttention**（Dao et al., 2022）：注意行列を明示的に保存せず，GPU の高バンド幅メモリ（SRAM）上でタイル処理することで IO を削減し，メモリ使用量 $O(T)$，高速化（10倍以上の場合も）
- **線形注意（Linear Attention）：** ソフトマックスを近似カーネル関数で置き換えて $O(T)$ にする（品質と効率のトレードオフ）
- **Sliding Window Attention（Longformer）：** ウィンドウ内のローカル注意 + グローバルトークンの組み合わせ

---

## まとめ

| 概念 | 内容 |
|------|------|
| RNN | 隠れ状態による系列処理；長距離依存に弱く，並列化不可 |
| LSTM/GRU | ゲート機構でセル状態を保護；長距離依存を大幅改善 |
| Self-Attention | 全トークン対間の依存を直接計算；$O(1)$ の経路長 |
| Transformer | Self-Attention + FFN + 残差 + LN；現代の大規模モデルの基盤 |
| ViT | 画像をパッチトークンとして処理；CNN を凌駕する視覚表現 |
| FlashAttention | IO 最適化により $O(T^2)$ メモリを $O(T)$ に削減 |

次章では，本章までで紹介したアーキテクチャを「どのような監督信号で学習するか」という観点から整理する**学習パラダイム**を論じる．
