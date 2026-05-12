# 第9章　大規模言語モデルの原理と整合技術

## 概要

**大規模言語モデル（Large Language Model; LLM）**は，大規模テキストコーパスに対する自己回帰的な次トークン予測という単純な目的関数で事前学習された Transformer モデルである．スケールの拡大に伴い，指示に従う能力・論理的推論・コード生成など，事前学習目標には含まれない能力が**創発（emergence）**することが観察され，現代 AI の中心的な存在となっている．本章では，LLM の事前学習の原理，スケーリング則，インコンテキスト学習の機構，そして人間の価値観への整合（RLHF・Direct Alignment）を体系的に論じる．

---

## 9.1　自己回帰言語モデルの原理

### 9.1.1　言語モデルとしての定式化

言語モデルは，トークン列 $x = (x_1, x_2, \ldots, x_T)$ に確率を割り当てる：

$$p(x) = p(x_1) \prod_{t=2}^{T} p(x_t \mid x_1, \ldots, x_{t-1}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

連鎖律による因数分解は厳密であり，自己回帰モデルは完全に一般的な結合分布をモデル化できる．

### 9.1.2　Transformer デコーダによる実装

GPT 系の LLM はデコーダのみの Transformer で実装される（第6章）：

- 入力トークン列に**因果的マスク（Causal Mask）**を適用：時刻 $t$ のトークンは時刻 $< t$ のトークンにのみ注意できる
- 各位置のトークンに対して次トークンの確率分布を出力
- 語彙サイズ $V$ の線形ヘッドとソフトマックスで確率分布を生成

事前学習の目的関数（**負の対数尤度の最小化**）：

$$\mathcal{L}_{\mathrm{LM}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)})$$

この目的関数は：文法・事実知識・論理関係・コード・数式など，ウェブテキストに含まれるあらゆる構造を統一的に学習する枠組みとなっている．

### 9.1.3　トークン化（Tokenization）

テキストを整数インデックスのシンボル列に変換する前処理．代表的な手法：

- **BPE（Byte Pair Encoding）**（Sennrich et al., 2016）：最も頻出するバイト対を反復的にマージして語彙を構築する．GPT-2，Llama 等で使用
- **SentencePiece**（Kudo & Richardson, 2018）：言語非依存でバイト・文字・サブワードを統合的に扱う（T5，Llama 3 等）
- **語彙サイズ**：通常 $3 \times 10^4 \sim 1 \times 10^5$ トークン

日本語などマルチバイト言語は，バイトレベルトークン化のため英語より多くのトークンに分割される傾向がある．

---

## 9.2　スケーリング則（Scaling Laws）

### 9.2.1　Chinchilla スケーリング則

Kaplan et al.（2020）が LLM の損失とモデルサイズ $N$・訓練トークン数 $D$・計算量 $C$ の関係を定式化した：

$$L(N, D) \approx A N^{-\alpha} + B D^{-\beta} + L_\infty$$

$L_\infty$ は不可逆な損失（データの固有エントロピー）．計算量が固定された場合の最適配分：初期の研究では固定 $C = 6ND$ のもとでモデルサイズを大きくする方向が示唆されたが，Hoffmann et al.（2022）**「Chinchilla」**はデータとモデルサイズを均等に増やすべきことを示した：

$$N_{\mathrm{opt}} \propto C^{0.5}, \quad D_{\mathrm{opt}} \propto C^{0.5}$$

**実用的含意：** Chinchilla 則によれば，同じ計算量なら「大きなモデルを少ない訓練データで」より「小さなモデルを多くのデータで」訓練した方が低い損失を達成できる．GPT-4 以降の多くのモデルは Chinchilla より訓練過剰（over-training）だが，推論コストを下げるためにモデルを小さくしたまま多くデータを使うトレードオフが実用的に重視される．

### 9.2.2　スケーリングと創発能力

モデルサイズ・データ量・計算量をスケールアップすると，ある閾値を超えたときに急激に能力が向上する**創発（emergent abilities）**現象が観察された（Wei et al., 2022）：

- 少数ショット学習（few-shot learning）
- 連鎖推論（Chain-of-Thought reasoning）
- 算術計算
- コード生成

ただし，創発が「真に非連続的な相転移」か「評価指標の非線形性」かは議論が続いている（Schaeffer et al., 2023）．

---

## 9.3　インコンテキスト学習（In-Context Learning）

### 9.3.1　ICL の現象

**インコンテキスト学習（ICL）**は，LLM のパラメータを更新せずに，プロンプト内に少数の例示（Few-Shot Examples）を提示するだけでタスクに適応する能力である：

```
Q: 「リンゴは何色？」 A: 「赤」
Q: 「バナナは何色？」 A: 「黄」
Q: 「空は何色？」 A: [LLM が予測]
```

パラメータ更新なしでこの能力が生じる理由の理論的説明として，「コンテキスト内で勾配降下法を暗黙的に実行している」という仮説（Akyürek et al., 2023）や，事前学習中に類似タスクが存在するためという説明がある．

### 9.3.2　Chain-of-Thought（CoT）プロンプティング

Wei et al.（2022）は，例示に「中間推論ステップ」を含めることで複雑な推論が飛躍的に改善することを示した：

```
通常: Q: 「15 × 17 = ?」A: 「255」
CoT: Q: 「15 × 17 = ?」
     A: 「15 × 17 = 15 × (10 + 7) = 150 + 105 = 255」
```

**Zero-shot CoT**（Kojima et al., 2022）：「ステップバイステップで考えてください」と指示するだけで同様の効果が得られる．

CoT の効果は大規模モデルでのみ顕著であり，スケールと推論能力の関係を示している．

---

## 9.4　事後学習：指示チューニングと RLHF

事前学習済み LLM は次トークン予測の「続き」を生成するモデルに過ぎず，人間の指示に従う・有害出力を避ける・誠実に応答するという性質（**整合 / Alignment**）は自動的には得られない．

### 9.4.1　教師あり指示チューニング（SFT: Supervised Fine-Tuning）

人間が作成した（プロンプト，望ましい応答）ペアデータセットで LLM をファインチューニングする：

$$\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)$$

（$x$：プロンプト，$y$：応答）

SFT は LLM に「指示に答えるフォーマット」を教えるが，品質の高い指示データが必要で，高コストな人手収集が必要となる．

### 9.4.2　RLHF（Reinforcement Learning from Human Feedback）

**RLHF**（Christiano et al., 2017；Ouyang et al., 2022）は，複数の応答に対する人間の比較フィードバックから報酬モデルを学習し，PPO で LLM を最適化する3段階のパイプラインである：

**Step 1: SFT**（上述）

**Step 2: 報酬モデル（Reward Model; RM）の学習：**

同一プロンプトに対して複数の応答 $y_1$，$y_2$ を生成し，人間評価者が好む応答 $y_w \succ y_l$ を収集する．ブラッドリー・テリーモデルを仮定して報酬モデルを学習：

$$\mathcal{L}_{\mathrm{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

$r_\phi$：報酬モデル（LLM をベースに最終層を線形層に置き換えたスカラー出力モデル）．

**Step 3: PPO による強化学習：**

$$J_{\mathrm{RL}}(\theta) = \mathbb{E}_{(x, y) \sim \pi_\theta}\left[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\right] \quad \text{（最大化）}$$

$\pi_{\mathrm{ref}}$：SFT モデル（参照方策），$\beta$：KL 正則化の強度（$D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$ を KL ペナルティとして追加）．KL ペナルティ項は LLM が元の能力を失う**アライメント税（alignment tax）**を防ぐ安全バルブとして機能する．実際の PPO 更新ではこの目標に対して確率比クリッピング（第8章）を組み合わせる．

RLHF は ChatGPT，Claude，Llama-2 などの整合済みモデルの基盤となった．

### 9.4.3　DPO（Direct Preference Optimization）

**DPO**（Rafailov et al., 2023）は，RLHF の PPO ステップを不要にし，比較データから直接 LLM を最適化する閉形式の手法である：

$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right]$$

この式は RLHF の最適解が暗黙的に定義されていることを利用した代入から導出される．PPO より実装が単純で不安定性がなく，現代のモデル学習で広く採用されている．

### 9.4.4　RLVR（Reinforcement Learning with Verifiable Rewards）

数学・コード・論理問題のような**検証可能な正解**がある場合，人間のフィードバックなしに，正誤の検証結果を報酬とする強化学習が有効である（DeepSeek-R1，Qwen などで採用）：

$$r(y, x) = \begin{cases} +1 & \text{if } y \text{ が正解} \\ -1 & \text{otherwise} \end{cases}$$

**GRPO（Group Relative Policy Optimization）**などの変形が長鎖推論（Chain-of-Thought reasoning）能力の強化に成功しており，「Thinking models」（OpenAI o1，DeepSeek-R1 等）の基盤となっている．

---

## 9.5　推論技術

### 9.5.1　KV キャッシュ（KV Cache）

Transformer の自己回帰生成では，時刻 $t$ のトークンを生成する際に時刻 $1, \ldots, t-1$ のすべての Key・Value を再計算する必要がある．**KV キャッシュ**は，生成済みトークンの Key・Value をメモリに蓄積しておき，再計算を避ける最適化技術である：

- メモリ使用量：$O(T \times d_{\mathrm{model}} \times L)$（$T$：系列長，$L$：層数）；長文で大量のメモリが必要
- 計算量：KV キャッシュなしの $O(T^2)$ から，新規トークンのみ $O(T)$ に削減

**Multi-Query Attention（MQA）**（Shazeer, 2019）や **Grouped-Query Attention（GQA）**（Ainslie et al., 2023）は，Key・Value ヘッド数を削減して KV キャッシュのメモリ使用量を大幅に削減する手法であり，Llama 2/3，Mistral 等の実用モデルで採用されている．

### 9.5.2　サンプリング戦略

LLM の生成（推論）では，次トークンの確率分布からのサンプリング方法が出力の質に大きく影響する：

- **Greedy Decoding：** 毎回最確率トークンを選択．決定論的だが多様性が低い
- **Temperature Sampling：** $p_t(v) \propto \exp(\log p(v) / T)$，$T > 1$ で多様性増加，$T < 1$ で確率が集中
- **Top-k Sampling：** 上位 $k$ トークンのみからサンプリング
- **Nucleus Sampling（Top-p）：** 累積確率が $p$ に達するまでのトークンからサンプリング；適応的なカットオフ

### 9.5.2　長文コンテキストへの対応

LLM のコンテキスト長を 4K→128K→1M トークンに拡張する技術が進展：

- **RoPE の外挿：** YaRN，LongRoPE などで訓練長を超えた長さへの外挿を改善
- **スライディングウィンドウ注意：** Longformer，Mistral の SWA で計算量を $O(T)$ に削減
- **Position Interpolation：** 訓練時の位置エンコーディングをスケールして延伸

---

## まとめ

| 概念 | 内容 |
|------|------|
| 自己回帰 LM | 因果的 Transformer による次トークン予測 |
| Chinchilla 則 | 計算量最適配分：モデルサイズとデータを均等にスケール |
| 創発能力 | 十分な規模で CoT・算術など予期しない能力が発現 |
| ICL / CoT | パラメータ更新なしのプロンプトによるタスク適応 |
| SFT + RLHF | 指示チューニング→比較フィードバック→PPO による整合 |
| DPO | RLHF の最適化を単純な分類損失に変換 |

次章では，「生成」の観点からデータ分布を直接モデル化する**拡散モデル**の数理を詳述する．
