---
marp: true
theme: default
paginate: true
math: katex
---

# 第9章　大規模言語モデルの原理と整合技術

**次トークン予測という単純な目的関数で，スケールに伴い予期しない能力が創発する**

---

## 言語モデルは連鎖律でテキストの結合確率を自己回帰的に分解する

テキストトークン列 $(x_1, \ldots, x_T)$ の確率：

$$p(x) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

- この分解は**厳密**（連鎖律）であり，近似ではない
- 事前学習目的関数（負の対数尤度最小化）：

$$\mathcal{L}_{\mathrm{LM}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}\log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)})$$

---

## 因果的マスクにより，デコーダ Transformer が自己回帰生成を実現する

- GPT 系 LLM：**デコーダのみの Transformer**（第6章）
- **因果的マスク（Causal Mask）：** 位置 $t$ のトークンは位置 $< t$ のみ参照可能
  → 並列学習しながら自己回帰の条件を保持
- トークン化（BPE）：最頻出バイト対を反復マージ → 語彙サイズ $3\times10^4 \sim 10^5$

---

## Chinchilla スケーリング則は，計算量最適配分のためにモデルとデータを均等にスケールすることを示した

Hoffmann et al.（2022）：固定計算量 $C$ のもとでの最適配分：

$$N_{\mathrm{opt}} \propto C^{0.5}, \quad D_{\mathrm{opt}} \propto C^{0.5}$$

- 計算量を増やすなら，**モデルサイズとデータ量を均等に増やすべき**
- 実用上：推論コスト削減のため，小さなモデルを大量データで学習する傾向（over-training）
- スケールとともに損失が予測可能に低下するため，事前に性能予測が可能

---

## スケールを超えた閾値で，事前学習目標に含まれない能力が「創発」する

- 十分な規模のモデルで観察される**創発（emergent abilities）**（Wei et al., 2022）：
  - 少数ショット学習（few-shot），連鎖推論（Chain-of-Thought）
  - 算術計算，コード生成，言語横断的推論

- 注意：創発の「非連続性」は評価指標の非線形性による見かけの効果という議論もある（Schaeffer et al., 2023）

---

## インコンテキスト学習はパラメータを更新せず，プロンプトだけでタスクに適応する

**Few-Shot ICL：** プロンプト内に入出力の例示を与えるだけで新しいタスクを解く

- 解釈仮説：「コンテキスト内で暗黙的な勾配降下を実行している」
- **Chain-of-Thought（CoT）：** 中間推論ステップを例示に含めることで複雑推論が改善
- **Zero-shot CoT：**「ステップバイステップで考えてください」という指示だけで有効

---

## SFT + RLHF は「次トークン予測モデル」を「人間の意図に従うモデル」に変換する

事後学習の3段階：

1. **SFT（Supervised Fine-Tuning）：** （プロンプト，応答）ペアで指示フォーマットを学習
2. **報酬モデル学習：** ブラッドリー・テリーモデルによる人間の好み比較データから学習：
   $\mathcal{L}_{\mathrm{RM}} = -\mathbb{E}\!\left[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\right]$
3. **PPO による強化学習（RLHF）：** $J_{\mathrm{RL}}(\theta) = \mathbb{E}\!\left[r_\phi(x,y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]$ を最大化

---

## DPO は RLHF の最適解を閉形式で直接求め，強化学習を不要にする

RLHF の最適方策が満たす条件を代入して導出：

$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]$$

- PPO 不要で実装が単純；現代モデル学習で広く採用
- **RLVR（検証可能報酬による RL）：** 数学・コード等の正誤を報酬に → DeepSeek-R1 等の「Thinking モデル」の基盤

---

## KV キャッシュは自己回帰生成の計算量を $O(T^2)$ から $O(T)$ に削減する

- 生成済みトークンの Key・Value をメモリに蓄積 → 再計算を排除
- メモリコスト：$O(T \times d_{\mathrm{model}} \times L)$（長文で大量消費）
- **GQA（Grouped-Query Attention）：** Key・Value ヘッド数を削減して KV キャッシュメモリを大幅削減 → Llama 2/3，Mistral 等で採用

---

## まとめ：LLM の原理と整合技術

| 概念 | 内容 |
|------|------|
| 自己回帰 LM | 因果的 Transformer による次トークン予測 |
| Chinchilla 則 | モデルサイズとデータを均等スケールが最適 |
| 創発能力 | 閾値を超えると予期しない能力が発現 |
| ICL / CoT | パラメータ更新なしのプロンプトによる適応 |
| RLHF | SFT + 報酬モデル + PPO の3段階整合 |
| DPO / RLVR | RL フリーの整合；検証可能報酬による推論強化 |

次章では，データへのノイズ付加とその逆転を学習する拡散モデルの数理を詳述する
