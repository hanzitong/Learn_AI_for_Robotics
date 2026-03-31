---
marp: true
theme: default
paginate: true
math: katex
---

# 第8章　強化学習の理論と方策最適化

**エージェントは報酬信号だけを頼りに，試行錯誤を通じて最適な方策を学習する**

---

## 強化学習は「正解行動を列挙できないタスク」のための学習パラダイムである

- 教師あり学習：正解ラベル $y$ がある → 損失関数で直接学習
- 強化学習：正解行動は未知だが，行動の「良さ」を示す**報酬信号 $r$** だけが得られる
- 適用領域：ロボット操作・自律移動・物理シミュレーション学習
  → 報酬は設計できても，全状態の最適行動を列挙することは現実的でない

---

## MDP はエージェントと環境の相互作用を数学的に定式化する

MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$：

- $\mathcal{S}$：状態空間，$\mathcal{A}$：行動空間
- $P(s' \mid s, a)$：遷移確率（マルコフ性：次状態は $(s_t, a_t)$ のみに依存）
- $r(s, a, s')$：報酬関数，$\gamma \in [0,1)$：割引率

**目標：** 割引累積報酬の期待値 $J(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r_{t+1}\right]$ を最大化する方策 $\pi$ を求める

---

## ベルマン方程式は価値関数の再帰的な関係を記述する

**状態価値関数：** $V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t = s\right]$

**ベルマン期待方程式：**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[r(s,a,s') + \gamma V^\pi(s')\right]$$

**ベルマン最適方程式（最適 Q 関数）：**

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[r + \gamma \max_{a'} Q^*(s',a')\right]$$

---

## Q学習は遷移経験から最適行動価値関数を反復的に推定する

TD（Temporal Difference）目標との差（**TD誤差**）を縮小する方向に更新：

$$Q(s,a) \leftarrow Q(s,a) + \alpha\underbrace{\left[r + \gamma\max_{a'} Q(s',a') - Q(s,a)\right]}_{\text{TD誤差}}$$

- **オフポリシー：** 収集方策（$\varepsilon$-greedy 等）と学習対象方策が異なってもよい
- **DQN（Mnih et al., 2015）：** Q関数をニューラルネットで近似；Atari で人間レベルを達成
  - **経験リプレイ：** 連続遷移の相関を破る
  - **ターゲットネットワーク：** TD目標の安定化

---

## 方策勾配定理は，累積報酬を直接最大化する勾配の計算式を与える

確率的方策 $\pi_\theta$ の目標 $J(\theta)$ の勾配（**方策勾配定理，Sutton et al., 1999**）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log\pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]$$

- $\nabla_\theta \log\pi_\theta(a|s)$：**スコア関数** — 行動 $a$ の確率を上げる方向
- $Q^{\pi_\theta}(s,a)$ が大きいほど，その行動の確率を強く上げる

---

## アドバンテージ関数による分散低減が，実用的な方策勾配法を可能にする

アドバンテージ関数：$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$（平均からの相対的な優位性）

**REINFORCE with baseline：**

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i,t} \nabla_\theta\log\pi_\theta(a_t^i|s_t^i) \cdot (G_t^i - b(s_t^i))$$

**GAE（Generalized Advantage Estimation，Schulman et al., 2016）：** $\lambda$ でバイアス・分散トレードオフを調整

$$\hat{A}_t^{\mathrm{GAE}} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

---

## PPO は確率比のクリッピングで大きな方策更新を防ぎ，安定した学習を実現する

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

$r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\mathrm{old}}(a_t|s_t)$：確率比，$\varepsilon = 0.2$ が標準

- 実装が簡単で安定；RLHF（第9章），ロボット操作で広く使われる
- **SAC（Haarnoja et al., 2018）：** エントロピー最大化オフポリシー法；連続制御の標準

---

## モデルベース強化学習は，環境の遷移モデルを学習して想像上で訓練する

| | モデルフリー | モデルベース |
|--|------------|------------|
| 遷移モデル | 不要 | 学習または既知 |
| データ効率 | 低い | 高い |
| 漸近性能 | 高い | モデル誤差に依存 |
| 代表例 | DQN，PPO，SAC | Dreamer，MuZero |

**Dreamer：** 潜在空間の遷移モデルで Actor-Critic を訓練 → ロボットビジョン制御に応用

---

## まとめ：強化学習の主要な設計選択

| 概念 | 内容 |
|------|------|
| MDP | 状態・行動・遷移・報酬・割引率による定式化 |
| ベルマン方程式 | 価値関数の再帰的定義；最適方策は Q 関数から導出 |
| DQN | 経験リプレイ + ターゲットネットワークで安定化 |
| PPO | 確率比クリッピングによる安定した方策更新 |
| GAE | バイアス・分散トレードオフの連続的制御 |
| SAC | エントロピー最大化；データ効率の高い連続制御 |

次章では，強化学習（RLHF）を組み込んだ大規模言語モデルの原理と整合技術を詳述する
