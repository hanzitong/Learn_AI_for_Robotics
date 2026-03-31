# 第8章　強化学習の理論と方策最適化

## 概要

**強化学習（Reinforcement Learning; RL）**は，エージェントが環境と相互作用しながら試行錯誤を通じて最適な行動方策を学習するパラダイムである．教師あり学習が「正解ラベル」を必要とするのに対し，強化学習は**報酬信号**（ある行動が良かったか悪かったかのスカラー評価）のみを頼りに学習する．ロボット操作・自律移動・物理シミュレーション学習など，正解行動の列挙が困難なタスクに適している．本章では，MDP（マルコフ決定過程）による定式化，価値ベース法（Q学習・DQN），方策勾配法（REINFORCE・PPO），そして Actor-Critic アーキテクチャを順に論じる．

---

## 8.1　マルコフ決定過程（MDP）

### 8.1.1　MDP の定義

強化学習の問題は，**マルコフ決定過程（Markov Decision Process; MDP）**として定式化される．MDP は以下の組 $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ で定義される：

- $\mathcal{S}$：**状態空間（State Space）** — 環境の完全な状態の集合（エージェントが状態 $s$ に完全にアクセスできると仮定する；カメラ画像のみに基づく部分観測の場合は POMDP（Partially Observable MDP）に拡張される）
- $\mathcal{A}$：**行動空間（Action Space）** — エージェントが取りうる行動の集合
- $P(s' \mid s, a)$：**遷移確率（Transition Probability）** — 状態 $s$ で行動 $a$ を取ったとき，次の状態が $s'$ になる確率
- $r(s, a, s')$：**報酬関数（Reward Function）** — 遷移 $(s, a, s')$ に対するスカラー報酬
- $\gamma \in [0, 1)$：**割引率（Discount Factor）** — 将来の報酬を現在価値に割り引く係数

**マルコフ性：** 次の状態 $s_{t+1}$ は現在の状態と行動 $(s_t, a_t)$ のみに依存し，それ以前の履歴には依存しない：
$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

### 8.1.2　方策と累積報酬

**方策（Policy）** $\pi(a \mid s)$ は，状態 $s$ において行動 $a$ を選択する確率分布である（確定的方策 $a = \pi(s)$ も含む）．

エージェントの目標は，**割引累積報酬（Return）**の期待値を最大化する方策を見つけることである：

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$

$$J(\pi) = \mathbb{E}_\pi[G_0] = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1}\right]$$

$\gamma$ が $1$ に近いほど将来の報酬を重視し，$0$ に近いほど直近の報酬を重視する．

---

## 8.2　価値関数とベルマン方程式

### 8.2.1　価値関数の定義

方策 $\pi$ のもとでの**状態価値関数（State-Value Function）** $V^\pi(s)$ と**行動価値関数（Action-Value Function / Q-Function）** $Q^\pi(s, a)$ を定義する：

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]$$

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s, a_t = a\right]$$

$V^\pi(s)$ は状態 $s$ からの将来の期待累積報酬，$Q^\pi(s, a)$ は状態 $s$ で行動 $a$ を取った後の期待累積報酬を表す．両者の関係：

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)$$

### 8.2.2　ベルマン期待方程式

**ベルマン期待方程式（Bellman Expectation Equation）**は，価値関数の再帰的な関係を表す：

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[r(s, a, s') + \gamma V^\pi(s')\right]$$

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]$$

これらは $|\mathcal{S}|$ 本の連立一次方程式として解くことができる（モデルが既知の場合）．

### 8.2.3　ベルマン最適方程式

最適方策 $\pi^*$ のもとでの最適価値関数 $V^*(s) = \max_\pi V^\pi(s)$ は，**ベルマン最適方程式**を満たす：

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[r(s, a, s') + \gamma V^*(s')\right]$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[r(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right]$$

最適方策は $\pi^*(s) = \arg\max_a Q^*(s, a)$ として Q 関数から導出できる．

### 8.2.4　アドバンテージ関数

**アドバンテージ関数（Advantage Function）**は，ある行動が平均的な行動より「どれだけ良いか」を表す：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

$A^\pi(s, a) > 0$ は行動 $a$ が平均より良いことを，$< 0$ は平均より悪いことを意味する．方策勾配法の分散低減に重要な役割を果たす（8.4節）．

---

## 8.3　価値ベース法

### 8.3.1　Q 学習（Q-Learning）

**Q 学習**（Watkins & Dayan, 1992）は，最適行動価値関数 $Q^*$ をモデルを使わず（**モデルフリー**）に直接推定する手法である．TD（Temporal Difference）ターゲット：

$$\text{TD target} = r + \gamma \max_{a'} Q(s', a')$$

パラメータ更新：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

$\alpha$：学習率，括弧内の量を **TD 誤差（Temporal Difference Error）** と呼ぶ．Q 学習は**オフポリシー（off-policy）**手法：データ収集方策（探索方策，例：$\varepsilon$-greedy）が現在の方策と異なってもよい．

### 8.3.2　DQN（Deep Q-Network）

Mnih et al.（2015）が提案した **DQN** は，Q 関数をニューラルネットワーク $Q_\phi(s, a)$ でパラメータ化し，Atari ゲームで人間レベルを達成した：

$$\mathcal{L}(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[(r + \gamma \max_{a'} Q_{\phi^-}(s', a') - Q_\phi(s, a))^2\right]$$

DQN の二つの重要な技術：

1. **経験リプレイ（Experience Replay）：** 過去の遷移 $(s, a, r, s')$ をリプレイバッファ $\mathcal{B}$ に蓄積し，ミニバッチサンプリングで学習する．連続する遷移の相関を破り，データ効率と安定性を向上させる
2. **ターゲットネットワーク（Target Network）：** TD ターゲットの計算に別の「古い」ネットワーク $Q_{\phi^-}$（$N$ ステップごとに $\phi^- \leftarrow \phi$ と更新）を使う．$Q_\phi$ と TD ターゲットが同時に変動する学習の不安定さを防ぐ

### 8.3.3　連続行動空間の問題

Q 学習は $\arg\max_a Q(s, a)$ の計算を要するが，連続行動空間では全行動を列挙できない．このため，連続制御（ロボット関節角度など）では方策勾配法（8.4節）や Actor-Critic 法（8.5節）が採用される．

---

## 8.4　方策勾配法

### 8.4.1　方策勾配定理

方策をパラメータ $\theta$ でパラメータ化した確率的方策 $\pi_\theta(a \mid s)$ の性能 $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$ を直接最大化する：

$$\theta_{t+1} \leftarrow \theta_t + \eta \nabla_\theta J(\theta_t)$$

**方策勾配定理**（Sutton et al., 1999）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)\right]$$

$\nabla_\theta \log \pi_\theta(a \mid s)$ は**スコア関数（Score Function）**と呼ばれ，行動 $a$ の確率を上げる方向を与える．$Q^{\pi_\theta}(s, a)$ が大きいほど，その行動の確率を強く上げる．

### 8.4.2　REINFORCE アルゴリズム

モンテカルロ推定で $Q^{\pi_\theta}(s, a) \approx G_t$（エピソード全体の累積報酬）を代入した最も基本的な方策勾配法：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i) \cdot G_t^i$$

**ベースライン（Baseline）による分散低減：** $G_t$ の代わりに $G_t - b(s_t)$（$b(s_t)$：状態価値関数 $V^\pi(s_t)$ など）を用いると，勾配の期待値は変わらず分散のみが低下する（$\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot b] = 0$）：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i,t} \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i) \cdot (G_t^i - b(s_t^i))$$

これがアドバンテージ関数 $A^\pi(s, a)$ への接続である（$G_t - V(s_t) \approx A^\pi(s_t, a_t)$）．

### 8.4.3　方策勾配法の問題点

- **高い分散：** エピソード全体のリターンはノイズが多く，多くのサンプルが必要
- **オンポリシー：** 現在の方策 $\pi_\theta$ で収集したデータのみ使用可能（データ効率が低い）
- **学習率の感度：** 大きな更新ステップで方策が崩壊しうる（PPO で対処）

---

## 8.5　Actor-Critic とその発展

### 8.5.1　Actor-Critic の基本

**Actor-Critic** は方策（Actor）と価値関数（Critic）を同時に学習する：

- **Actor** $\pi_\theta(a \mid s)$：方策勾配定理に従って更新
- **Critic** $V_\phi(s)$：TD 誤差でベースラインを学習

$$\mathcal{L}_{\mathrm{critic}} = \mathbb{E}\left[(r + \gamma V_\phi(s') - V_\phi(s))^2\right]$$

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot \underbrace{(r + \gamma V_\phi(s') - V_\phi(s))}_{\text{TD誤差} \approx A^\pi(s,a)}\right]$$

TD 誤差がアドバンテージの推定量として機能し，REINFORCE より低分散を実現する．ただし，Critic $V_\phi$ が真の $V^\pi$ を完全に推定できない場合，**バイアス**が生じる（偏った勾配推定）．分散とバイアスのトレードオフを調整するのが GAE である（次節）．

### 8.5.2　一般化アドバンテージ推定（GAE）

**GAE（Generalized Advantage Estimation）**（Schulman et al., 2016）は，1ステップ TD 誤差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ を指数移動平均で組み合わせることで，バイアスと分散を連続的に制御する：

$$\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

$\lambda = 0$：1ステップ TD（低分散・高バイアス），$\lambda = 1$：モンテカルロリターン（高分散・低バイアス）として，$\lambda \in (0,1)$（通常 $0.95$）でバランスをとる．PPO の実装では GAE によるアドバンテージ推定が標準的に使われる．

### 8.5.3　PPO（Proximal Policy Optimization）

**PPO**（Schulman et al., 2017）は，大きな方策更新による性能崩壊を防ぐために，確率比 $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\mathrm{old}}(a_t \mid s_t)$ をクリップする：

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

$\varepsilon = 0.2$ が標準的．$r_t(\theta)$ が $[1-\varepsilon, 1+\varepsilon]$ の範囲を超えると，それ以上の更新を抑制する．

PPO は実装が簡単で安定しており，OpenAI Five，ロボット操作，LLM の RLHF（第9章）など広い領域で標準的に使われる．

### 8.5.3　SAC（Soft Actor-Critic）

**SAC**（Haarnoja et al., 2018）はエントロピー最大化強化学習の枠組みで，最適方策の目標に**方策エントロピー**を加える：

$$J(\pi) = \mathbb{E}_\pi\left[\sum_t \gamma^t \left(r_t + \alpha H(\pi(\cdot \mid s_t))\right)\right]$$

$H(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ は方策のエントロピー，$\alpha$ は温度パラメータ（学習可能）．エントロピーを最大化する項が探索を促し，学習安定性と多様な行動の学習を促進する．

SAC はオフポリシー手法であり，データ効率が高く，連続制御タスクでの標準的な手法となっている．ロボット実機学習（サンプル効率が重要な場面）でよく使われる．

---

## 8.6　モデルベース強化学習

### 8.6.1　モデルフリーとモデルベースの比較

| | モデルフリー | モデルベース |
|--|------------|------------|
| 遷移モデル | 不要 | 必要（または学習） |
| データ効率 | 低い | 高い |
| 漸近性能 | 高い（バイアスなし） | モデル誤差に依存 |
| 代表例 | DQN，PPO，SAC | Dreamer，MuZero |

### 8.6.2　世界モデル（World Models）

環境の遷移モデル $\hat{P}(s_{t+1} \mid s_t, a_t)$ を学習し，**想像上のシミュレーション**でモデルフリー法を適用することでデータ効率を向上させる（Ha & Schmidhuber, 2018）：

**Dreamer**（Hafner et al., 2020，2023）：RGB 画像から学習した潜在空間の遷移モデル（RSSM: Recurrent State Space Model）内でシミュレートした軌跡で Actor-Critic を学習する．ロボットビジョンベース制御への応用が進んでいる．

### 8.6.3　探索と活用のトレードオフ

強化学習の根本的な課題として，**探索（Exploration）**（新しい行動を試す）と**活用（Exploitation）**（既知の最良行動を取る）のトレードオフがある：

- **$\varepsilon$-greedy：** 確率 $\varepsilon$ でランダム行動，$1-\varepsilon$ で greedy 行動
- **UCB（Upper Confidence Bound）：** 不確実性の大きい行動を優先的に探索（マルチアームバンディット文脈）
- **Entropy Bonus（SAC）：** 方策エントロピーを目標に加え，自動的に探索を促進

---

## まとめ

| 概念 | 内容 |
|------|------|
| MDP | 状態・行動・遷移・報酬・割引率による強化学習の定式化 |
| ベルマン方程式 | 価値関数の再帰的定義；最適価値関数はベルマン最適方程式を満たす |
| Q 学習・DQN | オフポリシー価値ベース法；経験リプレイとターゲットネットワークで安定化 |
| 方策勾配・REINFORCE | 直接の方策最適化；アドバンテージで分散低減 |
| PPO | クリッピングによる安定した方策更新；RLHF の標準手法 |
| SAC | エントロピー最大化オフポリシー法；連続制御の標準 |

次章では，強化学習（特に RLHF）を組み込んだ**大規模言語モデル**の原理と整合技術を詳述する．
