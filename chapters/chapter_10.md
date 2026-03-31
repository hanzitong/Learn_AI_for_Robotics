# 第10章　拡散モデルの生成メカニズム

## 概要

**拡散モデル（Diffusion Model）**は，データに段階的にノイズを加えて破壊するプロセス（forward process）と，ノイズから元のデータを復元するプロセス（reverse process）を学習することで，高品質なデータを生成する確率的生成モデルである．2020年代に入り，画像生成の品質と多様性においてGANを凌駕し，現在の生成AIの主流となっている．本章では，**DDPM（Denoising Diffusion Probabilistic Models）**の数理的定式化，スコアマッチングとの接続，効率的なサンプリング手法（DDIM，フローマッチング），そしてロボット行動生成への応用（Diffusion Policy）を詳述する．

---

## 10.1　拡散モデルの直感

### 10.1.1　基本的なアイデア

拡散モデルは二つのプロセスを対として定義する：

1. **Forward Process（拡散過程）：** データ $x_0 \sim p_{\mathrm{data}}(x_0)$ に段階的にガウスノイズを加え，$T$ ステップ後に純粋なノイズ $x_T \sim \mathcal{N}(0, I)$ に近づける（学習不要，既知の過程）
2. **Reverse Process（逆拡散過程）：** ノイズ $x_T$ から逆方向にノイズを除去して $x_0$ を復元する過程をニューラルネットワークで学習する（生成に使う）

画像生成の直感：スタジオ写真（$x_0$）から徐々に砂嵐（$x_T$）になる $T$ ステップの動画を考える．逆再生すれば砂嵐から写真が生まれる——これを「学習する」のが拡散モデルである．

---

## 10.2　DDPM（Denoising Diffusion Probabilistic Models）

### 10.2.1　Forward Process の定式化

Ho et al.（2020）が提案した DDPM では，forward process を以下のマルコフ連鎖として定義する：

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

$\{\beta_t\}_{t=1}^{T}$ はノイズスケジュール（通常 $\beta_1 < \beta_2 < \cdots < \beta_T$，線形または余弦スケジュール）．

**Reparameterization による閉形式：** $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ と定義すると，任意の時刻 $t$ への遷移が閉形式で表される：

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

すなわち，

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

$t \to T$ で $\bar{\alpha}_T \to 0$ となるよう設計されているため，$x_T \approx \mathcal{N}(0, I)$．

### 10.2.2　Reverse Process の定式化

**Reverse process** は真の事後分布 $q(x_{t-1} \mid x_t, x_0)$（forward の逆）を近似する分布 $p_\theta(x_{t-1} \mid x_t)$ を学習する：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

$q(x_{t-1} \mid x_t, x_0)$ は $x_0$ が既知であれば解析的に計算できる（ガウス分布の乗法）：

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I\right)$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \beta_t$$

### 10.2.3　訓練目標：ノイズ予測

DDPM の訓練目標はまず変分下界（ELBO）から導出される：

$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] = -L_T - \sum_{t=2}^{T} L_{t-1} - L_0$$

各項は KL ダイバージェンスとして整理され，特に主要項 $L_{t-1} = D_{\mathrm{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$ は 両者が ガウス分布であるため解析的に計算可能．さらに $\mu_\theta(x_t, t)$ を $\varepsilon$-パラメータ化（$\varepsilon_\theta$ でノイズを予測し $\mu_\theta$ を導出）すると，全項の重み付き和が**ノイズ予測損失**に帰着する：

$$\mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|_2^2\right]$$

ここで $\varepsilon_\theta(x_t, t)$ はノイズを予測するニューラルネットワーク（通常 U-Net アーキテクチャ），$\varepsilon \sim \mathcal{N}(0, I)$ は実際に加えたノイズである．

**予測パラメータ化の等価性：** ノイズ $\varepsilon$ の予測と，元データ $x_0$ の直接予測（$x_0$-prediction）は，$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$ を通じて相互変換でき，実質等価である．**$v$-予測**（$v = \sqrt{\bar{\alpha}_t}\varepsilon - \sqrt{1-\bar{\alpha}_t}x_0$）は高解像度生成や数値安定性で有利であり，Stable Diffusion XL 等で採用されている．

**訓練アルゴリズム（DDPM）：**

```
1. x_0 ~ p_data(x_0) をサンプル
2. t ~ Uniform{1, ..., T} をサンプル
3. ε ~ N(0, I) をサンプル
4. x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε を計算
5. L = ||ε - ε_θ(x_t, t)||² を最小化
```

### 10.2.4　サンプリング（生成）

学習後のサンプリングは以下のように行う：

```
1. x_T ~ N(0, I)
2. t = T, T-1, ..., 1 について:
   - ε_hat = ε_θ(x_t, t)
   - x_0_hat = (x_t - sqrt(1-ᾱ_t) ε_hat) / sqrt(ᾱ_t)
   - μ_θ = μ̃_t(x_t, x_0_hat)
   - x_{t-1} = μ_θ + sqrt(β̃_t) z, z ~ N(0, I) [t>1の場合]
3. x_0 を返す
```

DDPM では $T = 1000$ ステップのサンプリングが必要で，計算コストが高い．

---

## 10.3　スコアマッチングとの接続

### 10.3.1　スコア関数と拡散モデル

**スコア関数（Score Function）**は対数密度の勾配である：

$$s_\theta(x) = \nabla_x \log p_\theta(x)$$

Song & Ermon（2019）は，スコア関数を推定する**スコアマッチング**による生成モデルを提案した（NCSN）．拡散モデルとの重要な接続（Song et al., 2021）：

$$\varepsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)$$

ノイズ予測ネットワーク $\varepsilon_\theta$ は，ノイズ化されたデータ $x_t$ の対数密度勾配 $\nabla_{x_t} \log q(x_t)$ の（スケール変換した）推定量と等価である．

### 10.3.2　確率微分方程式（SDE）の統一的枠組み

Song et al.（2021）「Score-Based Generative Modeling through Stochastic Differential Equations」は，離散時間の拡散を連続時間の SDE として定式化した：

**Forward SDE：**
$$dx = f(x, t)dt + g(t)dW$$

**Reverse SDE（Anderson, 1982）：**
$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{W}$$

この枠組みはDDPM，NCSN，および多くの拡散モデルを統一的に記述できる．

---

## 10.4　効率的なサンプリング

### 10.4.1　DDIM（Denoising Diffusion Implicit Models）

**DDIM**（Song et al., 2020）は，DDPM のマルコフ連鎖をノンマルコフ過程に置き換え，同じ学習済み $\varepsilon_\theta$ を使いながら少ないステップ数でサンプリングできる：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}}_{\text{予測された } x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \varepsilon_\theta(x_t,t)}_{\text{方向成分}} + \sigma_t \varepsilon_t$$

$\sigma_t = 0$ のとき完全に決定論的（DDIM）で，$\sigma_t = \tilde{\beta}_t$ のとき確率的（DDPM に一致）．

実用上，DDPM の $T=1000$ ステップから DDIM の $S=50$ ステップへの大幅な削減が可能（Ancestral Sampling でも $S=50 \sim 200$ ステップが一般的）．

### 10.4.2　フローマッチング（Flow Matching）

**フローマッチング**（Lipman et al., 2022；Liu et al., 2022）は，ノイズ分布からデータ分布への連続的な確率的フローを学習する拡散モデルの一般化：

$$\frac{dx}{dt} = v_\theta(x, t)$$

（$v_\theta$：学習済みのベクトルフィールド，$t \in [0, 1]$）

**条件付きフローマッチング（CFM）** は，単純な線形補間フロー（**Rectified Flow**）を直接回帰することで学習：

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right], \quad x_t = (1-t)x_0 + t x_1$$

（$x_0 \sim \mathcal{N}(0,I)$，$x_1 \sim p_{\mathrm{data}}$）

直線的なフロー経路は曲線的な拡散プロセスよりも少ないステップで近似でき，SD3（Stable Diffusion 3），Flux.1，π0 ロボット基盤モデル（第14章）などで採用されている．

---

## 10.5　条件付き生成

### 10.5.1　Classifier-Free Guidance（CFG）

**Classifier-Free Guidance**（Ho & Salimans, 2022）は，条件付き生成の品質（忠実度）と多様性のトレードオフをガイダンススケール $w$ で制御する：

$$\tilde{\varepsilon}_\theta(x_t, t, c) = (1+w) \varepsilon_\theta(x_t, t, c) - w \varepsilon_\theta(x_t, t, \emptyset)$$

$c$：条件（テキストプロンプトなど），$\emptyset$：条件なし（Null 条件），$w > 0$：ガイダンス強度．

$w$ が大きいほど条件への忠実度が上がるが多様性が低下する．Stable Diffusion では $w = 7.5$ がデフォルト．

### 10.5.2　テキスト条件付き画像生成

**Stable Diffusion**（Rombach et al., 2022）は，ピクセル空間ではなく**潜在空間**で拡散を行う Latent Diffusion Model（LDM）：

1. 事前学習済み VAE の潜在空間に画像をエンコード（$512 \times 512 \to 64 \times 64 \times 4$）
2. 潜在空間で拡散モデルを学習
3. 生成後に VAE デコーダでピクセル空間に復元

テキスト条件には CLIP テキストエンコーダを使用し，U-Net のクロスアテンション（第6章）で注入する．

---

## 10.6　ロボット行動生成への応用：Diffusion Policy

### 10.6.1　Diffusion Policy の動機

拡散モデルをロボットの**行動生成**に適用する **Diffusion Policy**（Chi et al., 2023）は，従来の行動空間における多峰性分布（同一状態に対して複数の有効な行動がある場合）を自然に扱える：

- 通常の回帰モデル（MSE）は平均的な行動を学習 → 多峰性では失敗
- 拡散モデルは条件付き分布 $p(a \mid o)$ を陰的に学習 → 多峰性をモデル化可能

### 10.6.2　Diffusion Policy の定式化

観測 $o_t$（カメラ画像，関節角など）を条件に，将来の行動系列 $a_{t:t+T}$ を生成する条件付き拡散モデル：

$$\mathcal{L} = \mathbb{E}_{t, a^0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(a^k, k, o_t)\|^2\right]$$

（$a^0$：真の行動，$a^k$：$k$ ステップのノイズ付き行動，$o_t$：観測コンテキスト）

**Consistency Policy**（Wang et al., 2024）など，少ないステップ数での推論を可能にする後継手法が提案されており，実機ロボット実験でのリアルタイム性が改善されている．

---

## まとめ

| 概念 | 内容 |
|------|------|
| Forward Process | ノイズスケジュールによる段階的データ破壊；$q(x_t \mid x_0)$ が閉形式で計算可能 |
| Reverse Process | ノイズ除去 U-Net $\varepsilon_\theta(x_t, t)$ の学習；目的関数はノイズ予測 MSE |
| スコアマッチング | スコア関数 $\nabla_x \log p$ の推定との等価性 |
| DDIM | 決定論的サンプリングで $T=1000 \to 50$ ステップ |
| フローマッチング | 線形フロー経路でさらに効率化；Flux.1，π0 で採用 |
| Diffusion Policy | 行動の多峰性分布をモデル化；ロボット模倣学習に応用 |

次章では，テキスト・画像・音声など複数のモダリティを統合する**マルチモーダル学習**を詳述する．
