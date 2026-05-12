---
marp: true
theme: default
paginate: true
math: katex
---

# 第10章　拡散モデルの生成メカニズム

**ノイズへの段階的破壊プロセスの逆転を学習することで，高品質な生成を実現する**

---

## 拡散モデルは「ノイズ付加の逆過程」を学習する生成モデルである

二つのプロセスを対として定義：

1. **Forward Process（拡散過程）：** データ $x_0 \sim p_{\mathrm{data}}$ に段階的にノイズを加え，$x_T \approx \mathcal{N}(0, I)$ に変換（学習不要・固定）
2. **Reverse Process（逆拡散過程）：** ノイズ $x_T$ から逆方向にノイズを除去して $x_0$ を復元（ニューラルネットで学習）

---

## Forward Process の閉形式により，任意の時刻 $t$ のノイズ付きデータを直接計算できる

$\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$ と定義すると：

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)I)$$

すなわち，

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

$t \to T$ で $\bar{\alpha}_T \to 0$ → $x_T \approx \mathcal{N}(0, I)$（純粋なノイズ）

---

## DDPM の学習目標は ELBO から導出される「ノイズ予測 MSE 損失」に帰着する

ELBO の最小化は各ステップの KL ダイバージェンスの最小化に分解され，$\varepsilon$-パラメータ化を通じて：

$$\mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|_2^2\right]$$

- $\varepsilon_\theta(x_t, t)$：ノイズを予測する U-Net（第5章のアーキテクチャを使用）
- $\varepsilon$：実際に加えたノイズ（正解）
- **$v$-予測：** $v = \sqrt{\bar{\alpha}_t}\varepsilon - \sqrt{1-\bar{\alpha}_t}x_0$ を予測する変形；高解像度で有利（Stable Diffusion XL）

---

## ノイズ予測ネットワークはスコア関数の推定量と等価である

**スコア関数（Score Function）：** 対数密度の勾配 $s(x) = \nabla_x \log p(x)$

Song et al.（2021）による接続：

$$\varepsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t}\log q(x_t)$$

- ノイズ予測 $\varepsilon_\theta$ は，データ密度の高い方向へのポインタ（スコア関数）を推定している
- この解釈から，拡散モデルは**スコアマッチング**（Song & Ermon, 2019）と統一的に理解できる

---

## DDIM は確定論的サンプリングにより生成ステップを $T \to 50$ に削減する

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{予測された}x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\varepsilon_\theta + \sigma_t\varepsilon_t$$

- $\sigma_t = 0$：完全に決定論的（DDIM）；同じノイズから常に同じ画像
- $\sigma_t = \tilde{\beta}_t$：確率的（DDPM）
- 同じ $\varepsilon_\theta$ を使いながら $T=1000 \to 50$ ステップへ大幅削減

---

## フローマッチングは直線的な経路を学習し，さらに少ないステップで生成する

ノイズ $x_0 \sim \mathcal{N}(0,I)$ とデータ $x_1 \sim p_{\mathrm{data}}$ を線形補間：

$$x_t = (1-t)x_0 + tx_1, \quad t \in [0,1]$$

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,x_0,x_1}\!\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

- 目標ベクトルフィールドが定数（$x_1 - x_0$）→ 直線的な軌跡 → より少ないステップで近似可能
- **Stable Diffusion 3，Flux.1，π0 ロボット基盤モデル（第14章）で採用**

---

## CFG はガイダンス強度で生成品質と多様性のトレードオフを制御する

**Classifier-Free Guidance（Ho & Salimans, 2022）：**

$$\tilde{\varepsilon}_\theta(x_t, t, c) = (1+w)\varepsilon_\theta(x_t, t, c) - w\varepsilon_\theta(x_t, t, \emptyset)$$

$c$：条件（テキストプロンプト），$\emptyset$：条件なし，$w > 0$：ガイダンス強度

- $w$ が大きい → 条件への忠実度↑，多様性↓
- Stable Diffusion では $w = 7.5$ がデフォルト

---

## Diffusion Policy は行動の多峰性分布を条件付き拡散モデルで自然に扱う

ロボット操作では，同一の観測に対して複数の有効な行動が存在する（**多峰性**）：
- 通常の MSE 回帰 → 平均的な行動を学習 → 多峰性で失敗
- 拡散モデル → 条件付き分布 $p(a \mid o)$ を陰的にモデル化 → 多峰性に対応

$$\mathcal{L} = \mathbb{E}_{t, a^0, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(a^k, k, o_t)\|^2\right]$$

観測 $o_t$ を条件に，将来の行動チャンク $a_{t:t+T}$ を生成

---

## まとめ：拡散モデルの設計原理

| 概念 | 内容 |
|------|------|
| Forward Process | 閉形式の $q(x_t \mid x_0)$ で任意時刻のノイズを直接計算 |
| 学習目標 | ELBO から導出されるノイズ予測 MSE |
| スコアマッチング | $\varepsilon_\theta$ は対数密度勾配の推定量と等価 |
| DDIM | 決定論的サンプリング；$T=1000 \to 50$ ステップ |
| フローマッチング | 線形経路でさらに効率化；Flux.1，π0 で採用 |
| Diffusion Policy | 行動の多峰性分布をロボット模倣学習に応用 |

次章では，テキスト・画像・点群などの複数モダリティを統合するマルチモーダル学習を詳述する
