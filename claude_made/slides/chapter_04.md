---
marp: true
theme: default
paginate: true
math: katex
---

# 第4章　汎化理論と正則化技術

**学習の真の目標は訓練データへの適合ではなく，未知データへの予測精度である**

---

## 訓練誤差が低くてもテスト誤差が高い状態が「過学習」である

- 機械学習の目標：真のリスク（汎化誤差）$R(\theta) = \mathbb{E}_{p}[\ell(f_\theta(x), y)]$ の最小化
- 訓練データしか観測できないため，経験リスク $\hat{R}(\theta)$ を代理として最小化する
- **過学習（Overfitting）：** 訓練誤差 $\ll$ テスト誤差 → バリアンスが高い
- **過小適合（Underfitting）：** 訓練誤差 $\approx$ テスト誤差，両者が高い → バイアスが高い

---

## バイアス・バリアンス分解が，モデル複雑度と汎化の関係を定量化する

回帰タスクにおける予測の期待二乗誤差の分解：

$$\mathbb{E}_{\mathcal{D}}[(f^*(x) - \hat{f}(x;\mathcal{D}))^2] = \underbrace{(\mathbb{E}_\mathcal{D}[\hat{f}] - f^*)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_\mathcal{D}[(\hat{f} - \mathbb{E}_\mathcal{D}[\hat{f}])^2]}_{\text{Variance}}$$

- **Bias：** モデルの表現力不足による系統的誤差（モデルが単純すぎる）
- **Variance：** データの変動への感度（モデルが複雑すぎる）

---

## 過パラメータ化された深層モデルでは「二重降下」が起きる

- 古典的統計学：モデル複雑度の最適点（Bias-Variance のスイートスポット）が存在
- **二重降下現象（Belkin et al., 2019；Nakkiran et al., 2020）：**
  - 補間しきい値（$p = n$）を超えてモデルを大きくすると，汎化誤差が再び低下
  - SGD の暗黙の正則化が「最小ノルム解」に収束する傾向を持つ
- 現代の深層学習は意図的に補間しきい値を超えた領域で動作している

---

## VC次元が，仮説クラスの複雑度と汎化上界を結ぶ

**PAC 学習上界（確率 $1-\delta$ で成立）：**

$$R(\hat{f}) \leq \hat{R}(\hat{f}) + O\!\left(\sqrt{\frac{d_{\mathrm{VC}} \log n + \log(1/\delta)}{n}}\right)$$

- $d_{\mathrm{VC}}$：VC次元（仮説クラスの複雑度），$n$：データ数
- 線形分類器（$d$ 次元）：$d_{\mathrm{VC}} = d+1$
- 実際の深層ネットワークは VC 次元が非常に大きいが，PAC 上界より良い汎化性能を示す → SGD の暗黙の正則化が寄与

---

## $\ell_2$ 正則化は「パラメータを小さく保つ」という MAP 推定の実装である

損失にパラメータノルムの二乗を加える：

$$\mathcal{L}_{\mathrm{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$$

- 勾配更新：$\theta_{t+1} \leftarrow \theta_t(1-\eta\lambda) - \eta\nabla_\theta\mathcal{L}$ → 各ステップでパラメータが縮小（**重み減衰**）
- ベイズ的解釈：等方ガウス事前分布 $p(\theta) = \mathcal{N}(0, \lambda^{-1}I)$ のもとでの MAP 推定に相当
- $\ell_1$ 正則化（Lasso）：スパース解を誘導；ラプラス事前分布に対応

---

## Dropout は推論時の動作を変えずに訓練時に確率的マスクを適用する

各ニューロンを確率 $p$ で無効化し，残りをスケールアップする（**Inverted Dropout**）：

$$m_j \sim \text{Bernoulli}(1-p), \quad \tilde{h}_j = \frac{m_j}{1-p} \cdot h_j$$

- スケーリングにより，訓練時と推論時の期待値が一致する → 推論時はそのまま全ニューロンを使用
- **解釈：** $2^n$ 通りのサブネットワークの近似的アンサンブル学習
- **注意：** Batch Normalization と組み合わせると相互干渉が生じやすい

---

## Batch Normalization は活性化の分布を安定させ，高い学習率を可能にする

ミニバッチ統計量で活性化を正規化し，学習可能なパラメータでスケール・シフト：

$$\hat{z} = \frac{z - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \varepsilon}}, \quad \tilde{z} = \gamma \odot \hat{z} + \beta$$

- 推論時：訓練中に蓄積した指数移動平均 $\hat{\mu}$，$\hat{\sigma}^2$ を使用
- **CNN に適している；** バッチサイズが小さいと統計推定が不安定

---

## Layer Normalization はバッチサイズに依存せず Transformer で標準となっている

特徴次元方向で正規化：

$$\mu_L = \frac{1}{d}\sum_{j=1}^{d} z_j, \quad \hat{z} = \frac{z - \mu_L}{\sqrt{\sigma_L^2 + \varepsilon}}$$

| | Batch Norm | Layer Norm |
|--|-----------|-----------|
| 正規化軸 | バッチ（サンプル間） | 特徴次元 |
| バッチサイズ依存 | あり | なし |
| 主な用途 | CNN | Transformer，RNN |

---

## データ拡張は，元のデータ量を増やさずに分布の多様性を広げる

- **画像：** ランダムクロップ・水平反転・色相変換・Cutout・RandAugment
- **テキスト：** 同義語置換，逆翻訳
- **ロボットデータ：** センサノイズ付加，剛体変換（第12章）
- **早期停止（Early Stopping）：** バリデーション損失が改善しなくなった時点で訓練終了 → 追加コストなしの暗黙的正則化

---

## まとめ：汎化と正則化の設計指針

| 概念 | 内容 |
|------|------|
| バイアス・バリアンス | 過学習 = 高バリアンス；過小適合 = 高バイアス |
| 二重降下 | 過パラメータ化域では汎化誤差が再低下 |
| PAC / VC 次元 | 有限データでの汎化誤差の理論的上界 |
| $\ell_1/\ell_2$ 正則化 | ノルム抑制；MAP 推定と等価 |
| Dropout | アンサンブル効果；Inverted Dropout で推論コストなし |
| BN / LN | 活性化の分布安定化；用途で使い分け |

次章では，空間的局所性という帰納バイアスを構造に埋め込んだ CNN を詳述する
