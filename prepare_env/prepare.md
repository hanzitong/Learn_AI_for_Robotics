
# Windows PC での事前準備
インターネット環境必要です。

## vscodeをインストールする
- [参考　install vscode](https://code.visualstudio.com/download)


## vscode内の拡張機能をインストールする
vscode内で下記の拡張機能を検索してインストール。
- Jypter (Jupyter Extension for Visual Studio Code)
  - たぶんjupyterと調べたら出てくる
- C/C++ (C/C++ for Visual Studio Code)
  - たぶんC++と調べたら出てくる。


## uvをインストール
- [参考 uv install](https://docs.astral.sh/uv/getting-started/installation/)
- powershellを開き、下記のコマンドでuvをインストールする
  - ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```


## MSYS2をインストール
- [参考 Example: Install MinGW-x64 on Windows](https://code.visualstudio.com/docs/languages/cpp)
- **MSYS2 UCRT64**を開き、そこからpacmanコマンドを実行（詳細はvscodeのダウンロードページを参照)
  - ```pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain```
  - Enter連打でOK


## 配布コードを入手する
- メールで演習コードを事前配布します。
- 圧縮しているので、C:\Users\社員番号\以下に解凍しておいてください。
<!-- - 社内LAN(Xサーバー)に接続できる
  - Xサーバーに入れときます。
- 社内LANに接続不可 & インターネットあり
  - メールで配信する。
- 社内LANに接続不可 & インターネットなし
  - 通信端末を借してgithubからダウンロードしてもらう。 -->


## uvでpython環境を自動構築する
- 配布ディレクトリ内の、pyproject.toml, uv.lockが存在する場所で以下のコマンドを実行
  - ```uv sync```




