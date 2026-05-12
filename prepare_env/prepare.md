
# Windows PC での事前準備
## vscodeをインストールする
- [vscode installer download](https://code.visualstudio.com/download)

## vscode内で拡張機能**jupyter**を検索してインストールする。

## powershellを開き、下記のコマンドでuvをインストールする
- ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```
- [参考 uv install](https://docs.astral.sh/uv/getting-started/installation/)

## 配布コードを入手する
- 社内LAN(Xサーバー)に接続できる
  - Xサーバーに入れときます。
- 社内LANに接続不可 & インターネットあり
  - メールで配信する。
- 社内LANに接続不可 & インターネットなし
  - 通信端末を借してgithubからダウンロードしてもらう。
- 

## uvで環境構築する
- pyproject.toml, uv.lockが存在するディレクトリ内で```uv sync```を実行する




