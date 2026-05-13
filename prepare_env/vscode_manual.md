


# vscode でjupyter notebook を設定するメモ。

## 配布したディレクトリをvscodeで開いてもらう
- .venvが存在するディレクトリ以下の場所を開く必要があることに注意。
- a

## jupyter notebookでPython実行
###
- zipファイルを解凍し、ディレクトリ内に移動。
- uv syncを実行。
- 
### vscodeでの操作
- command palet 開く
- python: select interpreter 
  - .venv環境を選択

  > Developpers: Reload window で仮想環境lecture-aiを認識することが多い。


## windows 環境
### windows上での操作
- MSYS2を入れる。(VSCodeに解説ページあり。)
  - [url](https://code.visualstudio.com/docs/languages/cpp)
- MSYS2 UCRT64からpacmanコマンドを実行（詳細はvscodeのダウンロードページ)
### vscodeでの操作
- 拡張機能をvscodeに入れる。C/C++ extension for VS Code
- 右上のrun C++ fileから実行できる。成功するとターミナルに結果が出てくる。

