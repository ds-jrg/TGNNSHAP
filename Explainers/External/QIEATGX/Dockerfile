# ベースイメージとして Python 3.10 を使用
FROM python:3.10-slim

# 作業ディレクトリを /app に設定
# WORKDIR /QTGX

# カレントディレクトリの内容を /app にコピー
# COPY . /QTGX
# パッケージリストを更新してからインストール
RUN apt-get update && apt-get -y install git

# Pythonのパッケージ管理ツールpipを最新化
RUN pip install --upgrade pip

# 依存関係をインストール
RUN pip install --no-cache-dir .

# Pythonパッケージのインストール
RUN pip install jupyter
RUN pip install "ipywidgets<8.0.0"
RUN pip install jupyterlab
RUN pip install torch_geometric
RUN apt-get install tk-dev