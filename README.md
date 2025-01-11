# スケートボードトリック分類チャレンジ 3位ソリューション

## 概要

このリポジトリは、NEDO Challenge, Motion Decoding Using Biosignalsで実施された[スケートボードトリック分類チャレンジ](https://signate.jp/competitions/1429)で3位を獲得したソリューションを公開しています。  
スケートボーダーの脳波(EEG)からスケートボードトリックの種類を予測します。モデルは1stステージ,2ndステージのスタッキングになっています。  
1stステージ：1D-CNN,2D-CNN,3D-CNN,LSTM  
2ndステージ：pycaretにより最適な学習器を選択  
詳細は[こちら](docs/report.pdf)を参照ください。  

## データセット

コンペティションで使用したデータセットは、こちらの[SOTAコンペティション](https://signate.jp/competitions/1587)で入手できます。

## セットアップ

このプロジェクトをローカル環境で実行するための手順は以下の通りです。

1. リポジトリをクローンします。
    ```bash
    git clone https://github.com/kimpara3/skateboard-tricks.git
    cd skateboard-tricks
    ```
2. 必要なパッケージをインストールします。
    ```bash  
    pip install -r requirements.txt
    ```
3. データセットを`train`及び`test`ディレクトリに展開します。
4. 結果の保存にmlflowを使用しているので、mlflowをセットアップしてください。
    ```bash  
    mlflow
    ```

## 実行方法

1. 1stステージの学習および推論  
    検証用データ及びテストデータでの推論結果が保存されます。  
    2D-CNNと3D-CNNはmlflowにjsonファイルが保存されるので、2ndステージ用に、`conv2d_val`,`conv2d_test`,`r3d_val`,...ディレクトリにコピーしてください。  
    | モデル | 実行ファイル |
    | --- | --- |
    | 1D-CNN | `eeg_conv1d.ipynb` |
    | 2D-CNN | `eeg_conv2d.py` |
    | 3D-CNN | 前処理<br>`python mne_topomap.py`<br>学習・推論<br>`python eeg_r3d.py r3d_18`<br>`python eeg_r3d.py mc3_18`<br>`python eeg_r3d.py r2plus1d_18` |
    | LSTM | `eeg_lstm.ipynb` |
2. 2ndステージの学習および推論  
    `stacking.ipynb`によって、各ディレクトリに保存された1stステージの結果で学習と推論を行います。
    

## ライセンス

このプロジェクトは、MIT Licenseの下でライセンスされています。
