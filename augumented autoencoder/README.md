
## 環境
python 3.5.2  
pytorch 1.3.1  

# AAE
## 学習
AAEの学習方法です．
学習結果は--nameで指定したフォルダに出力されます．
ディレクトリckptに重み，ディレクトリresに訓練データの出力，ディレクトリtestに検証データの出力が保存されます．

`$ python3 main.py --root ./dataset/train　--target ./datasets/train_target --test ./dataset/test --test_target ./datasets/test_target --name AAEt`

### 引数のパラメータ
--root ： 入力(変形ARマーカ)データのディレクトリ  
--target : 入力の正解（平面状ARマーカ）データのディレクトリ  
--test ： 検証データ用の入力データのディレクトリ  
--test_target ： 検証データの正解データ  
--name ： 出力結果のディレクトリ  
--batchsz ： バッチサイズ  
--z_dim ： 潜在変数の次元数  
--beta ： 復元誤差の重み  


## データベース作成
学習済みのAAEを用いたデータベースの作成方法です．
各姿勢の潜在変数は--nameで指定したフォルダに出力されます．

`$ python3 pose_encoder2.py --load ./AAEt/ckpt/aae_0000.mdl --pose ./datasets/AAE_pose/０００/ --name data_base`

### 引数のパラメータ
--name ： 出力結果のディレクトリ  
--batchsz ： バッチサイズ  
--z_dim ： 潜在変数の次元数  
--name ： データベースの保管場所
--load ： 学習済みAAEの重み
--pose ： データベースとなるデータのディレクトリ

## 対象画像の潜在変数の取得
学習済みのAAEを用いた潜在変数の取得方法です．
潜在変数は--nameで指定したフォルダに出力されます．

`$　python3 suitei_encoder.py --load ./AAEt_700/ckpt/aae_0000000265.mdl --pose ./datasets/hyouka2/ --name  Z`

### 引数のパラメータ
--name ： 潜在変数の保管場所
--load ： 学習済みAAEの重み
--pose ： 評価データのディレクトリ
