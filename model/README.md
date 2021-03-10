## 3Dモデルの構成
gazeboのオブジェクトはsdf形式で扱われます．
オブジェクトは基本的にmodel.sdfとmodel.configで構成されます．
単純な構造はsdfで記述可能ですが複雑な形状はBlenderなどで作成したColladaファイル(.dae)をmeshとして読み込みます．

```
model_gazebo
- models
  - model_name
    - model.sdf
    - model.config
    - meshes
      - model_name.dae //作成した3Dモデル
```

## 動作環境
このパッケージの動作環境です．
Kineticのデフォルトはgazebo 7.0ですがバグがあるためアップデートしてください．

* ROS Kinetic
* Gazebo  7.16.1

## 距離画像の撮影手順
model_gazeboをWORK_SPACE/srcに配置します．
センサを配置したWorldを起動します．
launchファイルは2種類あります．
sensor_world.launchはgazeboがGUIありで起動します．
モデルの確認や距離画像の撮影テストに使用してください．
sensor_world2.launchはgazeboのGUIが非表示のファイルです．
距離画像を大量に撮影する際はGUIがない分処理が軽いのでこちらの使用を推奨します．

```
$ cd ~/WORK_SPACE //WORK_SPACEは自分のワークスペース名
$ catkin_make
$ source devel/setup.bash
$ roslaunch model_gazebo sensor_world.launch
```

3Dモデルの初回設置は反映まで時間がかかるため，撮影に使用するモデルを設置してから消します．
モデルの選択はspawn_model.pyのmodel_listで指定してください．

```
$ cd model_gazebo/src
$ python spawn_model.py
$ rosservice call /gazebo/delete_model "モデル名"
```

距離画像の撮影はimage_collector.pyで行います．
撮影枚数とファイル名は89行目のrangeの範囲で調整可能です．

```
$ cd model_gazebo/src
$ python image_collector.py
```

## 画像の前処理コード
フォルダimgに画像の前処理用コードがあります．
必要に応じて撮影したデータに前処理を行ってください．

* digital_zoom.py：倍率2倍でデジタルズームを行う．
* data_aug.py：上下，左右，上下左右を反転した画像を生成する．
* add_noise.py：画像に正規分布状のノイズを与える．

