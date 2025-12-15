# IC_jxy
Code for analysis

### Structure
analysis (核心功能代码附有中文介绍)
│  .gitignore
│  clustering.py 聚类 + shuffle
│  fast_test.py 用于测试不同筛选参数组合的筛选结果
│  four_class.py  用于RR-神经元筛选
│  graph_ana.py 进行图分析
│  image_test.py
│  loaddata.py
│  network.py 构建相关矩阵
│  README.md
│  rr_self.py
│  svm_j.py 初步svm，以及参数测试
│  svm_shuffle.py svm + shuffle
│
├─IC 保存原始数据以及分析结果
│  │  count.py
│  │  modify.py
│  │
│  ├─M21
│  │  │  2025`024_M21-0.dat
│  │  │  2025`024_M21.txt
│  │  │  2025`024_M21_sleep.txt
│  │  │  indexed_mod_2025`024_M21.txt
│  │  │  M21.csv
│  │  │  M21.json
│  │  │  mod_2025`024_M21.txt
│  │  │  visual_stimuli_with_label.mat
│  │  │  wholebrain_output.mat
│  │  │  whole_brain_3d.tif
│  │
│  └─M79
│      │  2025`024_M79-0.dat
│      │  2025`024_M79.txt
│      │  2025`024_M79_sleep.txt
│      │  AVG_whole_brain_3d.tif
│      │  brain_results.mat
│      │  indexed_mod_2025`024_M79.txt
│      │  M79.csv
│      │  M79.json
│      │  mappoints.mat
│      │  mod_2025`024_M79.txt
│
├─stimuli 刺激播放的记录
│      M21.py
│      stimuli_20251024_1108.txt
│      stimuli_20251024_1149.txt
│
├─sub 个人探索性分析尝试
│  │  fast_test_false.py
│  │  four_class_false.py
│  │  graph_false.py
│  │  graph_whole.py
│  │  network_false.py
│  │  network_whole.py
│  │
│  ├─IC
│  │  ├─M21
│  │  └─M79
│  │
│  └─__pycache__
│          four_class_false.cpython-311.pyc
│
└─__pycache__
        four_class.cpython-311.pyc
