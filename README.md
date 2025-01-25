# EZBM-CS

> Easy Balanced Mixing Long - Tailed Data Classification System
>
> 基于简单平衡混合的长尾数据分类系统

```
EZBM/
  ├── data/                # 数据模块
  │   ├── __init__.py
  │   ├── dataset.py       # 数据集加载
  │   └── transforms.py    # 数据增强
  ├── models/              # 模型模块
  │   ├── __init__.py
  │   ├── net.py           # 网络结构
  │   └── feature_mix.py   # 新增：特征混合层
  ├── losses/              # 损失函数模块
  │   ├── __init__.py
  │   └── loss.py          # 自定义损失（如Balanced Softmax）
  ├── utils/               # 工具模块
  │   ├── __init__.py
  │   └── utils.py         # 日志、指标计算
  ├── configs/             # 配置文件
  │   └── train_config.yaml
  ├── main.py              # 主训练脚本
  └── README.md            # 项目说明
```

未来又可以做什么？
