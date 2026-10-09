# 本地数据准备

原始 H&M 数据不随仓库分发，也不应提交到 Git。请自行获取合法的数据副本，并将以下三个文件放入 `data/raw/`：

```text
data/raw/
├── articles.parquet
├── customers.parquet
└── transactions_train.parquet
```

当前阶段使用的本地文件规模如下；这些数值来自 Stage 0A 的 Parquet 文件元数据：

| 文件 | 行数 | 文件大小（bytes） |
| --- | ---: | ---: |
| `articles.parquet` | 105,542 | 6,378,835 |
| `customers.parquet` | 1,371,980 | 167,711,266 |
| `transactions_train.parquet` | 31,788,324 | 774,279,764 |

数据文件不会被项目脚本自动下载。准备完成后，可在项目根目录执行：

```powershell
python scripts/recon_workspace.py
```

该命令只读取 Parquet 元数据和极少量脱敏样本，并重新生成 Stage 0A 报告与清单。
