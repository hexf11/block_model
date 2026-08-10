# block_model

针对 Polymarket 5 分钟 UP/DOWN 加密货币市场的短周期价格预测。

Polymarket 用 Chainlink TWAP-30 预言机来结算这类市场。下注是二元的：窗口收盘时预言机报出的价格会高于还是低于开盘价？由于预言机在设计上本身就滞后于现货，当下观测到的交易所订单流就带有关于预言机约 2 分钟后落点的信息。

模型在 **T-110s** 做预测，也就是窗口收盘前 110 秒。

市场：`btc/usd`、`eth/usd`、`sol/usd`、`xrp/usd`、`doge/usd`

## 架构

```
                    ┌──────────────────────────┐
  Polymarket RTDS ──┤                          │
  (Chainlink TWAP)  │                          │
                    │      pm-collector        │──→ ClickHouse (pm.*)
  Binance ─┐        │   (24/7 Rust 守护进程)   │──→ JSONL (data/raw/)
  OKX      │        │                          │
  Bybit    ├────────┤                          │
  Coinbase │        └──────────────────────────┘
  Kraken  ─┘                     │
                                 │ 读取
                                 ▼
                    ┌──────────────────────────┐
                    │       pm-features        │  ← 共享库
                    │  (窗口、盘口、快照)      │     无训练/服务偏差
                    └──────────────────────────┘
                          │              │
              ┌───────────┘              └──────────┐
              ▼                                     ▼
   ┌────────────────────┐              ┌────────────────────────┐
   │  research/ (Python)│              │     pm-inference       │
   │  LightGBM → ONNX   │──.onnx──────→│   HTTP /predict        │
   │  仅离线            │              │   实时                 │
   └────────────────────┘              └────────────────────────┘
                                                   │
                                                   ▼
                                        执行层（独立项目）
```

生产环境全部是 Rust。Python 只出现在 `research/` 里，用于离线训练和模型导出。

### Crates

| Crate | 职责 | 状态 |
|---|---|---|
| `pm-collector` | 24/7 守护进程。每个交易所一个 WebSocket 任务，外加 RTDS，写入 ClickHouse，并有 JSONL 兜底。 | 已完成 |
| `pm-features` | 特征计算 —— 训练快照和实时推理共用的唯一事实来源。 | 滚动窗口和 L2 盘口已完成；快照聚合尚未开始 |
| `pm-inference` | 包装 ONNX 模型的 HTTP 服务。 | 桩实现 —— 固定返回 0.5 |

执行层（下单、仓位管理、风控）刻意**不放**在这个仓库里，它是一个独立项目。

## 安装配置

需要 Rust（stable）和一个可访问的 ClickHouse 实例。

```bash
cp .env.example .env          # 然后修改 CH_PASSWORD 等配置

clickhouse-client < migrations/001_twap.sql
clickhouse-client < migrations/002_exchanges.sql

cargo build --release
./target/release/collector
```

Collector 会并发连接全部六路数据源。每一路独立重连，采用指数退避，所以某一个交易所挂掉不会拖住其他的。它每 60 秒记录一次各交易所的吞吐量。

```bash
cargo test --workspace         # 41 个测试，无需网络
```

## 数据模型

| 表 | 内容 | 保留期 |
|---|---|---|
| `pm.twap` | 来自 Polymarket RTDS 的 Chainlink TWAP-30，每个 symbol 每秒 1 条 | — |
| `pm.ex_bbo` | 各交易所的最优买卖价 | 90 天 |
| `pm.ex_trades` | 逐笔成交，含 taker 方向 | 90 天 |
| `pm.ex_book` | L2 深度快照和增量 | 30 天 |
| `pm.conn_events` | 连接生命周期，用于排查数据缺口 | — |

价格以 `String` 存储，以保留精确的十进制文本 —— 交易所到特征代码之间不做浮点往返转换。

`pm.twap.price` 是 `Decimal128(18)`。Chainlink 的 `full_accuracy_value` 本身就是一个放大了 1e18 的整数，所以直接按放大后的表示原样存储，而不做除法：六位数的 BTC 价格乘上 1e18 会超出 `u64`，而原始值可以无损往返。原始字符串同时保存在 `value_e18` 里。

### 持久性

RTDS 没有重放也没有快照 —— 丢一帧就是永久丢失。因此每条 RTDS 记录都无条件镜像到 JSONL，包括建模集合之外的 symbol（约 2 MB/天，而且万一目标集合扩大，这些额外 symbol 等于白捡）。

交易所数据源约 2500 msg/s。全量镜像会耗掉约 43 GB/天，所以那边的 JSONL 只在 ClickHouse 插入失败时才写。失败的批次落到 `data/raw/<table>.failed.jsonl` 以便重放。

## 给下一个读到这里的人

**时间戳。** 每条记录都同时带 `ts_ex`（交易所时间）和 `recv_ms`（本地接收时间）。特征代码必须按 `recv_ms` 开窗 —— 只有这个时钟才反映某个时点上实际可知的信息。各交易所的时钟彼此不一致，偶尔还会倒退。

不是每个交易所都提供可用的事件时间。Binance 的 `bookTicker` 和 Kraken 的 `ticker` 完全没有这个字段，所以那里的 `ts_ex` 回退到 `recv_ms`；这个回退在 parser 测试里有断言，因为若是悄悄继承本地时钟，反倒会让人误以为这家交易所的时钟同步得可疑地好。

**Taker 方向。** Binance 上报的是 `m`（买方是否为 *maker*），所以 taker 方向是它的反面。搞反的话，所有订单流特征的符号都会翻转，而数值大小看起来仍然合理 —— 这正是那种肉眼扫一遍发现不了的 bug。已有测试覆盖。

**盘口重建。** `is_snapshot` 决定一帧是重置盘口还是修改盘口。Bybit 用 `u == 1` 来表示服务重启，而该帧仍然被标为 `delta`；把它当成增量处理会从那一刻起悄悄搞坏盘口。同样有测试覆盖。

规划内容及其理由见 `docs/roadmap.md`。
