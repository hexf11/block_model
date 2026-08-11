-- ex_book 增加 first_seq 列：增量流的连续性校验依赖它。
--
-- 背景：binance 的 @depth@100ms 是 100 毫秒聚合推送，一条消息覆盖
-- U..u 一整个更新区间。只存末尾的 u，相邻两条之间本来就会跳号——
-- 用 u 做连续性检查会把正常的聚合误判成 96% 的丢帧率（实际发生过）。
--
-- 正确的连续性条件：本条.first_seq == 上条.seq + 1
--
-- 各交易所的字段来源：
--   binance   U          区间首个更新 ID
--   okx       prevSeqId  上一条的 seqId
--   bybit     u          单调递增，无独立区间起点，与 seq 同值
--   coinbase  sequence_num
--   kraken    无序列号，靠 checksum 校验，此列为 0

ALTER TABLE pm.ex_book
    ADD COLUMN IF NOT EXISTS first_seq UInt64 DEFAULT 0 AFTER seq;
