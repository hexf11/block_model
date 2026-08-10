//! pm-collector 的库接口。
//!
//! 对外暴露，使得 parser 可以针对抓取到的帧做单元测试，无需打开 socket。
//! `collector` 二进制使用的正是这些模块。

pub mod ch;
pub mod config;
pub mod exchanges;
pub mod rtds;
pub mod sink;
