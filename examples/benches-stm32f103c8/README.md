# STM32F103C8 Embedded Benchmark

实时 NOE 操作符基准测试，在 STM32F103C8T6 微控制器（ARM Cortex-M3, 72MHz）上运行。

## 功能

- **DWT 循环计数器**：使用核心内的 Data Watchpoint and Trace (DWT) 单元进行精确的 CPU 周期计数
- **拓展式 I8 量化操作符**：Add、Linear、Conv1d 等
- **RTT 实时日志输出**：使用 defmt-rtt 通过调试器实时输出结果

## 硬件要求

- STM32F103C8T6 开发板（或兼容的板卡）
- ST-LINK v2 调试器（或兼容）
- USB 线缆
- 主机与 probe-rs 和 openocd 连接环境

## 构建

### Debug 模式
```bash
cargo build
```

### Release 模式（推荐用于更准确的计时）
```bash
cargo build --release
```

生成的二进制位于：`target/thumbv7m-none-eabi/release/benches-stm32f103c8`

## 部署到硬件

### 使用 probe-rs（推荐）
```bash
probe-rs run --chip stm32f103c8 --release
```

### 使用 OpenOCD
```bash
openocd -f interface/stlink.cfg -f target/stm32f1x.cfg
# 在另一个终端
cargo build --release
arm-none-eabi-gdb target/thumbv7m-none-eabi/release/benches-stm32f103c8
(gdb) target remote :3333
(gdb) load
(gdb) c
```

## 基准测试

### Add 操作符
- 大小：512 元素
- 迭代：1000 次
- 输出：平均周期数

### Linear 操作符（32→16）
- 形状：32 输入特征 → 16 输出特征
- 迭代：200 次
- 输出：平均周期数

## 内存布局

```
STM32F103C8 资源：
- FLASH: 64 KB
- RAM: 20 KB

当前使用：
- Add 缓冲区: 512 * 3 = 1.5 KB
- Linear 缓冲区: 32*16 + 32 + 16*2 = 0.7 KB
- Stack + 其他: ~16 KB
```

## 性能指标解释

输出示例：
```
add_1024: 2500 cycles (avg over 1000 runs)
linear_32x16: 8200 cycles (avg over 200 runs)
```

- **cycles**: 执行操作所用的 CPU 循环周期总数
- **avg over N runs**: N 次迭代的平均值

## 实时查看输出

使用 defmt RTT 查看器：
```bash
# 使用 cargo-embed
cargo embed --release
```

## 调试

如果遇到编译问题：
1. 确保安装了 `thumbv7m-none-eabi` 工具链：
   ```bash
   rustup target add thumbv7m-none-eabi
   ```

2. 检查 probe-rs 安装：
   ```bash
   cargo install probe-rs-tools
   ```

3. 对于 Windows 用户，可能需要配置 USB 驱动程序或使用 libusb

## 扩展

要添加更多操作符基准测试：

1. 在 `src/main.rs` 中添加静态缓冲区
2. 实现初始化函数（见 `make_test_i8_generic`）
3. 创建操作符实例并调用 `measure()` 辅助函数

示例：
```rust
unsafe {
    // 初始化缓冲区
    make_test_i8_generic(MY_BUFFER.as_mut_ptr(), size, init_val);
    
    // 创建操作符
    let mut op = MyOperator::new(...);
    
    // 基准测试
    measure("my_operator", iterations, || {
        op.forward_chw();
    });
}
```

## 参考资源

- [ARM Cortex-M DWT](https://developer.arm.com/documentation/dui0314/h/Cortex-M-Peripherals/Data-Watchpoint-and-Trace-Unit)
- [Embassy STM32 文档](https://docs.embassy.dev/embassy-stm32/)
- [probe-rs 文档](https://probe.rs/)

## 许可

与 NOE 库相同。
