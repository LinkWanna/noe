#![no_std]
#![no_main]
#![allow(static_mut_refs)]

#[cfg(feature = "add")]
mod add;
#[cfg(feature = "linear")]
mod linear;

use defmt::info;
use embassy_executor::Spawner;
use embassy_stm32::{Config, Peripherals, rcc, time::Hertz};

use {defmt_rtt as _, panic_probe as _};

fn init_clock() -> rcc::Config {
    let mut cfg = rcc::Config::new();
    cfg.hsi = false;
    cfg.hse = Some(rcc::Hse {
        freq: Hertz::mhz(8),
        mode: rcc::HseMode::Oscillator,
    });
    cfg.sys = rcc::Sysclk::PLL1_P;
    cfg.pll = Some(rcc::Pll {
        src: rcc::PllSource::HSE,
        prediv: rcc::PllPreDiv::DIV1,
        mul: rcc::PllMul::MUL9,
    });
    cfg.ahb_pre = rcc::AHBPrescaler::DIV1;
    cfg.apb1_pre = rcc::APBPrescaler::DIV2;
    cfg.apb2_pre = rcc::APBPrescaler::DIV1;
    cfg.adc_pre = rcc::ADCPrescaler::DIV6;
    cfg
}

fn init_dwt() {
    unsafe {
        // Enable DWT via DCB
        const DCB_DEMCR: *mut u32 = 0xE000EDFC as *mut u32;
        const DCB_DEMCR_TRCENA: u32 = 1 << 24;
        *DCB_DEMCR |= DCB_DEMCR_TRCENA;

        // Enable cycle counter in DWT
        const DWT_CTRL: *mut u32 = 0xE0001000 as *mut u32;
        const DWT_CTRL_CYCCNTENA: u32 = 1;
        *DWT_CTRL |= DWT_CTRL_CYCCNTENA;
    }
}

fn dwt_cycles() -> u32 {
    unsafe {
        const DWT_CYCCNT: *const u32 = 0xE0001004 as *const u32;
        *DWT_CYCCNT
    }
}

pub fn measure<F: FnMut()>(name: &str, iterations: u32, mut f: F) {
    cortex_m::interrupt::free(|_| {
        let start = dwt_cycles();
        for _ in 0..iterations {
            f();
        }
        let end = dwt_cycles();
        let cycles = end.wrapping_sub(start);
        let avg_cycles = cycles / iterations;
        info!(
            "{}: {} cycles (avg over {} runs)",
            name, avg_cycles, iterations
        );
    });
}

pub fn make_test_i8_generic(ptr: *mut i8, size: usize, val: i8) {
    unsafe {
        for i in 0..size {
            *ptr.add(i) = val.wrapping_add((i % 256) as i8);
        }
    }
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let mut config = Config::default();
    config.rcc = init_clock();
    let _: Peripherals = embassy_stm32::init(config);

    init_dwt();
    info!("Starting operator benchmarks on STM32F103C8 @ 72 MHz");

    #[cfg(feature = "add")]
    add::run();

    #[cfg(feature = "linear")]
    linear::run();

    #[cfg(not(any(feature = "add", feature = "linear")))]
    info!("No benchmark feature enabled");

    info!("Benchmarks completed");
    loop {}
}
