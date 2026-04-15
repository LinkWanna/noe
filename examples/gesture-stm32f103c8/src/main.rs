#![no_std]
#![no_main]

mod generated;
mod mpu6050;

use defmt::info;
use embassy_executor::Spawner;
use embassy_stm32::{
    Config, Peripherals, bind_interrupts, dma, exti, gpio, i2c, interrupt, peripherals, rcc,
    time::{Hertz, khz},
    usart,
};
use mpu6050_dmp::{address::Address, sensor_async::Mpu6050};

use crate::{
    generated::model_run,
    mpu6050::{Mpu6050State, mpu6050_init},
};
use {defmt_rtt as _, panic_probe as _};

bind_interrupts!(struct Irqs {
    USART1 => usart::BufferedInterruptHandler<peripherals::USART1>;
    EXTI0 => exti::InterruptHandler<interrupt::typelevel::EXTI0>;

    I2C1_EV => i2c::EventInterruptHandler<peripherals::I2C1>;
    I2C1_ER => i2c::ErrorInterruptHandler<peripherals::I2C1>;
    DMA1_CHANNEL6 => dma::InterruptHandler<peripherals::DMA1_CH6>;
    DMA1_CHANNEL7  => dma::InterruptHandler<peripherals::DMA1_CH7>;
});

fn argmax(input: &[i8]) -> usize {
    let mut max_index = 0;
    let mut max_value = input[0];
    for (i, &value) in input.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    max_index
}

static MOTION_NAMES: [&str; 13] = [
    "RightAngle", // 0
    "SharpAngle", // 1
    "Lightning",  // 2
    "Triangle",   // 3
    "Letter_h",   // 4
    "letter_R",   // 5
    "letter_W",   // 6
    "letter_phi", // 7
    "Circle",     // 8
    "UpAndDown",  // 9
    "Horn",       // 10
    "Wave",       // 11
    "NoMotion",   // 12
];

fn init_clock() -> rcc::Config {
    let mut cfg = rcc::Config::new();

    // 1. Disable internal clock
    cfg.hsi = false;

    // 2. Configure external high-speed clock HSE
    cfg.hse = Some(rcc::Hse {
        freq: Hertz::mhz(8),            // Clock source frequency is 8MHz
        mode: rcc::HseMode::Oscillator, // Use external crystal oscillator
    });

    // 3. Set system clock to PLL output
    cfg.sys = rcc::Sysclk::PLL1_P;

    // 4. Configure PLL parameters
    cfg.pll = Some(rcc::Pll {
        src: rcc::PllSource::HSE,     // Use HSE as PLL input source
        prediv: rcc::PllPreDiv::DIV1, // Set PLL input pre-divider to 1
        mul: rcc::PllMul::MUL9,       // Set PLL multiplication factor to 9
    });

    // 5. Configure high-performance bus (AHB)
    cfg.ahb_pre = rcc::AHBPrescaler::DIV1;
    // 6. Configure low-speed peripheral bus (APB1), ≤36 MHz limit
    cfg.apb1_pre = rcc::APBPrescaler::DIV2;
    // 7. Configure high-speed peripheral bus (APB2)
    cfg.apb2_pre = rcc::APBPrescaler::DIV1;
    // 8. Configure ADC clock, ≤14 MHz limit
    cfg.adc_pre = rcc::ADCPrescaler::DIV6;

    cfg
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let mut config = Config::default();
    config.rcc = init_clock();
    let p: Peripherals = embassy_stm32::init(config);

    // button
    let mut button = exti::ExtiInput::new(p.PA0, p.EXTI0, gpio::Pull::None, Irqs);

    // mpu6050
    let scl = p.PB6;
    let sda = p.PB7;
    let mut i2c_config = i2c::Config::default();
    i2c_config.frequency = khz(400);
    i2c_config.gpio_speed = gpio::Speed::VeryHigh;
    let i2c = i2c::I2c::new(p.I2C1, scl, sda, p.DMA1_CH6, p.DMA1_CH7, Irqs, i2c_config);
    let mut mpu6050 = Mpu6050::new(i2c, Address::default()).await.unwrap();

    mpu6050_init(&mut mpu6050).await;
    let input = &mut [0i8; 150 * 3];
    let output = &mut [0i8; 13];
    let mut start;
    let mut end;

    let mut state = Mpu6050State::Idle;
    loop {
        match state {
            Mpu6050State::Idle => {
                button.wait_for_rising_edge().await;
                info!("Key pressed, start sampling");
                state = state.next();
            }
            Mpu6050State::Sampling => {
                mpu6050::mpu6050_sampling(&mut mpu6050, input).await;
                state = state.next();
            }
            Mpu6050State::Sampled => {
                start = embassy_time::Instant::now().as_millis();
                model_run(input, output);
                end = embassy_time::Instant::now().as_millis();

                info!("Inference output: {:?}", output);
                info!("Predicted class: {}", MOTION_NAMES[argmax(output)]);
                info!("Inference time: {} ms", end - start);

                state = state.next();
            }
        }
    }
}
