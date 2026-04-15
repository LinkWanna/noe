use defmt::info;
use embassy_stm32::i2c::{self, Master};
use embassy_stm32::mode::Async;
use embassy_time::{Delay, Duration, Ticker};
use mpu6050_dmp::calibration::CalibrationParameters;
use mpu6050_dmp::sensor_async::Mpu6050;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Mpu6050State {
    Idle,
    Sampling,
    Sampled,
}

impl Mpu6050State {
    pub fn next(&self) -> Mpu6050State {
        match self {
            Mpu6050State::Idle => Mpu6050State::Sampling,
            Mpu6050State::Sampling => Mpu6050State::Sampled,
            Mpu6050State::Sampled => Mpu6050State::Idle,
        }
    }
}

pub async fn mpu6050_init(sensor: &mut Mpu6050<i2c::I2c<'static, Async, Master>>) {
    info!("MPU6050 Sensor Initialized");

    // // Reset sensor to ensure it's in a known state
    // sensor.initialize_dmp(&mut Delay).await.unwrap();

    // Configure sensor settings
    sensor
        .set_clock_source(mpu6050_dmp::clock_source::ClockSource::Xgyro)
        .await
        .unwrap();

    // Set accelerometer full scale to +/- 4g for better sensitivity
    sensor
        .set_accel_full_scale(mpu6050_dmp::accel::AccelFullScale::G4)
        .await
        .unwrap();

    sensor
        .set_gyro_full_scale(mpu6050_dmp::gyro::GyroFullScale::Deg500)
        .await
        .unwrap();

    // Configure DLPF for maximum sensitivity
    sensor
        .set_digital_lowpass_filter(mpu6050_dmp::config::DigitalLowPassFilter::Filter6)
        .await
        .unwrap();

    // Set sample rate to 100Hz (10ms period)
    sensor.set_sample_rate_divider(9).await.unwrap();

    info!("Calibrating MPU6050");

    // Configure calibration parameters
    let calibration_params = CalibrationParameters::new(
        mpu6050_dmp::accel::AccelFullScale::G4,
        mpu6050_dmp::gyro::GyroFullScale::Deg500,
        mpu6050_dmp::calibration::ReferenceGravity::ZN,
    );

    sensor
        .calibrate(&mut Delay, &calibration_params)
        .await
        .unwrap();
    info!("MPU6050 Calibrated");
}

pub async fn mpu6050_sampling(
    sensor: &mut Mpu6050<i2c::I2c<'static, Async, Master>>,
    data: &mut [i8],
) {
    let mut ticker = Ticker::every(Duration::from_millis(10));

    for i in 0..150 {
        let (_, gyro) = sensor.motion6().await.unwrap();

        data[i] = (gyro.x() / (8192 / 16)) as i8;
        data[i + 150] = (gyro.y() * (8192 / 16)) as i8;
        data[i + 300] = (gyro.z() * (8192 / 16)) as i8;

        info!(
            "Sample {}: x: {}, y: {}, z: {}",
            i,
            data[i],
            data[i + 150],
            data[i + 300]
        );

        ticker.next().await;
    }

    info!("MPU6050 Sampling Completed");
}
