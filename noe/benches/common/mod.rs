#![allow(dead_code)]

use std::slice;

pub const ACT_MIN: isize = -127;
pub const ACT_MAX: isize = 127;

pub fn leak_i8_slice(data: Vec<i8>) -> &'static mut [i8] {
    Box::leak(data.into_boxed_slice())
}

pub fn leak_i16_slice(data: Vec<i16>) -> &'static mut [i16] {
    Box::leak(data.into_boxed_slice())
}

pub fn i8_bytes(slice: &'static [i8]) -> &'static [u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len()) }
}

pub fn i16_bytes(slice: &'static [i16]) -> &'static [u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * 2) }
}

pub fn make_i8_data(len: usize, offset: i8) -> Vec<i8> {
    (0..len)
        .map(|idx| ((idx as i32 * 17 + offset as i32) % 255 - 128) as i8)
        .collect()
}

pub fn make_i16_data(len: usize, offset: i16) -> Vec<i16> {
    (0..len)
        .map(|idx| ((idx as i32 * 31 + offset as i32) % 8192 - 4096) as i16)
        .collect()
}
