pub fn activation_range(activation: &Option<String>) -> (i8, i8) {
    if let Some(act) = activation {
        match act.as_str() {
            "Relu" => (0, 127),
            "Relu6" => (0, 6),
            _ => panic!("Unsupported activation: {}", act),
        }
    } else {
        (-127, 127)
    }
}
