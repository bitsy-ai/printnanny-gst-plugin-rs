use gst_sys;
use libc::{c_char, c_int, c_void};
use std::ffi::CString;

#[repr(C)]
pub struct GstTensorMemory {
    pub address: *mut libc::c_char,
    pub address_size: usize,
    pub x: libc::c_int,
}

#[repr(C)]
pub struct GstTensorsConfig {
    pub address: *mut libc::c_char,
    pub address_size: usize,
    pub x: libc::c_int,
}

extern "C" fn tensor_decoder_custom_printnanny_cb(
    input: *const GstTensorMemory,
    config: *const GstTensorsConfig,
    data: libc::c_void,
    out_buf: gst_sys::GstBuffer,
) {
    println!("I'm called from C with value {:?}", input);
}

#[link(name = "nnstreamer")]
extern "C" {
    fn nnstreamer_decoder_custom_register(
        name: *const c_char,
        tensor_decoder_custom: extern "C" fn(
            input: *const GstTensorMemory,
            config: *const GstTensorsConfig,
            data: libc::c_void,
            out_buf: gst_sys::GstBuffer,
        ),
        data: *mut c_void,
    ) -> c_int;
}

pub fn register_nnstreamer_callbacks() {
    unsafe {
        let name = CString::new("printnanny_decoder").unwrap();
        nnstreamer_decoder_custom_register(
            name.as_ptr(),
            tensor_decoder_custom_printnanny_cb,
            std::ptr::null_mut(),
        );
    }
}
