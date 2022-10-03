use byte_slice_cast::*;

use std::collections::BTreeMap;
use std::ffi::CString;
use std::io::BufWriter;
use std::{ptr, slice}; // or NativeEndian

use gst::prelude::*;

use gst_sys;
use gst_sys::{gst_buffer_get_size, GST_FLOW_ERROR, GST_FLOW_OK};
use ndarray::ShapeBuilder; // Needed for .strides() method
use once_cell::sync::Lazy;

use polars::export::arrow::io::ipc::write;

use polars::prelude::*;

use libc::{c_char, c_float, c_int, c_void, size_t};

const NNS_TENSOR_RANK_LIMIT: usize = 4;
const NNS_TENSOR_SIZE_LIMIT: usize = 16;
const NNS_TENSOR_SIZE_LIMIT_STR: &str = "16";

// This module contains the private implementation details of our element
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "printnanny",
        gst::DebugColorFlags::empty(),
        Some("Decode polars Dataframe from nnstreamer tensors"),
    )
});

#[repr(C)]
#[derive(Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    NNS_INT32,
    NNS_UINT32,
    NNS_INT16,
    NNS_UINT16,
    NNS_INT8,
    NNS_UINT8,
    NNS_FLOAT64,
    NNS_FLOAT32,
    NNS_INT64,
    NNS_UINT64,
    NNS_FLOAT16,
    /**< added with nnstreamer 2.1.1-devel. If you add any operators (e.g., tensor_transform) to float16, it will either be not supported or be too inefficient. */
    NNS_END,
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub enum TensorFormat {
    NNS_TENSOR_FORMAT_STATIC,
    NNS_TENSOR_FORMAT_FLEXIBLE,
    NNS_TENSOR_FORMAT_SPARSE,
    NNS_TENSOR_FORMAT_END,
}

pub type TensorDimension = [u32; NNS_TENSOR_RANK_LIMIT];

#[repr(C)]
#[derive(Debug)]
pub struct GstTensorMemory {
    pub data: *mut c_void, // pointer to mapped gstreamer memory
    pub size: size_t,
}

#[repr(C)]
#[derive(Debug)]
pub struct GstTensorInfo {
    pub name: *const c_char,
    pub tensor_type: TensorType,
    pub tensor_dim: TensorDimension, // NNstreamer framework supports up to 4th rank
}

#[repr(C)]
#[derive(Debug)]
pub struct GstTensorsInfo {
    pub num_tensors: c_int,
    pub info: [GstTensorInfo; NNS_TENSOR_RANK_LIMIT],
}

#[repr(C)]
#[derive(Debug)]
pub struct GstTensorsConfig {
    pub info: GstTensorsInfo,
    pub format: TensorFormat, // tensor stream type
    pub rate_n: c_int,        //framerate is in fraction, which is numerator/denominator
    pub rate_d: c_int,        //  framerate is in fraction, which is numerator/denominator
}

// based on: https://github.com/nnstreamer/nnstreamer/blob/f2c3bcd87f34ac2ad52ca0a17f6515c54e6f2d66/tests/nnstreamer_decoder/unittest_decoder.cc#L28
extern "C" fn printnanny_bb_dataframe_decoder(
    input: *const GstTensorMemory,
    config: *const GstTensorsConfig,
    data: libc::c_void,
    out_buf: *mut gst_sys::GstBuffer,
) -> i32 {
    let num_tensors = unsafe { (*config).info.num_tensors };
    if num_tensors != 4 {
        gst::error!(
            CAT,
            "printnanny_bb_dataframe_decoder requires a tensor with rank 4, but got tensor with rank {}", num_tensors
        );
        return GST_FLOW_ERROR;
    }

    // data / sanity checks
    let df_config = unsafe { config.as_ref().clone() };
    if df_config.is_none() {
        gst::error!(
            CAT,
            "printnanny_bb_dataframe_decoder received NULL GstTensorsConfig"
        );
        return GST_FLOW_ERROR;
    }
    let df_config = df_config.unwrap();
    // ensure memory layout matches expected tensor shape, which is:
    // 4:N:1:1,N:1:1:1,N:1:1:1,1:1:1:1 where N is the number of detections returned
    let input_data = unsafe { std::slice::from_raw_parts(input, num_tensors as usize) };

    // assert tensor dimensions are in expected shape
    if df_config.info.info[0].tensor_dim[0] != 4 {
        gst::error!(
            CAT,
            "printnanny_bb_dataframe_decoder expected tensor 0 to have shape 4:N:1:1, but received shapes {:?}",
            df_config.info.info
        );
        return GST_FLOW_ERROR;
    }
    if df_config.info.info[0].tensor_dim[1] != df_config.info.info[1].tensor_dim[0]
        || df_config.info.info[0].tensor_dim[1] != df_config.info.info[2].tensor_dim[0]
    {
        gst::error!(
            CAT,
            "printnanny_bb_dataframe_decoder expected tensor 1/2 to have shape N:1:1:1, but received shapes {:?}",
            df_config.info.info
        );
        return GST_FLOW_ERROR;
    }

    if df_config.info.info[0].tensor_type != TensorType::NNS_FLOAT32
        || df_config.info.info[1].tensor_type != TensorType::NNS_FLOAT32
        || df_config.info.info[2].tensor_type != TensorType::NNS_FLOAT32
        || df_config.info.info[3].tensor_type != TensorType::NNS_FLOAT32
    {
        gst::error!(
            CAT,
            "printnanny_bb_dataframe_decoder expected tensors to be FLOAT32, but received types: {:?}",
            df_config.info.info
        );
        return GST_FLOW_ERROR;
    }

    // beyond this point, memory is guaranteed to be mapped to rank 4 float32 tensor with shapes 4:N:1:1,N:1:1:1,N:1:1:1,1:1:1:1

    gst::log!(
        CAT,
        "printnanny_bb_dataframe_decoder handling tensors {:?} with shapes {:?}",
        input_data,
        df_config.info
    );

    let ts = gst::util_get_timestamp().nseconds();

    // flatten bounding boxes into x0, y0 x1, y1 columns
    let num_boxes = df_config.info.info[0].tensor_dim[0];
    let num_detections: u32 = df_config.info.info[0].tensor_dim[1];
    let boxes = unsafe { slice::from_raw_parts(input_data[0].data as *mut u8, input_data[0].size) };
    let boxes = boxes.as_slice_of::<c_float>().unwrap().to_vec();
    let boxes =
        ndarray::Array::from_shape_vec((num_detections as usize, num_boxes as usize), boxes)
            .expect("Failed to deserialize GstTensorMemory into detection_boxes ndarray");
    // create classes / labels ndarrays
    let classes =
        unsafe { slice::from_raw_parts(input_data[1].data as *mut u8, input_data[1].size) };
    let classes: Vec<i32> = classes
        .as_slice_of::<c_float>()
        .unwrap()
        .to_vec()
        .iter()
        .map(|v| *v as i32)
        .collect();

    let scores =
        unsafe { slice::from_raw_parts(input_data[2].data as *mut u8, input_data[2].size) };
    let scores = scores.as_slice_of::<c_float>().unwrap().to_vec();

    let mut df = df!(
        "detection_boxes_x0" => boxes.column(0).to_vec(),
        "detection_boxes_y0" => boxes.column(1).to_vec(),
        "detection_boxes_x1" => boxes.column(2).to_vec(),
        "detection_boxes_y1" => boxes.column(3).to_vec(),
        "detection_classes" => classes,
        "detection_scores" => scores,
        "ts" => vec![ts; num_detections as usize]
    )
    .expect("Failed to initialize dataframe");

    let metadata = BTreeMap::from([
        ("frame_rate_n".to_string(), df_config.rate_n.to_string()),
        ("frame_rate_d".to_string(), df_config.rate_d.to_string()),
    ]);

    let arrow_schema = df.schema().to_arrow();
    let arrow_schema = arrow_schema.with_metadata(metadata);

    let mut bufwriter = std::io::BufWriter::new(Vec::new());
    let mut ipcwriter =
        write::StreamWriter::new(&mut bufwriter, write::WriteOptions { compression: None });

    ipcwriter
        .start(&arrow_schema, None)
        .expect("Failed to initialize ipc writer from arrow schema}");

    df.rechunk();
    for batch in df.iter_chunks() {
        ipcwriter
            .write(&batch, None)
            .expect("Failed to write chunk to ipcwriter");
    }
    ipcwriter
        .finish()
        .expect("Failed to finalize ipcwriter buffer");

    let arrow_msg = bufwriter
        .into_inner()
        .expect("Failed to flush arrow ipc buffer");

    // derefrence a pointer to GstBuffer, allocate memory from gstreamer memory pool
    let gstbufref = unsafe { gst::BufferRef::from_mut_ptr(out_buf) };
    let gstmem = gst::Memory::with_size(arrow_msg.len());
    gstbufref.replace_all_memory(gstmem);

    let mut buffermap = gstbufref
        .map_writable()
        .expect("Failed to map writable buffer");

    // println!("Writing buffer {:?}", arrow_msg);

    buffermap.copy_from_slice(&arrow_msg);

    return GST_FLOW_OK;
}

#[link(name = "nnstreamer")]
extern "C" {
    fn nnstreamer_decoder_custom_register(
        name: *const c_char,
        tensor_decoder_custom: extern "C" fn(
            input: *const GstTensorMemory,
            config: *const GstTensorsConfig,
            data: libc::c_void,
            out_buf: *mut gst_sys::GstBuffer,
        ) -> i32,
        data: *mut c_void,
    ) -> c_int;
}

pub fn register_nnstreamer_callbacks() {
    unsafe {
        let name = CString::new("printnanny_bb_dataframe_decoder").unwrap();
        nnstreamer_decoder_custom_register(
            name.as_ptr(),
            printnanny_bb_dataframe_decoder,
            std::ptr::null_mut(),
        );
    }
}
