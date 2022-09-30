use gst::prelude::*;
use gstprintnanny::message_nnstreamer::nnstreamer;
use std::fs;
use std::path::PathBuf;

fn init() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        gst::init().unwrap();
        gstprintnanny::plugin_register_static().unwrap();
    });
}

#[test]
fn test_nnstreamer_callback() {
    init();

    let expected_buffers = 16;
    let pipeline = format!(
        "videotestsrc num-buffers={} \
        ! tensor_converter \
        ! tensor_decoder mode=custom-code option1=printnanny_decoder",
        expected_buffers
    );
    let mut h = gst_check::Harness::new_parse(&pipeline);
    let bus = gst::Bus::new();
    h.element().unwrap().set_bus(Some(&bus));
    h.play();

    let mut num_buffers = 0;
    while let Some(_buffer) = h.pull_until_eos().unwrap() {
        num_buffers += 1;
    }
    assert_eq!(num_buffers, expected_buffers);
}

// #[test]
// fn test_flatbuf_dataframe_decoder() {
//     init();
//     let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//     // let fixture = base_path.join("data/capture.flexbuf");
//     let fixture = base_path.join("data/fixture0.flatbuf");
//     // let fixture = base_path.join("data/fixture0_orig.mp4");
//     let model_path = base_path.join("data/model.tflite");

//     // let pipeline = format!(
//     //     "filesrc location={fixture} \
//     //     ! decodebin \
//     //     ! videoscale \
//     //     ! videoconvert \
//     //     ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGB \
//     //     ! tensor_converter \
//     //     ! tensor_transform mode=arithmetic option=typecast:uint8,add:0,div:1 \
//     //     ! capsfilter caps=other/tensors,num_tensors=1,format=static \
//     //     ! tensor_filter framework=tensorflow2-lite model={model_file} \
//     //     ! tensor_decoder mode=flatbuf",
//     //     fixture = fixture.display(),
//     //     tensor_width = 320,
//     //     tensor_height = 320,
//     //     model_file = model_path.display(),
//     // );

//     let pipeline = format!(
//         "filesrc location={fixture} \
//         ! other/flatbuf-tensor \
//         ! arrow_decoder name=printnannydec",
//         fixture = fixture.display()
//     );

//     // let pipeline = format!(
//     //     "filesrc location={fixture} ! testsink",
//     //     fixture = fixture.display(),
//     // );
//     let mut h = gst_check::Harness::new_parse(&pipeline);
//     let bus = gst::Bus::new();
//     h.element().unwrap().set_bus(Some(&bus));
//     h.play();

//     // h.pull().unwrap();
//     // Pull all buffers until EOS
//     let mut num_buffers = 0;
//     while let Some(_buffer) = h.pull_until_eos().unwrap() {
//         num_buffers += 1;
//     }
//     assert_eq!(num_buffers, 5);
// }

// #[test]
// fn test_flatbuf_arrow_decoder() {
//     init();
//     let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//     let fixture = base_path.join("data/fixture_0.mp4");
//     let model_path = base_path.join("data/model.tflite");

//     // let mut h1 = gst_check::Harness::new("arrow_decoder");
//     let bin = gst::parse_bin_from_description(
//         &format!(
//             "filesrc location={fixture} ! decodebin ! queue \
//             ! videoscale \
//             ! videoconvert \
//             ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGB \
//             ! tensor_converter \
//             ! tensor_transform mode=arithmetic option=typecast:uint8,add:0,div:1 \
//             ! capsfilter caps=other/tensors,num_tensors=1,format=static \
//             ! queue \
//             ! tensor_filter framework=tensorflow2-lite model={model_file} \
//             ! tensor_decoder mode=flexbuf \
//             ! arrow_decoder name=printnannydec",
//             fixture = fixture.display(),
//             tensor_width = 320,
//             tensor_height = 320,
//             model_file = model_path.display(),
//         ),
//         false,
//     )
//     .unwrap();

//     let srcpad = bin
//         .by_name("printnannydec")
//         .unwrap()
//         .static_pad("src")
//         .unwrap();
//     // let _ = bin.add_pad(&gst::GhostPad::with_target(Some("src"), &srcpad).unwrap());
//     let mut h = gst_check::Harness::with_element(&bin, None, Some("src"));
//     h.play();

//     let buf = h.pull().unwrap();
//     let frame = buf.into_mapped_buffer_readable().unwrap();

//     // h1.set_sink_caps_str("other/flexbuf");

//     // h1.play();

//     // push flatbuffer data
//     // let data = fs::read(&fixtures).unwrap();
//     // let buf = gst::Buffer::from_slice(data);

//     // assert_eq!(h1.push(buf), Ok(gst::FlowSuccess::Ok));
// }
