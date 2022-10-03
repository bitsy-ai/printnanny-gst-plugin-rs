use gst::prelude::*;
use gst::MessageView;

use polars::io::ipc::{IpcReader, IpcStreamReader};
use polars::io::SerReader;
use polars::prelude::*;

use std::fs;
use std::fs::File;
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
    let base_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tmp_dir = base_path.join(".tmp");
    let model_path: PathBuf = base_path.join("data/model.tflite");

    let num_detections = 40;

    let expected_buffers = 16;
    let pipeline = format!(
        "videotestsrc num-buffers={expected_buffers} \
        ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGB \
        ! videoscale \
        ! videoconvert \
        ! tensor_converter \
        ! capsfilter caps=other/tensors,num_tensors=1,format=static \
        ! tensor_filter framework=tensorflow2-lite model={model_file} output=4:{num_detections}:1:1,{num_detections}:1:1:1,{num_detections}:1:1:1,1:1:1:1 outputname=detection_boxes,detection_classes,detection_scores,num_detections outputtype=float32,float32,float32,float32 \
        ! tensor_decoder mode=custom-code option1=printnanny_bb_dataframe_decoder",
        expected_buffers = expected_buffers,
        num_detections = num_detections,
        tensor_width = 320,
        tensor_height = 320,
        model_file = model_path.display()
    );
    let mut h = gst_check::Harness::new_parse(&pipeline);
    let bus = gst::Bus::new();
    let element = h.element().unwrap();
    element.set_bus(Some(&bus));
    h.play();

    let mut num_buffers = 0;
    while let Some(buffer) = h.pull_until_eos().unwrap() {
        let cursor = buffer.as_cursor_readable();
        let df = IpcStreamReader::new(cursor)
            .finish()
            .expect("Failed to extract dataframe");

        // dataframe should have 7 columns and num_detections rows
        assert_eq!(df.shape(), (num_detections, 7));

        println!("Pulled dataframe from buffer {:?}", df);
        num_buffers += 1;
    }
    assert_eq!(num_buffers, expected_buffers);
}

#[test]
fn test_dataframe_filesink() {
    init();
    let base_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_path: PathBuf = base_path.join("data/model.tflite");
    let tmp_dir = base_path.join(".tmp");

    let dataframe_location = format!("{}/videotestsrc_%05d.ipc", tmp_dir.display());
    let num_detections = 40;
    let expected_buffers = 16;

    let pipeline_str = format!(
        "videotestsrc num-buffers={expected_buffers} \
        ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGB \
        ! videoscale \
        ! videoconvert \
        ! tensor_converter \
        ! capsfilter caps=other/tensors,num_tensors=1,format=static \
        ! tensor_filter framework=tensorflow2-lite model={model_file} output=4:{num_detections}:1:1,{num_detections}:1:1:1,{num_detections}:1:1:1,1:1:1:1 outputname=detection_boxes,detection_classes,detection_scores,num_detections outputtype=float32,float32,float32,float32 \
        ! tensor_decoder mode=custom-code option1=printnanny_bb_dataframe_decoder \
        ! dataframe_filesink location={dataframe_location}
        ",
        expected_buffers = expected_buffers,
        num_detections = num_detections,
        tensor_width = 320,
        tensor_height = 320,
        model_file = model_path.display()
    );

    let pipeline = gst::parse_launch(&pipeline_str).expect("Failed to construct pipeline");
    pipeline.set_state(gst::State::Playing).unwrap();
    let bus = pipeline.bus().unwrap();
    let mut events = vec![];

    loop {
        let msg = bus.iter_timed(gst::ClockTime::NONE).next().unwrap();

        match msg.view() {
            MessageView::Error(_) | MessageView::Eos(..) => {
                events.push(msg.clone());
                break;
            }
            // check stream related messages
            MessageView::StreamCollection(_) | MessageView::StreamsSelected(_) => {
                events.push(msg.clone())
            }
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    let pattern = format!("{}/*.ipc", tmp_dir.display());
    let paths = glob::glob(&pattern).expect("Failed to parse glob pattern");

    let dataframes: Vec<LazyFrame> = paths
        .map(|p| {
            let p = p.unwrap();
            let f = File::open(&p).expect("file not found");
            IpcStreamReader::new(f).finish().unwrap().lazy()
        })
        .collect();

    let df = concat(&dataframes, true, true).unwrap().collect().unwrap();
    assert_eq!(df.shape(), (expected_buffers * num_detections, 7));
}

#[test]
fn test_dataframe_agg() {
    init();

    let base_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tmp_dir = base_path.join(".tmp");
    let model_path: PathBuf = base_path.join("data/model.tflite");
    let fixture = base_path.join("data/fixture_0.mp4");

    let dataframe_location = format!("{}/fixture_0_%05d.ipc", tmp_dir.display());
    let num_detections = 40;
    let expected_buffers = 16;

    let pipeline_str= format!(
        "filesrc location={fixture} \
        ! decodebin \
        ! queue \
        ! videoscale \
        ! videoconvert \
        ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGB \
        ! tensor_converter \
        ! capsfilter caps=other/tensors,num_tensors=1,format=static \
        ! queue \
        ! tensor_filter framework=tensorflow2-lite model={model_file} output=4:{num_detections}:1:1,{num_detections}:1:1:1,{num_detections}:1:1:1,1:1:1:1 outputname=detection_boxes,detection_classes,detection_scores,num_detections outputtype=float32,float32,float32,float32 \
        ! tensor_decoder mode=custom-code option1=printnanny_bb_dataframe_decoder \
        ! dataframe_filesink location={dataframe_location}",
        fixture = fixture.display(),
        num_detections = num_detections,
        tensor_width = 320,
        tensor_height = 320,
        model_file = model_path.display(),
        dataframe_location = dataframe_location
    );

    let pipeline = gst::parse_launch(&pipeline_str).expect("Failed to construct pipeline");
    pipeline.set_state(gst::State::Playing).unwrap();
    let bus = pipeline.bus().unwrap();
    let mut events = vec![];

    loop {
        let msg = bus.iter_timed(gst::ClockTime::NONE).next().unwrap();

        match msg.view() {
            MessageView::Error(_) | MessageView::Eos(..) => {
                events.push(msg.clone());
                break;
            }
            // check stream related messages
            MessageView::StreamCollection(_) | MessageView::StreamsSelected(_) => {
                events.push(msg.clone())
            }
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
}
