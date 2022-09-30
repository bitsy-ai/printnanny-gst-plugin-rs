use std::i32;
use std::sync::Mutex;

use byte_slice_cast::*;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;

use flexbuffers::{Reader, ReaderError};
use gst_base::subclass::prelude::BaseTransformImpl;
use polars::prelude::*;

use once_cell::sync::Lazy;

use crate::message_nnstreamer;

#[derive(Debug)]
struct Settings {
    pub tensor_shapes: Vec<Vec<u32>>,
    pub tensor_num: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tensor_shapes: vec![
                vec![4, 40, 1, 1],
                vec![40, 1, 1, 1],
                vec![40, 1, 1, 1],
                vec![1, 1, 1, 1],
            ],
            tensor_num: 4 as usize,
        }
    }
}

#[derive(Debug)]
struct State {
    pub df: DataFrame,
}

impl Default for State {
    fn default() -> Self {
        let boxes_x0 =
            Series::new_empty("bounding_boxes_x0", &polars::datatypes::DataType::Float32);
        let boxes_y0 =
            Series::new_empty("bounding_boxes_y0", &polars::datatypes::DataType::Float32);
        let boxes_x1 =
            Series::new_empty("bounding_boxes_x1", &polars::datatypes::DataType::Float32);
        let boxes_y1 =
            Series::new_empty("bounding_boxes_y1", &polars::datatypes::DataType::Float32);

        let class = Series::new_empty("class_labels", &polars::datatypes::DataType::Float32);
        let scores = Series::new_empty("class_scores", &polars::datatypes::DataType::Float32);
        let ts = Series::new_empty("ts", &polars::datatypes::DataType::Int64);
        let df = DataFrame::new(vec![
            ts, boxes_x0, boxes_y0, boxes_x1, boxes_y1, class, scores,
        ])
        .unwrap();
        Self { df }
    }
}

#[derive(Debug)]
pub struct DataframeDecoder {
    // sinkpad: gst::Pad,
    // srcpad: gst::Pad,
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
}

impl DataframeDecoder {
    // fn sink_chain(
    //     &self,
    //     _pad: &gst::Pad,
    //     element: &super::DataframeDecoder,
    //     buffer: gst::Buffer,
    // ) -> Result<gst::FlowSuccess, gst::FlowError> {
    //     Ok(gst::FlowSuccess::Ok)
    // }

    // fn sink_event(
    //     &self,
    //     pad: &gst::Pad,
    //     element: &super::DataframeDecoder,
    //     event: gst::Event,
    // ) -> bool {
    //     gst::log!(CAT, obj: pad, "Handling event {:?}", event);
    //     pad.event_default(Some(element), event)
    // }
}

// This module contains the private implementation details of our element
//
// This module contains the private implementation details of our element
//
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "printnanny",
        gst::DebugColorFlags::empty(),
        Some("Decode polars Dataframe from nnstreamer flatbuffer"),
    )
});

const DEFAULT_SCORE_THRESHOLD: u32 = 50;

#[glib::object_subclass]
impl ObjectSubclass for DataframeDecoder {
    const NAME: &'static str = "DataframeDecoder";
    type Type = super::DataframeDecoder;
    type ParentType = gst_base::BaseTransform;

    fn new() -> Self {
        Self {
            // sinkpad,
            // srcpad,
            settings: Mutex::new(Settings::default()),
            state: Mutex::new(Some(State::default())),
        }
    }
}

impl ObjectImpl for DataframeDecoder {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
            //     glib::ParamSpecString::builder("input-shape")
            //         .nick("Input tensor shape")
            //         .blurb("Shape of incoming tensor buffer.")
            //         .default_value(Some("4:40:1:1,40:1:1:1,40:1:1:1,1:1:1:1"))
            //         .mutable_ready()
            //         .build(),
            //     glib::ParamSpecUInt::builder("score-threshold")
            //         .nick("Score threshold")
            //         .blurb("Filter out bounding box detections below score threshold")
            //         .default_value(DEFAULT_SCORE_THRESHOLD)
            //         .build(),
            ]
        });

        PROPERTIES.as_ref()
    }
}

impl GstObjectImpl for DataframeDecoder {}

impl ElementImpl for DataframeDecoder {
    // Set the element specific metadata. This information is what
    // is visible from gst-inspect-1.0 and can also be programatically
    // retrieved from the gst::Registry after initial registration
    // without having to load the plugin in memory.
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "DataframeDeocder",
                "Filter/Tensor",
                "Decoders Polars Dataframe from Apache Arrow buffer",
                "Leigh Johnson <leigh@printnanny.ai>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }
    // Create and add pad templates for our sink and source pad. These
    // are later used for actually creating the pads and beforehand
    // already provide information to GStreamer about all possible
    // pads that could exist for this type.
    //
    // Actual instances can create pads based on those pad templates
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            // element accepts TFlitePostProcess layer as input
            let incaps = gst::Caps::builder("other/arrow")
                // .field("num_tensors", 4)
                // .field("dimensions", "4:40:1:1,40:1:1:1,40:1:1:1,1:1:1:1")
                // .field("types", "float32,float32,float32,float32")
                // .field("format", "static")
                .build();
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &incaps,
            )
            .unwrap();

            let outcaps = gst::Caps::builder("other/flatbuf-tensor").build();
            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &outcaps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for DataframeDecoder {
    // Configure basetransform so that we are never running in-place,
    // don't passthrough on same caps and also never call transform_ip in passthrough mode
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    // fn accept_caps
    // fn before_transform
    // fn copy_metadata
    // https://github.com/google/flatbuffers/blob/master/samples/sample_flexbuffers_serde.rs
    // * Map {
    //     *   "num_tensors" : UInt32 | <The number of tensors>
    //     *   "rate_n" : Int32 | <Framerate numerator>
    //     *   "rate_d" : Int32 | <Framerate denominator>
    //     *   "tensor_#": Vector | { String | <tensor name>,
    //     *                          Int32 | <data type>,
    //     *                          Vector | <tensor dimension>,
    //     *                          Blob | <tensor data>
    //     *                         }
    //     * }
    fn transform(
        &self,
        element: &Self::Type,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or_else(|| {
            gst::element_error!(element, gst::CoreError::Negotiation, ["Have no state yet"]);
            gst::FlowError::NotNegotiated
        })?;

        let in_buffermap = inbuf.map_readable().or_else(|_| {
            gst::element_error!(
                element,
                gst::CoreError::Failed,
                ["Failed to map input buffer readable"]
            );
            Err(gst::FlowError::Error)
        })?;

        let nnstreamer_tensors =
            message_nnstreamer::nnstreamer::flatbuf::root_as_tensors(in_buffermap.as_slice())
                .map_err(|err| {
                    gst::element_error!(
                        element,
                        gst::CoreError::Failed,
                        [&format!(
                            "Failed to extract Tensor from flatbuffer {:?}",
                            err
                        )]
                    );
                    gst::FlowError::Error
                })?;

        // arrow::ipc::convert::try_schema_from_flatbuffer_bytes(in_buffermap.as_slice()).map_err(
        //     |err| {
        //         gst::element_error!(
        //             element,
        //             gst::CoreError::Failed,
        //             [&format!(
        //                 "Failed to extract arrow schema from flatbuffer {:?}",
        //                 err
        //             )]
        //         );
        //         gst::FlowError::Error
        //     },
        // )?;
        // let root = Reader::get_root(in_buffermap.as_slice()).unwrap();
        // let read_tensors = root.as_map();
        // let attrs: Vec<&str> = read_tensors.iter_keys().collect();
        // let expected_keys = [
        //     "num_tensors",
        //     "rate_n",
        //     "rate_d",
        //     "tensor_0",
        //     "tensor_1",
        //     "tensor_2",
        //     "tensor_3",
        // ];
        // // ensure expected keys are present in incoming flatbuffer
        // for key in expected_keys {
        //     if !attrs.contains(&key) {
        //         gst::error!(
        //             CAT,
        //             "Received flatbuffer without required property: {}",
        //             key
        //         );
        //         return Err(gst::FlowError::Error);
        //     }
        // }

        // let scores_reader = read_tensors.idx("tensor_2").as_vector();
        // let scores_t = scores_reader.idx(3).as_blob().0;
        // let scores_shape = scores_reader
        //     .idx(2)
        //     .as_vector()
        //     .iter()
        //     .map(|el| el.as_u32());
        // let scores_series =
        //     arrow::tensor::Float32Tensor::try_new(arrow::buffer::Buffer::from_bscores_t, scores_shape, None, None);

        // outbuf.append_memory(gst::Memory::from_slice(scores_t));

        // // mapping output buffer as writable
        // let mut out_buffermap = outbuf.map_writable().or_else(|_| {
        //     gst::element_error!(
        //         element,
        //         gst::CoreError::Failed,
        //         ["Failed to map output buffer writable"]
        //     );
        //     Err(gst::FlowError::Error)
        // })?;

        // // output buffer as slice
        // let out_samples = out_buffermap.as_mut_slice_of::<f32>().map_err(|err| {
        //     gst::element_error!(
        //         element,
        //         gst::CoreError::Failed,
        //         ["Failed to cast input buffer as f32 slice: {}", err]
        //     );
        //     gst::FlowError::Error
        // })?;
        // out_samples.

        Ok(gst::FlowSuccess::Ok)
    }
    fn transform_caps(
        &self,
        element: &Self::Type,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let other_caps = if direction == gst::PadDirection::Src {
            gst::Caps::builder("other/flatbuf-tensor").build()
        } else {
            gst::Caps::builder("other/arrow").build()
        };

        gst::debug!(
            CAT,
            obj: element,
            "Transformed caps from {} to {} in direction {:?}",
            caps,
            other_caps,
            direction
        );

        // In the end we need to filter the caps through an optional filter caps to get rid of any
        // unwanted caps.
        if let Some(filter) = filter {
            Some(filter.intersect_with_mode(&other_caps, gst::CapsIntersectMode::First))
        } else {
            Some(other_caps)
        }
    }
    // fn unit_size(&self, _element: &Self::Type, caps: &gst::Caps) -> Option<usize> {
    //     // gst_video::VideoInfo::from_caps(caps).map(|info| info.size())
    //     Some(40 as usize)
    // }
    fn transform_size(
        &self,
        element: &Self::Type,
        _direction: gst::PadDirection,
        _caps: &gst::Caps,
        size: usize,
        _othercaps: &gst::Caps,
    ) -> Option<usize> {
        assert_ne!(_direction, gst::PadDirection::Src);

        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut()?;

        // let othersize = {
        //     let full_blocks = (size + state.adapter.available()) / (state.input_block_size());
        //     full_blocks * state.output_block_size()
        // };

        // gst::log!(
        //     CAT,
        //     obj: element,
        //     "Adapter size: {}, input size {}, transformed size {}",
        //     state.adapter.available(),
        //     size,
        //     othersize,
        // );

        // Some(othersize)
        Some(size)
    }
}
