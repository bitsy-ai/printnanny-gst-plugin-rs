use arrow::record_batch::RecordBatch;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::BaseTransformImpl;

use arrow::tensor::{Float32Tensor, Tensor};
use flexbuffers::{Reader, ReaderError};
use once_cell::sync::Lazy;
use polars::prelude::*;
use std::i32;
use std::io::Write;
use std::sync::Mutex;

use crate::message_nnstreamer::nnstreamer;
use crate::tensor::parse_tensor_shapes;




#[derive(Debug)]
struct Settings {
    pub tensor_shapes: Vec<Vec<u32>>,
    pub tensor_num: usize,
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
pub struct ArrowDecoder {
    // sinkpad: gst::Pad,
    // srcpad: gst::Pad,
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
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

impl ArrowDecoder {}

// This module contains the private implementation details of our element
//
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "printnanny",
        gst::DebugColorFlags::empty(),
        Some("Decode Apache Arrow buffer from tensor"),
    )
});

#[glib::object_subclass]
impl ObjectSubclass for ArrowDecoder {
    const NAME: &'static str = "ArrowDecoder";
    type Type = super::ArrowDecoder;
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

impl ObjectImpl for ArrowDecoder {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("tensor-shapes")
                    .nick("Input tensor shapes")
                    .blurb("Shape of incoming tensor buffer.")
                    .default_value(Some("4:40:1:1,40:1:1:1,40:1:1:1,1:1:1:1"))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("tensor-types")
                    .nick("Input tensor shapes")
                    .blurb("Types contained by incoming tensor buffer")
                    .default_value(Some("float32,float32,float32,float32"))
                    .mutable_ready()
                    .build(),
                glib::ParamSpecString::builder("tensor-names")
                    .nick("Input tensor shapes")
                    .blurb("Names of input tensors")
                    .default_value(Some("boxes,class,score,num_detections"))
                    .mutable_ready()
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }
    // Called whenever a value of a property is changed. It can be called
    // at any time from any thread.
    fn set_property(
        &self,
        obj: &Self::Type,
        _id: usize,
        value: &glib::Value,
        pspec: &glib::ParamSpec,
    ) {
        match pspec.name() {
            "input-shape" => {
                let mut settings = self.settings.lock().unwrap();
                let property_value = value.get().expect("Failed to parse tensor_shapes");
                let (tensor_num, tensor_shapes) = parse_tensor_shapes(property_value).unwrap();
                if settings.tensor_shapes != tensor_shapes {
                    settings.tensor_num = tensor_num;
                    settings.tensor_shapes = tensor_shapes;
                }
            }
            _ => unimplemented!(
                "Property {} is not implemented by ArrowDecoder",
                pspec.name()
            ),
        }
    }
}

impl GstObjectImpl for ArrowDecoder {}

impl ElementImpl for ArrowDecoder {
    // Set the element specific metadata. This information is what
    // is visible from gst-inspect-1.0 and can also be programatically
    // retrieved from the gst::Registry after initial registration
    // without having to load the plugin in memory.
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "ArrowDecoder",
                "Converter/Tensor",
                "Decodes Apache Arrow buffer from C-Array",
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
            let srccaps = gst::Caps::builder("other/arrow").build();
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &srccaps,
            )
            .unwrap();

            // let sinkcaps = gst::Caps::builder("other/flatbuf-tensor").build();
            let sinkcaps = gst::Caps::builder("other/flatbuf-tensor").build();

            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &sinkcaps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
    // Called whenever the state of the element should be changed. This allows for
    // starting up the element, allocating/deallocating resources or shutting down
    // the element again.
    fn change_state(
        &self,
        element: &Self::Type,
        transition: gst::StateChange,
    ) -> Result<gst::StateChangeSuccess, gst::StateChangeError> {
        gst::trace!(CAT, obj: element, "Changing state {:?}", transition);

        // Call the parent class' implementation of ::change_state()
        self.parent_change_state(element, transition)
    }
}

// #[derive(Debug, PartialEq, Serialize, Deserialize)]
// struct NnstreamerTensorDim {
//     dimension: Vec<u32>,
//     rank_limit: i32,
// }

// #[derive(Debug, PartialEq, Serialize, Deserialize)]
// struct NnstreamerTensor {
//     tensor_name: String,
//     data_type: i32,
//     tensor_dimension: NnstreamerTensorDim,
//     data: &[u8],
// }

// #[derive(Debug, PartialEq, Serialize, Deserialize)]
// struct NnstreamerSSDTensors {
//     num_tensors: u32,
//     rate_n: i32,
//     rate_d: i32,
//     tensor_0: NnstreamerTensor,
// }

// Implementation of gst_base::BaseTransform virtual methods
impl BaseTransformImpl for ArrowDecoder {
    // Configure basetransform so that we are never running in-place,
    // don't passthrough on same caps and also never call transform_ip in passthrough mode
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(
        &self,
        element: &Self::Type,
        buf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // let mut settings = *self.settings.lock().unwrap();
        // settings.delay = cmp::min(settings.max_delay, settings.delay);

        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;

        let mut map = buf.map_writable().map_err(|_| gst::FlowError::Error)?;
        let nnstreamer_tensors =
            nnstreamer::flatbuf::root_as_tensors(map.as_slice()).map_err(|err| {
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
        let tensors = nnstreamer_tensors.tensor().unwrap();
        let scores_tensor = tensors.get(2);
        let scores_shape: Vec<usize> = scores_tensor
            .dimension()
            .unwrap()
            .iter()
            .map(|v| v as usize)
            .collect();

        // let arrow_tensor = arrow::tensor::Float32Tensor::try_new(
        //     arrow::buffer::Buffer::from(scores_tensor.data().unwrap()),
        //     Some(scores_shape),
        //     None,
        //     None,
        // )
        // .map_err(|err| {
        //     gst::element_error!(
        //         element,
        //         gst::CoreError::Failed,
        //         [&format!(
        //             "Failed to convert flatbuffer tensor data to arrow Float32Tensor {:?}",
        //             err
        //         )]
        //     );
        //     gst::FlowError::Error
        // })?;

        // let mem = gst::memory::Memory::from_slice(arrow_tensor.data());
        // let boxes_x0 =
        //     Series::new_empty("bounding_boxes_x0", &polars::datatypes::DataType::Float32);
        // let boxes_y0 =
        //     Series::new_empty("bounding_boxes_y0", &polars::datatypes::DataType::Float32);
        // let boxes_x1 =
        //     Series::new_empty("bounding_boxes_x1", &polars::datatypes::DataType::Float32);
        // let boxes_y1 =
        //     Series::new_empty("bounding_boxes_y1", &polars::datatypes::DataType::Float32);

        // let class = Series::new_empty("class_labels", &polars::datatypes::DataType::Float32);
        // let scores = Series::new("class_scores", scores_tensor.data());
        // let ts = Series::new_empty("ts", &polars::datatypes::DataType::Int64);
        // let df = DataFrame::new(vec![
        //     ts, boxes_x0, boxes_y0, boxes_x1, boxes_y1, class, scores,
        // ])
        // .unwrap();

        // map.copy_from_slice(arrow_tensor.data());

        gst::warning!(CAT, obj: element, "Wrote arrow_decoder buffer");
        // match state.info.format() {
        //     gst_audio::AUDIO_FORMAT_F64 => {
        //         let data = map.as_mut_slice_of::<f64>().unwrap();
        //         Self::process(data, state, &settings);
        //     }
        //     gst_audio::AUDIO_FORMAT_F32 => {
        //         let data = map.as_mut_slice_of::<f32>().unwrap();
        //         Self::process(data, state, &settings);
        //     }
        //     _ => return Err(gst::FlowError::NotNegotiated),
        // }

        Ok(gst::FlowSuccess::Ok)
    }

    // fn transform(
    //     &self,
    //     element: &Self::Type,
    //     inbuf: &gst::Buffer,
    //     outbuf: &mut gst::BufferRef,
    // ) -> Result<gst::FlowSuccess, gst::FlowError> {
    //     let mut state_guard = self.state.lock().unwrap();
    //     let state = state_guard.as_mut().ok_or_else(|| {
    //         gst::element_error!(element, gst::CoreError::Negotiation, ["Have no state yet"]);
    //         gst::FlowError::NotNegotiated
    //     })?;

    //     let in_buffermap = inbuf.map_readable().or_else(|_| {
    //         gst::element_error!(
    //             element,
    //             gst::CoreError::Failed,
    //             ["Failed to map input buffer readable"]
    //         );
    //         Err(gst::FlowError::Error)
    //     })?;

    //     let nnstreamer_tensors = nnstreamer::flatbuf::root_as_tensors(in_buffermap.as_slice())
    //         .map_err(|err| {
    //             gst::element_error!(
    //                 element,
    //                 gst::CoreError::Failed,
    //                 [&format!(
    //                     "Failed to extract Tensor from flatbuffer {:?}",
    //                     err
    //                 )]
    //             );
    //             gst::FlowError::Error
    //         })?;

    //     let tensors = nnstreamer_tensors.tensor().unwrap();
    //     let scores_tensor = tensors.get(2);
    //     let scores_shape: Vec<usize> = scores_tensor
    //         .dimension()
    //         .unwrap()
    //         .iter()
    //         .map(|v| v as usize)
    //         .collect();

    //     let arrow_tensor = arrow::tensor::Float32Tensor::try_new(
    //         arrow::buffer::Buffer::from(scores_tensor.data().unwrap()),
    //         Some(scores_shape),
    //         None,
    //         None,
    //     )
    //     .map_err(|err| {
    //         gst::element_error!(
    //             element,
    //             gst::CoreError::Failed,
    //             [&format!(
    //                 "Failed to convert flatbuffer tensor data to arrow Float32Tensor {:?}",
    //                 err
    //             )]
    //         );
    //         gst::FlowError::Error
    //     })?;

    //     // let score_array = arrow::datatypes::from(vec![1, 2, 3, 4, 5]);

    //     // let field_scores = arrow::datatypes::Field::new("scores", arrow::datatypes::DataType::Float32, false);
    //     // let schema = arrow::datatypes::Schema::new(vec![field_scores]);

    //     // let batch = arrow::record_batch::RecordBatch::try_new(Arc::new(schema), vec![arrow_tensor]);

    //     let mut buf_cursor = outbuf.as_cursor_writable().map_err(|err| {
    //         gst::element_error!(
    //             element,
    //             gst::CoreError::Failed,
    //             [&format!(
    //                 "Failed to get writable outbuffer cursor {:?}",
    //                 err
    //             )]
    //         );
    //         gst::FlowError::Error
    //     })?;
    //     buf_cursor.write_all(arrow_tensor.data()).map_err(|err| {
    //         gst::element_error!(
    //             element,
    //             gst::CoreError::Failed,
    //             [&format!(
    //                 "Failed to write arrow tensor data to buffer cursor {:?}",
    //                 err
    //             )]
    //         );
    //         gst::FlowError::Error
    //     })?;

    //     gst::warning!(CAT, obj: element, "Wrote arrow_decoder buffer");

    //     // let tensor_type: ArrowPrimitiveType = match scores_tensor.type_() {
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_INT32 => arrow::datatypes::Int32Type,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_UINT32 => arrow::tensor::UInt32Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_INT16 => arrow::tensor::Int16Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_UINT16 => arrow::tensor::UInt16Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_INT8 => arrow::tensor::Int8Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_UINT8 => arrow::tensor::UInt8Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_FLOAT64 => arrow::tensor::Float64Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_FLOAT32 => arrow::tensor::Float32Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_INT64 => arrow::tensor::Int64Tensor,
    //     //     nnstreamer::flatbuf::Tensor_type::NNS_UINT64 => arrow::tensor::UInt64Tensor,
    //     // };

    //     Ok(gst::FlowSuccess::Ok)
    // }

    // Called for converting caps from one pad to another to account for any
    // changes in the media format this element is performing.
    //
    // In our case that means that:
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
}
