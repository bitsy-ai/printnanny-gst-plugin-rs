// Build PrintNanny Gstreamer pipeline

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use gst::element_error;
use gst::element_warning;
use gst::glib;
use gst::prelude::*;

use anyhow::{Context, Error, Result};
use clap::{crate_authors, crate_description, value_parser, Arg, ArgMatches, Command};
use env_logger::Builder;
use rand::Rng;

use git_version::git_version;
use log::{error, info, warn, LevelFilter};
use thiserror::Error;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use printnanny_gst_config::config::{PrintNannyGstPipelineConfig, VideoSrcType};

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "PrintNannyGstPipeline",
        gst::DebugColorFlags::empty(),
        Some("PritnNanny demo video pipeline"),
    )
});

#[derive(Debug, Error)]
struct ErrorMessage {
    src: String,
    error: String,
    debug: Option<String>,
    source: glib::Error,
}

impl fmt::Display for ErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use `self.number` to refer to each positional data point.
        write!(
            f,
            "Received error from {}: {} (debug: {:?})",
            self.src, self.error, self.debug
        )
    }
}

#[derive(Clone, Debug, glib::Boxed)]
#[boxed_type(name = "ErrorValue")]
struct ErrorValue(Arc<Mutex<Option<Error>>>);

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PipelineApp {
    config: PrintNannyGstPipelineConfig,
}

impl PipelineApp {
    fn make_common_pipeline(&self) -> Result<gst::Pipeline, Error> {
        let start = SystemTime::now();
        let ts = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards, we've got bigger problems");
        let pipeline_name = format!("pipeline-{:?}", &ts);

        let video_udp_port = self.config.video_udp_port.clone();

        let video_width = self.config.video_width.clone();
        let video_height = self.config.video_height.clone();
        let tflite_model_file = self.config.tflite_model.model_file.clone();
        let tensor_height = self.config.tflite_model.tensor_height.clone();
        let tensor_width = self.config.tflite_model.tensor_width.clone();
        let video_framerate = self.config.video_framerate.clone();
        let tflite_label_file = self.config.tflite_model.label_file.clone();
        let nms_threshold = self.config.tflite_model.nms_threshold.clone();
        let overlay_udp_port = self.config.overlay_udp_port.clone();
        let nats_server_uri = self.config.nats_server_uri.clone();

        let pipeline = gst::Pipeline::new(Some(&pipeline_name));
        let h264_queue = gst::ElementFactory::make("queue").name("h264_q").build()?;

        let video_tee = gst::ElementFactory::make("tee")
            .name("tee__inputvideo")
            .build()?;

        let encoder = match gst::ElementFactory::make("v4l2h264enc")
            .property_from_str("extra-controls", "controls,repeat_sequence_header=1")
            .build()
        {
            Ok(el) => el,
            Err(_) => {
                warn!("v4l2h264enc not found, falling back to openh264enc");
                gst::ElementFactory::make("openh264enc").build()?
            }
        };

        let video_h264_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__video_h264_level")
            .build()?;
        video_h264_capsfilter.set_property(
            "caps",
            gst::Caps::builder("video/x-h264")
                .field("level", "3")
                .field("profile", "main")
                .build(),
        );

        let video_ssrc: u32 = rand::thread_rng().gen_range(0..=2147483647);
        let video_payloader = gst::ElementFactory::make("rtph264pay")
            .name("rtph264pay__video")
            .property("config-interval", 1)
            .property_from_str("aggregate-mode", "zero-latency")
            .property_from_str("pt", "96")
            .property("ssrc", video_ssrc)
            .build()?;

        let video_udp_sink = gst::ElementFactory::make("udpsink")
            .name("udpsink__video")
            .property("port", video_udp_port)
            .build()?;

        let raw_video_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__inputvideo")
            .build()?;
        raw_video_capsfilter.set_property(
            "caps",
            gst_video::VideoCapsBuilder::new()
                .width(video_width)
                .height(video_height)
                .framerate(video_framerate.into())
                .build(),
        );

        let invideoconverter = match gst::ElementFactory::make("v4l2convert")
            .name("videoconvert__input")
            .build()
        {
            Ok(el) => el,
            Err(_) => {
                warn!("v4l2convert not found, falling back to videoconvert");
                gst::ElementFactory::make("videoconvert")
                    .name("videoconvert__input")
                    .build()?
            }
        };

        let invideorate = gst::ElementFactory::make("videorate")
            .name("videorate__input")
            .build()?;

        let invideoscaler = gst::ElementFactory::make("videoscale")
            .name("videoscale__input")
            .build()?;

        // split h264-encoded video stream with a tee for compatibility with OctoPrint's webcam plugin
        // OctoPrint's WebRTC support is still experimental and is not compatible with Janus Gateway's signaling
        // Instead, write a ringbuffer of hls frames + playlist to disk (HTTP connection handled by nginx, outside of scope of this pipeline)

        let rtp_queue = gst::ElementFactory::make("queue")
            .name("queue__rtph264pay")
            .build()?;

        let insert_h264_sinks = |octoprint_compat: bool| -> Result<()> {
            match octoprint_compat {
                true => {
                    let h264_tee = gst::ElementFactory::make("tee")
                        .name("tee__h264_video")
                        .build()?;

                    let hls_queue = gst::ElementFactory::make("queue")
                        .name("queue__hlssink")
                        .build()?;

                    let hls_sink = gst::ElementFactory::make("hlssink2")
                        .property("location", &self.config.hls_segments)
                        .property("playlist-location", &self.config.hls_playlist)
                        .property("playlist-root", &self.config.hls_playlist_root)
                        .build()?;
                    let h264_video_elements = &[
                        &invideoconverter,
                        &invideorate,
                        &invideoscaler,
                        &raw_video_capsfilter,
                        &video_tee,
                        &h264_queue,
                        &encoder,
                        &video_h264_capsfilter,
                        &h264_tee,
                    ];

                    let hls_elements = &[&hls_queue, &hls_sink];
                    let rtp_elements = &[&rtp_queue, &video_payloader, &video_udp_sink];
                    pipeline.add_many(h264_video_elements)?;
                    pipeline.add_many(hls_elements)?;
                    pipeline.add_many(rtp_elements)?;

                    gst::Element::link_many(h264_video_elements)?;
                    gst::Element::link_many(&[&h264_tee, &hls_queue, &hls_sink])?;
                    gst::Element::link_many(&[
                        &h264_tee,
                        &rtp_queue,
                        &video_payloader,
                        &video_udp_sink,
                    ])?;

                    for e in h264_video_elements {
                        e.sync_state_with_parent()?
                    }

                    for e in hls_elements {
                        e.sync_state_with_parent()?
                    }

                    for e in rtp_elements {
                        e.sync_state_with_parent()?
                    }

                    Ok(())
                }
                false => {
                    let h264_video_elements = &[
                        &invideoconverter,
                        &invideorate,
                        &invideoscaler,
                        &raw_video_capsfilter,
                        &video_tee,
                        &h264_queue,
                        &encoder,
                        &video_h264_capsfilter,
                        &rtp_queue,
                        &video_payloader,
                        &video_udp_sink,
                    ];
                    pipeline.add_many(h264_video_elements)?;
                    gst::Element::link_many(h264_video_elements)?;

                    for e in h264_video_elements {
                        e.sync_state_with_parent()?
                    }

                    Ok(())
                }
            }
        };

        match self.config.hls_http_enabled {
            Some(true) => insert_h264_sinks(true)?,
            Some(false) => insert_h264_sinks(false)?,
            None => match self.config.detect_hls_http_enabled()? {
                true => insert_h264_sinks(true)?,
                false => insert_h264_sinks(false)?,
            },
        };

        let tensor_q = gst::ElementFactory::make("queue")
            .name("queue__leaky")
            .property_from_str("leaky", "2")
            .build()?;

        let tensor_vconverter = gst::ElementFactory::make("videoconvert")
            .name("videoconvert__tflite_dim")
            .build()?;

        let tensor_videoscale = gst::ElementFactory::make("videoscale").build()?;

        let tensor_converter = gst::ElementFactory::make("tensor_converter").build()?;
        let tensor_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__tensor")
            .build()?;

        tensor_capsfilter.set_property(
            "caps",
            gst::Caps::builder("other/tensors")
                .field("format", "static")
                .build(),
        );

        let tensor_transform = gst::ElementFactory::make("tensor_transform")
            .property_from_str("mode", "arithmetic")
            .property_from_str("option", "typecast:uint8,add:0,div:1")
            .build()?;

        let tensor_filter = gst::ElementFactory::make("tensor_filter")
            .property_from_str("framework", "tensorflow2-lite")
            .property_from_str("model", &tflite_model_file)
            .build()?;
        let raw_rgb_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__tflite_dim")
            .build()?;

        raw_rgb_capsfilter.set_property(
            "caps",
            gst_video::VideoCapsBuilder::new()
                .format(gst_video::VideoFormat::Rgb)
                .width(tensor_width)
                .height(tensor_height)
                // .framerate(tensor_framerate.into())
                .build(),
        );

        let tflite_output_tee = gst::ElementFactory::make("tee")
            .name("tflite_output_tee")
            .build()?;

        let tensor_pipeline_elements = &[
            &tensor_q,
            &tensor_vconverter,
            &tensor_videoscale,
            &raw_rgb_capsfilter,
            &tensor_converter,
            &tensor_transform,
            &tensor_capsfilter,
            &tensor_filter,
            &tflite_output_tee,
        ];
        pipeline.add_many(tensor_pipeline_elements)?;
        gst::Element::link_many(&[&video_tee, &tensor_q])?;
        gst::Element::link_many(tensor_pipeline_elements)?;

        let box_decoder_q = gst::ElementFactory::make("queue")
            .name("queue__box_decoder")
            .build()?;
        let box_decoder = gst::ElementFactory::make("tensor_decoder")
            .name("tensor__decoder_boxes")
            .property_from_str("mode", "bounding_boxes")
            .property_from_str("option1", "mobilenet-ssd-postprocess")
            .property_from_str("option2", &tflite_label_file)
            .property_from_str("option3", &format!("0:1:2:3,{}", nms_threshold))
            .property_from_str("option4", &format!("{video_width}:{video_height}"))
            .property_from_str("option5", &format!("{tensor_width}:{tensor_height}"))
            .build()?;
        let box_videoconverter = gst::ElementFactory::make("videoconvert")
            .name("videoconvert__boxes")
            .build()?;
        let box_videorate = gst::ElementFactory::make("videorate")
            .name("videorate__boxes")
            .build()?;
        let raw_box_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__boxes")
            .build()?;

        // drop/duplicate frames to match input video framerate
        raw_box_capsfilter.set_property(
            "caps",
            gst_video::VideoCapsBuilder::new()
                .framerate(video_framerate.into())
                .build(),
        );

        let box_h264encoder = match gst::ElementFactory::make("v4l2h264enc")
            .property_from_str("extra-controls", "controls,repeat_sequence_header=1")
            .build()
        {
            Ok(el) => el,
            Err(_) => {
                warn!("v4l2h264enc not found, falling back to openh264enc");
                gst::ElementFactory::make("openh264enc").build()?
            }
        };

        let box_h264_capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter__box_h264_level")
            .build()?;
        box_h264_capsfilter.set_property(
            "caps",
            gst::Caps::builder("video/x-h264")
                .field("level", "3")
                .field("profile", "main")
                .build(),
        );

        let boxes_ssrc: u32 = rand::thread_rng().gen_range(0..=2147483647);
        let boxes_payloader = gst::ElementFactory::make("rtph264pay")
            .name("rtph264pay__boxes")
            .property("config-interval", 1)
            .property_from_str("aggregate-mode", "zero-latency")
            .property_from_str("pt", "96")
            .property("ssrc", boxes_ssrc)
            .build()?;

        let box_udpsink_q2 = gst::ElementFactory::make("queue2")
            .name("queue2__boxes_udpsink")
            .build()?;

        let box_udpsink = gst::ElementFactory::make("udpsink")
            .name("udpsink__boxes")
            .property("port", overlay_udp_port)
            .build()?;

        let df_decoder_q = gst::ElementFactory::make("queue")
            .name("queue__df_decoder")
            .build()?;

        let box_overlay_elements = &[
            &box_decoder_q,
            &box_decoder,
            &box_videoconverter,
            &box_videorate,
            &raw_box_capsfilter,
            &box_h264encoder,
            &box_h264_capsfilter,
            &boxes_payloader,
            &box_udpsink_q2,
            &box_udpsink,
        ];

        let dataframe_decoder = gst::ElementFactory::make("tensor_decoder")
            .name("tensor_decoder__df")
            .property("mode", "custom-code")
            .property("option1", "printnanny_bb_dataframe_decoder")
            .build()?;

        let dataframe_agg = gst::ElementFactory::make("dataframe_agg")
            .name("dataframe_agg__df")
            .property("filter-threshold", nms_threshold as f32 / 100 as f32)
            .property_from_str("output-type", "json")
            .build()?;

        let nats_sink = gst::ElementFactory::make("nats_sink")
            .property("nats-address", &nats_server_uri)
            .build()?;

        let df_elements = &[
            &df_decoder_q,
            &dataframe_decoder,
            &dataframe_agg,
            &nats_sink,
        ];

        pipeline.add_many(box_overlay_elements)?;
        pipeline.add_many(df_elements)?;
        gst::Element::link_many(&[&tflite_output_tee, &box_decoder_q])?;
        gst::Element::link_many(box_overlay_elements)?;
        gst::Element::link_many(&[&tflite_output_tee, &df_decoder_q])?;
        gst::Element::link_many(df_elements)?;

        for e in tensor_pipeline_elements {
            e.sync_state_with_parent()?
        }

        for e in box_overlay_elements {
            e.sync_state_with_parent()?
        }

        for e in df_elements {
            e.sync_state_with_parent()?
        }

        Ok(pipeline)
    }

    fn make_device_pipeline(&self) -> Result<gst::Pipeline, Error> {
        let pipeline = self.make_common_pipeline()?;
        let videosrc = gst::ElementFactory::make("libcamerasrc").build()?;

        pipeline.add_many(&[&videosrc])?;

        let connect_element = pipeline
            .by_name("videoconvert__input")
            .expect("Element with name videoconvert__input not found");
        gst::Element::link_many(&[&videosrc, &connect_element])?;

        Ok(pipeline)
    }

    fn make_uri_pipeline(&self) -> Result<gst::Pipeline, Error> {
        let pipeline = self.make_common_pipeline()?;

        let uriencodebin = gst::ElementFactory::make("uridecodebin3")
            .property_from_str("caps", "video/x-raw")
            .property("use-buffering", true)
            .property("uri", &self.config.video_src)
            .build()?;

        pipeline.add_many(&[&uriencodebin])?;

        let pipeline_weak = pipeline.downgrade();

        uriencodebin.connect_pad_added(move |dbin, src_pad| {
            warn!("src_pad added {:?}", src_pad);
            let pipeline = match pipeline_weak.upgrade() {
                Some(p) => p,
                None => {
                    error!("Failed to upgrade pipeline reference");
                    return;
                }
            };

            let connect_element = pipeline
                .by_name("videoconvert__input")
                .expect("Element with name videoconvert__input not found");

            // We create a closure here, calling it directly below it, because this greatly
            // improves readability for error-handling. Like this, we can simply use the
            // ?-operator within the closure, and handle the actual error down below where
            // we call the insert_sink(..) closure.
            let insert_sink = || -> Result<(), Error> {
                // src pad for the video stream to it.
                let sink_pad = connect_element
                    .static_pad("sink")
                    .expect("tee has no sinkpad");
                src_pad.link(&sink_pad)?;
                Ok(())
            };

            // When adding and linking new elements in a callback fails, error information is often sparse.
            // GStreamer's built-in debugging can be hard to link back to the exact position within the code
            // that failed. Since callbacks are called from random threads within the pipeline, it can get hard
            // to get good error information. The macros used in the following can solve that. With the use
            // of those, one can send arbitrary rust types (using the pipeline's bus) into the mainloop.
            // What we send here is unpacked down below, in the iteration-code over sent bus-messages.
            // Because we are using the failure crate for error details here, we even get a backtrace for
            // where the error was constructed. (If RUST_BACKTRACE=1 is set)
            if let Err(err) = insert_sink() {
                // The following sends a message of type Error on the bus, containing our detailed
                // error information.
                element_error!(
                    dbin,
                    gst::LibraryError::Failed,
                    ("Failed to insert sink"),
                    details: gst::Structure::builder("error-details")
                                .field("error",
                                       &ErrorValue(Arc::new(Mutex::new(Some(err)))))
                                .build()
                );
            }
        });

        Ok(pipeline)
    }

    pub fn create_pipeline(&self) -> Result<gst::Pipeline, Error> {
        gst::init()?;

        let pipeline = match self.config.video_src_type {
            VideoSrcType::Uri => self.make_uri_pipeline()?,
            VideoSrcType::File => self.make_uri_pipeline()?,
            VideoSrcType::Device => self.make_device_pipeline()?,
        };

        Ok(pipeline)
    }
}

fn run(pipeline: gst::Pipeline) -> Result<()> {
    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    // This code iterates over all messages that are sent across our pipeline's bus.
    // In the callback ("pad-added" on the decodebin), we sent better error information
    // using a bus message. This is the position where we get those messages and log
    // the contained information.
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;

                match err.details() {
                    // This bus-message of type error contained our custom error-details struct
                    // that we sent in the pad-added callback above. So we unpack it and log
                    // the detailed error information here. details contains a glib::SendValue.
                    // The unpacked error is the converted to a Result::Err, stopping the
                    // application's execution.
                    Some(details) if details.name() == "error-details" => details
                        .get::<&ErrorValue>("error")
                        .unwrap()
                        .clone()
                        .0
                        .lock()
                        .unwrap()
                        .take()
                        .map(Result::Err)
                        .expect("error-details message without actual error"),
                    _ => Err(ErrorMessage {
                        src: msg
                            .src()
                            .map(|s| String::from(s.path_string()))
                            .unwrap_or_else(|| String::from("None")),
                        error: err.error().to_string(),
                        debug: err.debug(),
                        source: err.error(),
                    }
                    .into()),
                }?;
            }
            MessageView::StateChanged(s) => {
                let filename = format!("{}-{:?}-{:?}", pipeline.name(), &s.old(), &s.current());
                if s.src().map(|s| s == pipeline).unwrap_or(false) {
                    info!(
                        "State changed from {:?}: {:?} -> {:?} ({:?})",
                        s.src().map(|s| s.path_string()),
                        s.old(),
                        s.current(),
                        s.pending()
                    );
                    pipeline.debug_to_dot_file(gst::DebugGraphDetails::VERBOSE, &filename);
                    info!("Wrote {}", &filename);
                }
            }
            _ => (),
        }
    }

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}

impl From<&ArgMatches> for PipelineApp {
    fn from(args: &ArgMatches) -> Self {
        let config = PrintNannyGstPipelineConfig::from(args);
        Self { config }
    }
}

fn main() {
    let mut log_builder = Builder::new();

    let app_name = "printnanny-gst-pipeline";
    const GIT_VERSION: &str = git_version!();

    let cmd = Command::new(app_name)
        .author(crate_authors!())
        .about(crate_description!())
        // show git sha in --version
        .version(GIT_VERSION)
        // set level of verbosity
        .arg(
            Arg::new("v")
                .short('v')
                .multiple_occurrences(true)
                .help("Sets the level of verbosity. Info: -v Debug: -vv Trace: -vvv"),
        )
        .arg(
            Arg::new("config")
                .long("--config")
                .short('c')
                .takes_value(true)
                .conflicts_with_all(&[
                    "hls_http_enabled",
                    "label_file",
                    "model_file",
                    "nms_threshold",
                    "preview",
                    "tensor_batch_size",
                    "tensor_channels",
                    "tensor_height",
                    "tensor_width",
                    "overlay_udp_port",
                    "video_udp_port",
                    "video_framerate",
                    "video_height",
                    "video_src_type",
                    "video_src",
                    "video_width",
                    "nats_server_uri"
                ])
                .help("Read command-line args from config file. Config must be a valid PrintNannyConfig figment"),
        )
        .arg(
            Arg::new("preview")
                .long("--preview")
                .takes_value(false)
                .help("Show preview using autovideosink"),
        )
        .arg(
            Arg::new("nats_server_uri")
                .long("--nats-server-uri")
                .takes_value(true)
                .help("NATS server uri passed to nats_sink element"),
        )
        .arg(
            Arg::new("hls_http_enabled")
                .long("--hls-http-enabled")
                .takes_value(false)
                .help("Enable HLS HTTP server sink (required for compatibility with OctoPrint)"),
        )
        .arg(
            Arg::new("hls_segments")
                .long("--hls-segments")
                .takes_value(true)
                .default_value("/var/run/printnanny-hls/segment%05d.ts")
                .help("Location of hls segment files (passed to gstreamer hlssink2 location parameter)"),
        )
        .arg(
            Arg::new("hls_playlist")
                .long("--hls-playlist")
                .takes_value(true)
                .default_value("/var/run/printnanny-hls/playlist.m3u8")
                .help("Location of hls playlistfiles (passed to gstreamer hlssink2 playlist-location parameter)"),
        )
        .arg(
            Arg::new("hls_playlist_root")
                .long("--hls-playlist-root")
                .takes_value(true)
                .default_value("/printnanny-hls/")
                .help("HTTP serving directory prefix (configured via Nginx)"),
        )
        .arg(
            Arg::new("video_udp_port")
                .long("--video-udp-port")
                .takes_value(true)
                .default_value("20001")
                .help("Janus RTP stream port (UDP)"),
        )
        .arg(
            Arg::new("overlay_udp_port")
                .long("--overlay-udp-port")
                .takes_value(true)
                .default_value("20002")
                .help("Janus RTP stream port (UDP)"),
        )
        // --nms-threshold
        .arg(
            Arg::new("nms_threshold")
                .long("--nms-threshold")
                .takes_value(true)
                .default_value("50")
                .help("Non-max supression threshold"),
        )
        .arg(
            Arg::new("video_framerate")
                .long("video-framerate")
                .default_value("15")
                .takes_value(true)
                .help("Video framerate"),
        )
        .arg(
            Arg::new("video_src")
                .long("video-src")
                .takes_value(true)
                .help("Path to video file or camera device"),
        )
        .arg(
            Arg::new("video_height")
                .long("video-height")
                .default_value("480")
                .takes_value(true)
                .help("Height of input video file"),
        )
        .arg(
            Arg::new("video_width")
                .long("video-width")
                .default_value("640")
                .takes_value(true)
                .help("Width of input video file"),
        )
        // --video-stream-src
        .arg(
            Arg::new("video_src_type")
                .long("video-src-type")
                .value_parser(value_parser!(VideoSrcType))
                .takes_value(true),
        )
        // --tensor-batch-size
        .arg(
            Arg::new("tensor_batch_size")
                .long("tensor-batch-size")
                .takes_value(true)
                .default_value("1")
                .help("Batch size for tensor with shape: [Batch size, Height, Width, Channels]"),
        )
        // --tensor-height
        .arg(
            Arg::new("tensor_height")
                .long("tensor-height")
                .takes_value(true)
                .default_value("320")
                .help("Height value for tensor with shape: [Batch size, Height, Width, Channels]"),
        )
        .arg(
            Arg::new("tensor_width")
                .long("tensor-width")
                .takes_value(true)
                .default_value("320")
                .help("Width value for tensor with shape: [Batch size, Height, Width, Channels]"),
        )
        .arg(
            Arg::new("tensor_channels")
                .long("tensor-channels")
                .takes_value(true)
                .default_value("3")
                .help(
                    "Channels value for tensor with shape: [Batch size, Height, Width, Channels]",
                ),
        )
        .arg(
            Arg::new("model_file")
                .long("model-file")
                .takes_value(true)
                .help("Path to .tflite model file"),
        )
        .arg(
            Arg::new("label_file")
                .long("label-file")
                .takes_value(true)
                .help("Path to labels.txt file"),
        );
    let args = cmd.get_matches();
    // Vary the output based on how many times the user used the "verbose" flag
    // (i.e. 'printnanny -vvv' or 'printnanny -vv' vs 'printnanny -v'
    let verbosity = args.occurrences_of("v");
    match verbosity {
        0 => {
            log_builder.filter_level(LevelFilter::Warn).init();
        }
        1 => {
            log_builder.filter_level(LevelFilter::Info).init();
        }
        2 => {
            log_builder.filter_level(LevelFilter::Debug).init();
        }
        _ => {
            gst::debug_set_default_threshold(gst::DebugLevel::Trace);
            log_builder.filter_level(LevelFilter::Trace).init()
        }
    };

    let app = match args.value_of("config") {
        Some(config_file) => {
            let config = PrintNannyGstPipelineConfig::from_toml(PathBuf::from(config_file))
                .expect("Failed to extract config");
            info!("Pipeline config: {:?}", config);
            PipelineApp { config }
        }
        None => PipelineApp::from(&args),
    };

    match app.create_pipeline().and_then(run) {
        Ok(r) => r,
        Err(e) => error!("Error running pipeline: {:?}", e),
    }
}
