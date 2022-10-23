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
    fn decoded_video_src(&self) -> String {
        let vconverter = self.detect_vconverter();
        let video_height = &self.config.video_height;
        let video_width = &self.config.video_width;

        match self.config.video_src_type {
            VideoSrcType::File => format!(
                "filesrc location={video_src} do-timestamp=true ! qtdemux name=demux demux.video_0 ! decodebin ! tee name=decoded_video_t",
                video_src = self.config.video_src,
            ),
            VideoSrcType::Device => format!(
                "libcamerasrc ! video/x-raw,framerate={video_framerate}/1,width={video_width},height={video_height} ! {vconverter} ! tee name=decoded_video_t",
                video_framerate = &self.config.video_framerate
            ),
            VideoSrcType::Uri => {
                format!(
                    "uridecodebin uri={video_src} caps=video/x-raw download=true ring-buffer-max-size=134217728 ! videoscale name=videoscale_input ! video/x-raw,width={video_width},height={video_height} ! tee name=decoded_video_t",
                    video_src = self.config.video_src,
                )
            }
        }
    }

    fn h264_video_sinks(&self) -> Result<String> {
        let hls_http_enabled = match self.config.hls_http_enabled {
            Some(value) => value,
            None => self.config.detect_hls_http_enabled()?,
        };
        let result = match hls_http_enabled {
            true => {
                format!(
                    "h264_video_t. ! queue name=h264_video_janus_q ! h264parse ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 ! udpsink port={udp_port} \
                    h264_video_t. ! queue name=h264_video_hls_q ! hlssink2 location={hls_segment} playlist-location={hls_playlist} playlist-root={hls_playlist_root}",
                    udp_port = &self.config.video_udp_port,
                    hls_segment = &self.config.hls_segments,
                    hls_playlist = &self.config.hls_playlist,
                    hls_playlist_root = &self.config.hls_playlist_root
                )
            }
            false => {
                format!(
                    "h264_video_t. ! queue name=h264_video_janus_q ! h264parse ! rtph264pay config-interval=1 aggregate-mode=zero-latency pt=96 ! udpsink port={udp_port}",
                    udp_port = &self.config.video_udp_port,
                )
            }
        };

        Ok(result)
    }

    // detect hardware-accelerated video encoder available via video4linux
    fn detect_h264_encoder(&self) -> String {
        match gst::ElementFactory::make_with_name("v4l2h264enc", None) {
            Ok(_) => {
                info!("Detected v4l2 support, using v4l2h264enc element");
                "v4l2h264enc extra-controls=\"controls,repeat_sequence_header=1\" ! capsfilter caps=video/x-h264,level=3 ".into()
            }
            Err(_) => {
                info!(
                    "Error making v4l2h264enc element, falling back to software-based x264enc element"
                );
                "x264enc ! capsfilter caps=video/x-h264,level=3".into()
            }
        }
    }

    // detect hardware-accelerated video converter available via video4linux
    fn detect_vconverter(&self) -> String {
        match gst::ElementFactory::make_with_name("v4l2convert", None) {
            Ok(_) => {
                info!("Detected v4l2 support, using v4l2convert element");
                "v4l2convert".into()
            }
            Err(_) => {
                info!("Error making v4l2convert element, falling back to software-based videoconvert element");
                "videoconvert".into()
            }
        }
    }

    fn make_uri_pipeline(&self) -> Result<gst::Pipeline, Error> {
        let start = SystemTime::now();
        let ts = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let pipeline_name = format!("pipeline-{:?}", &ts);

        let pipeline = gst::Pipeline::new(Some(&pipeline_name));
        let uriencodebin = gst::ElementFactory::make("uridecodebin3")
            .property_from_str("caps", "video/x-raw")
            .property("use-buffering", true)
            // .property("async-handling", true)
            .property("uri", &self.config.video_src)
            .build()?;

        pipeline.add_many(&[&uriencodebin])?;

        let pipeline_weak = pipeline.downgrade();

        let video_udp_port = self.config.video_udp_port.clone();

        uriencodebin.connect_pad_added(move |dbin, src_pad| {
            warn!("src_pad added {:?}", src_pad);
            let pipeline = match pipeline_weak.upgrade() {
                Some(p) => p,
                None => {
                    error!("Failed to upgrade pipeline reference");
                    return;
                }
            };

            // We create a closure here, calling it directly below it, because this greatly
            // improves readability for error-handling. Like this, we can simply use the
            // ?-operator within the closure, and handle the actual error down below where
            // we call the insert_sink(..) closure.
            let insert_sink = || -> Result<(), Error> {
                // decodebin found a raw videostream, so we build the follow-up pipeline to encode h264 video, rtp payload, and sink to janus gateway rtp ports
                let queue = gst::ElementFactory::make("queue")
                    .name("uridecodebin_after_q")
                    .build()?;
                let converter = gst::ElementFactory::make("videoconvert").build()?;
                let scaler = gst::ElementFactory::make("videoscale").build()?;

                let encoder = match gst::ElementFactory::make("v4l2h264enc")
                    .property_from_str("extra-controls", "controls,repeat_sequence_header=1")
                    .build()
                {
                    Ok(el) => el,
                    Err(_) => {
                        warn!("v4l2h264enc not found, falling back to x264enc");
                        gst::ElementFactory::make("x264enc").build()?
                    }
                };

                let parser = gst::ElementFactory::make("h264parse").build()?;

                let payloader = gst::ElementFactory::make("rtph264pay")
                    .property("config-interval", 1)
                    .property_from_str("aggregate-mode", "zero-latency")
                    .property_from_str("pt", "96")
                    .build()?;

                let sink = gst::ElementFactory::make("udpsink")
                    .property("port", video_udp_port)
                    .build()?;

                let elements = &[
                    &queue, &converter, &scaler, &encoder, &parser, &payloader, &sink,
                ];
                pipeline.add_many(elements)?;
                gst::Element::link_many(elements)?;

                for e in elements {
                    e.sync_state_with_parent()?
                }

                // Get the queue element's sink pad and link the decodebin's newly created
                // src pad for the video stream to it.
                let sink_pad = queue.static_pad("sink").expect("queue has no sinkpad");
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

        let pipeline = self.make_uri_pipeline()?;

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
            PipelineApp { config }
        }
        None => PipelineApp::from(&args),
    };

    match app.create_pipeline().and_then(run) {
        Ok(r) => r,
        Err(e) => error!("Error running pipeline: {:?}", e),
    }
}
