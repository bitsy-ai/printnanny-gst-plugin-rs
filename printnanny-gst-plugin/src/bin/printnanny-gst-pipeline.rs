// Build PrintNanny Gstreamer pipeline

use clap::{crate_authors, crate_description, value_parser, Arg, ArgMatches, Command};
use env_logger::Builder;
use gst::prelude::*;
use std::path::PathBuf;

use anyhow::{Context, Error, Result};
use git_version::git_version;
use log::{error, info, warn, LevelFilter};
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
                    "uridecodebin uri={video_src} caps=video/x-raw download=true ring-buffer-max-size=134217728 ! videoscale ! video/x-raw,width={video_width},height={video_height} ! tee name=decoded_video_t",
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

    pub fn create_pipeline(&self) -> Result<gst::Pipeline, Error> {
        gst::init()?;

        let decoded_video_src = self.decoded_video_src();

        let h264_video_sinks = self.h264_video_sinks()?;

        let vconverter = self.detect_vconverter();

        let h264_encoder = self.detect_h264_encoder();

        let pipeline_str = format!(
            "{decoded_video_src} \
            decoded_video_t. ! queue name=decoded_video_tensor_q leaky=2 \
                ! videoscale \
                ! videoconvert \
                ! capsfilter caps=video/x-raw,width={tensor_width},height={tensor_height},format=RGBA \
                ! tensor_converter \
                ! tensor_transform mode=arithmetic option=typecast:uint8,add:0,div:1 \
                ! capsfilter caps=other/tensors,num_tensors=1,format=static \
                ! tensor_filter framework=tensorflow2-lite model={model_file} \
                ! tensor_rate framerate={tensor_framerate}/1 throttle=false \
                ! tee name=tensor_t \
                ! queue name=tensor_decoder_q \
                ! tensor_decoder mode=bounding_boxes \
                    option1=mobilenet-ssd-postprocess \
                    option2={label_file} \
                    option3=0:1:2:3,{nms_threshold} \
                    option4={video_width}:{video_height} \
                    option5={tensor_width}:{tensor_height} \
                ! {vconverter} ! {h264_encoder} \
                ! udpsink port={overlay_udp_port} \
            decoded_video_t. ! queue name=h264queue ! {h264_encoder} ! tee name=h264_video_t \
            tensor_t. ! queue name=custom_tensor_decoder_t ! tensor_decoder mode=custom-code option1=printnanny_bb_dataframe_decoder \
            ! dataframe_agg filter-threshold=0.5 output-type=json \
            ! nats_sink nats-address={nats_server_uri} {h264_video_sinks}",
            tensor_height = &self.config.tflite_model.tensor_height,
            tensor_width = &self.config.tflite_model.tensor_width,
            model_file = &self.config.tflite_model.model_file,
            label_file = &self.config.tflite_model.label_file,
            nms_threshold = &self.config.tflite_model.nms_threshold,
            video_width = &self.config.video_width,
            video_height = &self.config.video_height,
            decoded_video_src = decoded_video_src,
            h264_video_sinks = h264_video_sinks,
            nats_server_uri = &self.config.nats_server_uri,
            vconverter = vconverter,
            h264_encoder = h264_encoder,
            tensor_framerate = &self.config.tflite_model.tensor_framerate,
            overlay_udp_port = &self.config.overlay_udp_port
        );

        warn!("Building pipeline: {}", pipeline_str);

        let pipeline = gst::parse_launch(&pipeline_str)?;
        let pipeline = pipeline.dynamic_cast::<gst::Pipeline>().unwrap();

        Ok(pipeline)
    }
}

fn run(pipeline: gst::Pipeline) -> Result<(), Error> {
    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;
                gst::error!(CAT, "demo-video pipeline failed with error: {:?}", err);
            }
            MessageView::StateChanged(state_changed) => {
                gst::info!(
                    CAT,
                    "Setting pipeline {:?} state to {:?}",
                    pipeline,
                    &state_changed
                );
                if state_changed.src().map(|s| s == pipeline).unwrap_or(false) {
                    pipeline.debug_to_dot_file(
                        gst::DebugGraphDetails::VERBOSE,
                        format!(
                            "{}-{:?}-{:?}",
                            pipeline.name(),
                            &state_changed.old(),
                            &state_changed.current()
                        ),
                    );
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
