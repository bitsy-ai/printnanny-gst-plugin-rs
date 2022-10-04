use std::sync::{Arc, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;

use once_cell::sync::Lazy;
use polars::prelude::*;

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "dataframe_agg",
        gst::DebugColorFlags::empty(),
        Some("PrintNanny Dataframe aggregator"),
    )
});

const SIGNAL_NOZZLE_OK: &str = "nozzle-ok";
const SIGNAL_NOZZLE_UNKNOWN: &str = "nozzle-unknown";

const DEFAULT_MAX_SIZE_DURATION: u64 = 6e+10 as u64; // 1 minute (in nanoseconds)
const DEFAULT_MAX_SIZE_BUFFERS: u64 = 900; // approx 1 minute of buffer frames @ 15fps
const DEFAULT_WINDOW_INTERVAL: &str = "1s";
const DEFAULT_WINDOW_PERIOD: &str = "1s";
const DEFAULT_WINDOW_OFFSET: &str = "0s";
const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;
const DEFAULT_DDOF: i32 = 0; // delta degrees of freedom, used in std dev calculation. divisor = N - ddof, where N is the number of element in the set
const DEFAULT_WINDOW_TRUNCATE: bool = false;
const DEFAULT_WINDOW_INCLUDE_BOUNDARIES: bool = true;

struct State {
    dataframe: LazyFrame,
}

impl Default for State {
    fn default() -> Self {
        let x0: Vec<f32> = vec![];
        let y0: Vec<f32> = vec![];
        let y1: Vec<f32> = vec![];
        let x1: Vec<f32> = vec![];
        let classes: Vec<i32> = vec![];
        let scores: Vec<f32> = vec![];
        let ts: Vec<i64> = vec![];

        let dataframe = df!(
            "detection_boxes_x0" => x0,
            "detection_boxes_y0" => y0,
            "detection_boxes_x1" => x1,
            "detection_boxes_y1" =>y1,
            "detection_classes" => classes,
            "detection_scores" => scores,
            "ts" => ts
        )
        .expect("Failed to initialize dataframe")
        .lazy();
        Self { dataframe }
    }
}

struct Settings {
    max_size_duration: u64,
    max_size_buffers: u64,
    window_interval: String,
    window_period: String,
    window_offset: String,
    window_truncate: bool,
    window_include_boundaries: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_size_duration: DEFAULT_MAX_SIZE_DURATION,
            max_size_buffers: DEFAULT_MAX_SIZE_BUFFERS,
            window_interval: DEFAULT_WINDOW_INTERVAL.into(),
            window_period: DEFAULT_WINDOW_PERIOD.into(),
            window_offset: DEFAULT_WINDOW_OFFSET.into(),
            window_truncate: DEFAULT_WINDOW_TRUNCATE,
            window_include_boundaries: DEFAULT_WINDOW_INCLUDE_BOUNDARIES,
        }
    }
}

#[derive(Clone)]
pub struct DataframeAgg {
    settings: Arc<Mutex<Settings>>,
    state: Arc<Mutex<State>>,
    sinkpad: gst::Pad,
    srcpad: gst::Pad,
}

impl DataframeAgg {
    fn drain(&self) -> Result<(), gst::ErrorMessage> {
        Ok(())
    }
    fn sink_chain(
        &self,
        pad: &gst::Pad,
        element: &super::DataframeAgg,
        buffer: gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::log!(CAT, obj: pad, "Handling buffer {:?}", buffer);

        let mut state = self.state.lock().unwrap();

        let cursor = buffer.into_cursor_readable();

        let reader = IpcStreamReader::new(cursor);
        let df = reader
            .finish()
            .expect("Failed to deserialize Arrow IPC Stream")
            .lazy();

        state.dataframe = concat(vec![state.dataframe.clone(), df], true, true).map_err(|err| {
            gst::element_error!(
                element,
                gst::ResourceError::Read,
                ["Failed to merge dataframes: {}", err]
            );
            gst::FlowError::Error
        })?;
        let every = "1s";
        let period = "1s";
        let offset = "0s";
        let score_threshold = 0.5;
        let ddof = 0; // delta degrees of freedom, used for std deviation / variance calculations. divisor = N - ddof, where N is the number of elements in set.

        let group_options = DynamicGroupOptions {
            index_column: "ts".to_string(),
            every: Duration::parse(every),
            period: Duration::parse(period),
            offset: Duration::parse(offset),
            closed_window: ClosedWindow::Left,
            truncate: false,
            include_boundaries: true,
        };

        let windowed_df = state
            .dataframe
            .clone()
            .filter(col("detection_classes").eq(0))
            .filter(col("detection_scores").gt(score_threshold))
            .sort(
                "ts",
                SortOptions {
                    descending: false,
                    nulls_last: false,
                },
            )
            .groupby_dynamic([col("detection_classes")], group_options)
            .agg([
                col("detection_scores")
                    .filter(col("detection_classes").eq(0))
                    .count()
                    .alias("nozzle__count"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(0))
                    .mean()
                    .alias("nozzle__mean"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(0))
                    .std(ddof)
                    .alias("nozzle__std"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(1))
                    .count()
                    .alias("adhesion__count"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(1))
                    .mean()
                    .alias("adhesion__mean"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(1))
                    .std(ddof)
                    .alias("adhesion__std"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(2))
                    .count()
                    .alias("spaghetti__count"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(2))
                    .mean()
                    .alias("spaghetti__mean"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(2))
                    .std(ddof)
                    .alias("spaghetti__std"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(3))
                    .count()
                    .alias("print__count"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(3))
                    .mean()
                    .alias("print__mean"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(3))
                    .std(ddof)
                    .alias("print__std"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(4))
                    .count()
                    .alias("raft__count"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(4))
                    .mean()
                    .alias("raft__mean"),
                col("detection_scores")
                    .filter(col("detection_classes").eq(4))
                    .std(ddof)
                    .alias("raft__std"),
            ])
            .collect()
            .map_err(|err| {
                gst::element_error!(
                    element,
                    gst::ResourceError::Read,
                    ["Failed window/aggregate dataframes {}", err]
                );
                gst::FlowError::Error
            })?;

        Ok(gst::FlowSuccess::Ok)
    }
    fn sink_event(&self, pad: &gst::Pad, element: &super::DataframeAgg, event: gst::Event) -> bool {
        true
    }
}

#[glib::object_subclass]
impl ObjectSubclass for DataframeAgg {
    const NAME: &'static str = "DataframeAgg";
    type Type = super::DataframeAgg;
    type ParentType = gst::Element;

    // Called when a new instance is to be created. We need to return an instance
    // of our struct here and also get the class struct passed in case it's needed
    fn with_class(klass: &Self::Class) -> Self {
        let templ = klass.pad_template("src").unwrap();
        let srcpad = gst::Pad::builder_with_template(&templ, Some("src")).build();

        let templ = klass.pad_template("sink").unwrap();
        let sinkpad = gst::Pad::builder_with_template(&templ, Some("sink"))
            .chain_function(|pad, parent, buffer| {
                DataframeAgg::catch_panic_pad_function(
                    parent,
                    || Err(gst::FlowError::Error),
                    |parse, element| parse.sink_chain(pad, element, buffer),
                )
            })
            .event_function(|pad, parent, event| {
                DataframeAgg::catch_panic_pad_function(
                    parent,
                    || false,
                    |parse, element| parse.sink_event(pad, element, event),
                )
            })
            .build();
        // Return an instance of our struct
        Self {
            sinkpad,
            srcpad,
            state: Arc::new(Mutex::new(State::default())),
            settings: Arc::new(Mutex::new(Settings::default())),
        }
    }
}

impl ObjectImpl for DataframeAgg {
    fn constructed(&self, obj: &Self::Type) {
        self.parent_constructed(obj);
        obj.add_pad(&self.sinkpad).unwrap();
        obj.add_pad(&self.srcpad).unwrap();
    }
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecUInt64::builder("max-size-buffers")
                    .nick("Max Size Buffers")
                    .blurb("Max number of buffers to perform windowed aggregations over")
                    .default_value(DEFAULT_MAX_SIZE_BUFFERS)
                    .build(),
                glib::ParamSpecUInt64::builder("max-size-duration")
                    .nick("Max Size Buffers")
                    .blurb("Max buffer duration to perform windowed aggregations over")
                    .default_value(DEFAULT_MAX_SIZE_DURATION)
                    .build(),
                glib::ParamSpecString::builder("window-interval")
                    .nick("Window Interval")
                    .blurb("Interval between window occurrences")
                    .default_value(DEFAULT_WINDOW_INTERVAL)
                    .build(),
                glib::ParamSpecString::builder("window-period")
                    .nick("Window Period")
                    .blurb("Length/duration of window")
                    .default_value(DEFAULT_WINDOW_PERIOD)
                    .build(),
                glib::ParamSpecString::builder("window-offset")
                    .nick("Window Offset")
                    .blurb("Offset window calculation by this amount")
                    .default_value(DEFAULT_WINDOW_OFFSET)
                    .build(),
                glib::ParamSpecBoolean::builder("window-truncate")
                    .nick("Truncate window")
                    .blurb("Truncate window")
                    .default_value(DEFAULT_WINDOW_TRUNCATE)
                    .build(),
                glib::ParamSpecBoolean::builder("window-include-boundaries")
                    .nick("Window Include Boundaries")
                    .blurb("Include _lower_boundary and _upper_boundary columns in windowed dataframe projection")
                    .default_value(DEFAULT_WINDOW_INCLUDE_BOUNDARIES)
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
    }

    fn property(&self, _obj: &Self::Type, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.settings.lock().unwrap();
        match pspec.name() {
            "max-size-buffers" => settings.max_size_buffers.to_value(),
            "max-size-duration" => settings.max_size_duration.to_value(),
            _ => unimplemented!(),
        }
    }

    fn set_property(
        &self,
        _obj: &Self::Type,
        _id: usize,
        value: &glib::Value,
        pspec: &glib::ParamSpec,
    ) {
        let mut settings = self.settings.lock().unwrap();

        match pspec.name() {
            "max-size-buffers" => {
                settings.max_size_buffers = value.get::<u64>().expect("type checked upstream");
            }
            "max-size-duration" => {
                settings.max_size_duration = value.get::<u64>().expect("type checked upstream");
            }
            _ => unimplemented!(),
        }
    }
    // fn signals() -> &'static [glib::subclass::Signal] {
    //     static SIGNALS: Lazy<Vec<glib::subclass::Signal>> = Lazy::new(|| {
    //         vec![glib::subclass::Signal::builder("reset")
    //             .action()
    //             .class_handler(|_token, args| {
    //                 let this = args[0].get::<super::DataframeAgg>().unwrap();
    //                 let imp = this.imp();

    //                 gst::info!(CAT, obj: &this, "Resetting measurements",);
    //                 imp.reset.store(true, atomic::Ordering::SeqCst);

    //                 None
    //             })
    //             .build()]
    //     });

    //     &*SIGNALS
    // }
}

impl GstObjectImpl for DataframeAgg {}

impl ElementImpl for DataframeAgg {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "PrintNanny QC Dataframe aggregator",
                "Filter/Agg",
                "Aggregate windowed dataframes and calculate quality score",
                "Leigh Johnson <leigh@printnanny.ai>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::new_any();

            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            let caps = gst::Caps::new_any();
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}
