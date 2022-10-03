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

const DEFAULT_MAX_DURATION: u64 = 6e+10 as u64; // 1 minute (in nanoseconds)
const DEFAULT_MAX_SIZE_BUFFERS: u64 = 900; // approx 1 minute of buffer frames @ 15fps

struct State {
    df: Option<DataFrame>,
}

impl Default for State {
    fn default() -> Self {
        Self { df: None }
    }
}

struct Settings {
    max_size_duration: u64,
    max_size_buffers: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_size_duration: DEFAULT_MAX_DURATION,
            max_size_buffers: DEFAULT_MAX_SIZE_BUFFERS,
        }
    }
}

#[derive(Clone)]
pub struct DataframeAgg {
    settings: Arc<Mutex<Settings>>,
    state: Arc<Mutex<State>>,
    srcpad: gst::Pad,
    sinkpad: gst::Pad,
}

impl DataframeAgg {
    fn drain(&self) -> Result<(), gst::ErrorMessage> {
        Ok(())
    }
    fn sink_chain(
        &self,
        pad: &gst::Pad,
        _element: &super::DataframeAgg,
        buffer: gst::Buffer,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        gst::log!(CAT, obj: pad, "Handling buffer {:?}", buffer);

        let mut state = self.state.lock().unwrap();

        // state.adapter.push(buffer);

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
                    .default_value(DEFAULT_MAX_SIZE_BUFFERS)
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

    fn constructed(&self, obj: &Self::Type) {
        self.parent_constructed(obj);
        obj.add_pad(&self.sinkpad).unwrap();
        obj.add_pad(&self.srcpad).unwrap();
    }
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
