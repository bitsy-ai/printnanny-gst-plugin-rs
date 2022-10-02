use gst::glib;
use gst::prelude::*;

mod imp;

// This enum may be used to control what type of output printanny_qcbin produces
// #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Copy, glib::Enum)]
// #[repr(u32)]
// #[enum_type(name = "GstDataframeFileSink")]
// pub enum DataframeFileSink {
//     #[enum_value(
//         name = "Println: Outputs the quality control report using println! macro",
//         nick = "println"
//     )]
//     Println = 0,
//     #[enum_value(
//         name = "Debug Category: Outputs the quality control report using the element's debug category",
//         nick = "debug-category"
//     )]
//     DebugCategory = 1,
// }

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct DataframeFileSink(ObjectSubclass<imp::DataframeFileSink>) @extends gst::Bin, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "dataframe_filesink",
        gst::Rank::None,
        DataframeFileSink::static_type(),
    )
}
