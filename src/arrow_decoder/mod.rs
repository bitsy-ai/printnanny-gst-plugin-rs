use gst::glib;
use gst::prelude::*;

mod imp;

// The public Rust wrapper type for our element
glib::wrapper! {
    pub struct ArrowDecoder(ObjectSubclass<imp::ArrowDecoder>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

// the name "ArrowDecoder" for being able to instantiate it via e.g.
// gst::ElementFactory::make().
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "arrow_decoder",
        gst::Rank::None,
        ArrowDecoder::static_type(),
    )
}
