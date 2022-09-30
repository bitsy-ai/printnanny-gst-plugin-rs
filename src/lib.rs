use gst::glib;
mod arrow_decoder;
mod dataframe_decoder;
pub mod nnstreamer;
pub mod tensor;

#[allow(non_snake_case)]
#[path = "../target/flatbuffers/nnstreamer_generated.rs"]
pub mod message_nnstreamer;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    arrow_decoder::register(&plugin)?;
    dataframe_decoder::register(plugin)?;
    nnstreamer::register_nnstreamer_callbacks();
    Ok(())
}

gst::plugin_define!(
    printnanny,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "GPL",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
