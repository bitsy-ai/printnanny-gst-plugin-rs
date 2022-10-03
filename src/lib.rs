use gst::glib;
mod dataframe_filesink;
pub mod nnstreamer;
pub mod tensor;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    dataframe_filesink::register(plugin)?;
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
