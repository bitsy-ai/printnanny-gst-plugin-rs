use thiserror::Error;

#[derive(Error, Debug)]
pub enum PrintNannyGstConfigError {
    #[error(transparent)]
    FigmentError(#[from] printnanny_services::figment::error::Error),
    #[error(transparent)]
    TomlSerError(#[from] toml::ser::Error),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}
