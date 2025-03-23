use pyo3::{sync::GILOnceCell, PyObject};

pub static INTERNED_ROCKET_LEAGUE_API_MODULE: GILOnceCell<PyObject> = GILOnceCell::new();

#[macro_export]
macro_rules! get_class {
    ($py: ident, $name: expr) => {{
        crate::rocket_league::api_module::INTERNED_ROCKET_LEAGUE_API_MODULE
            .get_or_try_init::<_, PyErr>($py, || {
                Ok($py.import("rlgym.rocket_league.api")?.into_any().unbind())
            })?
            .bind($py)
            .getattr(pyo3::intern!($py, $name))?
    }};
}
