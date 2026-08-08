use pyany_serde::{DynPyAnySerdeOption, PyAnySerde};
use pyo3::{intern, prelude::*};

pub struct Serdes {
    pub agent_id_serde: Box<dyn PyAnySerde>,
    pub obs_serde: Box<dyn PyAnySerde>,
    pub action_serde: Box<dyn PyAnySerde>,
    pub reward_serde: Box<dyn PyAnySerde>,
    pub obs_space_serde: Box<dyn PyAnySerde>,
    pub action_space_serde: Box<dyn PyAnySerde>,
    pub shared_info_serde_option: Option<Box<dyn PyAnySerde>>,
    pub shared_info_setter_serde_option: Option<Box<dyn PyAnySerde>>,
    pub state_serde_option: Option<Box<dyn PyAnySerde>>,
}

impl FromPyObject<'_, '_> for Serdes {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        Ok(Serdes {
            agent_id_serde: obj
                .getattr(intern!(py, "agent_id_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            obs_serde: obj
                .getattr(intern!(py, "obs_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            action_serde: obj
                .getattr(intern!(py, "action_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            reward_serde: obj
                .getattr(intern!(py, "reward_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            obs_space_serde: obj
                .getattr(intern!(py, "obs_space_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            action_space_serde: obj
                .getattr(intern!(py, "action_space_serde_type"))?
                .extract::<Box<dyn PyAnySerde>>()?,
            shared_info_serde_option: obj
                .getattr(intern!(py, "shared_info_serde_type"))?
                .extract::<DynPyAnySerdeOption>()?
                .into(),
            shared_info_setter_serde_option: obj
                .getattr(intern!(py, "shared_info_setter_serde_type"))?
                .extract::<DynPyAnySerdeOption>()?
                .into(),
            state_serde_option: obj
                .getattr(intern!(py, "state_serde_type"))?
                .extract::<DynPyAnySerdeOption>()?
                .into(),
        })
    }
}
