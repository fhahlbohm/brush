use std::collections::HashMap;

use super::{BurnBack, BurnRuntime, FloatTensor};

#[derive(Clone, Copy)]
pub enum DimBound {
    Exact(usize),
    Any,
    Matching(&'static str),
}

pub struct DimCheck<'a> {
    bound: HashMap<&'a str, usize>,
    device: Option<&'a <BurnRuntime as burn_jit::Runtime>::Device>,
}

impl<'a> DimCheck<'a> {
    pub fn new() -> Self {
        DimCheck {
            bound: HashMap::new(),
            device: None,
        }
    }

    pub fn check_dims<const D: usize>(
        mut self,
        tensor: &'a FloatTensor<BurnBack, D>,
        bounds: [DimBound; D],
    ) -> Self {
        let dims = tensor.shape.dims;

        match self.device {
            None => self.device = Some(&tensor.device),
            Some(d) => assert_eq!(d, &tensor.device),
        }

        for (cur_dim, bound) in dims.into_iter().zip(bounds) {
            match bound {
                DimBound::Exact(dim) => assert_eq!(cur_dim, dim),
                DimBound::Any => (),
                DimBound::Matching(id) => {
                    let dim = *self.bound.entry(id).or_insert(cur_dim);
                    assert_eq!(cur_dim, dim);
                }
            }
        }
        self
    }
}

impl From<usize> for DimBound {
    fn from(value: usize) -> Self {
        DimBound::Exact(value)
    }
}

impl From<u32> for DimBound {
    fn from(value: u32) -> Self {
        DimBound::Exact(value as usize)
    }
}

impl From<i32> for DimBound {
    fn from(value: i32) -> Self {
        DimBound::Exact(value as usize)
    }
}

impl From<&'static str> for DimBound {
    fn from(value: &'static str) -> Self {
        match value {
            "*" => DimBound::Any,
            _ => DimBound::Matching(value),
        }
    }
}