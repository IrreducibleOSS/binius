// Copyright 2025 Irreducible Inc.

#[macro_export]
macro_rules! each_tower_subfield {
    (
        $tower_height:expr,
        $func:ident ::< _ $(, $type_args:ty)* >( $($args:expr),* $(,)? )
    ) => {
        match $tower_height {
            0 => $func::< ::binius_math::B1 $(, $type_args,)* >( $($args),* ),
            3 => $func::< ::binius_math::B8 $(, $type_args,)* >( $($args),* ),
            4 => $func::< ::binius_math::B16 $(, $type_args,)* >( $($args),* ),
            5 => $func::< ::binius_math::B32 $(, $type_args,)* >( $($args),* ),
            6 => $func::< ::binius_math::B64 $(, $type_args,)* >( $($args),* ),
            7 => $func::< ::binius_math::B128 $(, $type_args,)* >( $($args),* ),

            _ => {
                return Err(
                    $crate::layer::Error::InputValidation(
                        format!(
                            "unsupported value of {}: {}",
                            stringify!($tower_height),
                            $tower_height
                        )
                    )
                )
            }
        }
    };
}

pub(super) use each_tower_subfield;

#[macro_export]
macro_rules! each_generic_tower_subfield {
    (
        $tower_height:expr,
        $tower_ty:ident,
        $func:ident ::< _ $(, $type_args:ty)* >( $($args:expr),* $(,)? )
    ) => {
        match $tower_height {
            0 => $func::< $tower_ty::B1 $(, $type_args,)* >( $($args),* ),
            3 => $func::< $tower_ty::B8 $(, $type_args,)* >( $($args),* ),
            4 => $func::< $tower_ty::B16 $(, $type_args,)* >( $($args),* ),
            5 => $func::< $tower_ty::B32 $(, $type_args,)* >( $($args),* ),
            6 => $func::< $tower_ty::B64 $(, $type_args,)* >( $($args),* ),
            7 => $func::< $tower_ty::B128 $(, $type_args,)* >( $($args),* ),

            _ => {
                return Err(
                    $crate::layer::Error::InputValidation(
                        format!(
                            "unsupported value of {}: {}",
                            stringify!($tower_height),
                            $tower_height
                        )
                    )
                )
            }
        }
    };
}
