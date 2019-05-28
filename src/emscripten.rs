#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub type em_arg_callback_func = Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void)>;

pub type EMSCRIPTEN_WEBGL_CONTEXT_HANDLE = ::std::os::raw::c_int;

#[repr(C)]
#[derive(Debug, Copy)]
pub struct EmscriptenWebGLContextAttributes {
    pub alpha: ::std::os::raw::c_int,
    pub depth: ::std::os::raw::c_int,
    pub stencil: ::std::os::raw::c_int,
    pub antialias: ::std::os::raw::c_int,
    pub premultipliedAlpha: ::std::os::raw::c_int,
    pub preserveDrawingBuffer: ::std::os::raw::c_int,
    pub preferLowPowerToHighPerformance: ::std::os::raw::c_int,
    pub failIfMajorPerformanceCaveat: ::std::os::raw::c_int,
    pub majorVersion: ::std::os::raw::c_int,
    pub minorVersion: ::std::os::raw::c_int,
    pub enableExtensionsByDefault: ::std::os::raw::c_int,
    pub explicitSwapControl: ::std::os::raw::c_int,
}

impl Clone for EmscriptenWebGLContextAttributes {
    fn clone(&self) -> Self {
        *self
    }
}

extern "C" {
    pub fn emscripten_set_main_loop_arg(
        func: em_arg_callback_func,
        arg: *mut ::std::os::raw::c_void,
        fps: ::std::os::raw::c_int,
        simulate_infinite_loop: ::std::os::raw::c_int,
    );

    pub fn emscripten_GetProcAddress(
        name: *const ::std::os::raw::c_char,
    ) -> *const ::std::os::raw::c_void;

    pub fn emscripten_webgl_init_context_attributes(
        attributes: *mut EmscriptenWebGLContextAttributes,
    );

    pub fn emscripten_webgl_create_context(
        target: *const ::std::os::raw::c_char,
        attributes: *const EmscriptenWebGLContextAttributes,
    ) -> EMSCRIPTEN_WEBGL_CONTEXT_HANDLE;

    pub fn emscripten_webgl_make_context_current(
        context: EMSCRIPTEN_WEBGL_CONTEXT_HANDLE,
    ) -> ::std::os::raw::c_int;

    pub fn emscripten_get_element_css_size(
        target: *const ::std::os::raw::c_char,
        width: *mut f64,
        height: *mut f64,
    ) -> ::std::os::raw::c_int;
}
