extern crate gleam;

mod emscripten;
mod matrix;

use emscripten::{
    emscripten_GetProcAddress, emscripten_get_element_css_size, emscripten_set_main_loop_arg,
    emscripten_webgl_create_context, emscripten_webgl_init_context_attributes,
    emscripten_webgl_make_context_current, EmscriptenWebGLContextAttributes,
};
use gleam::gl;
use gleam::gl::{GLenum, GLuint};
use matrix::{matmul, perspective_matrix, rotate_x, rotate_y, viewing_matrix, Matrix44};

type GlPtr = std::rc::Rc<gl::Gl>;

#[repr(C)]
struct Context {
    gl: GlPtr,
    program: GLuint,
    buffer: GLuint,
    theta: f32,
    camera: Matrix44,
    p_matrix: Matrix44,
    width: u32,
    height: u32,
}

fn load_shader(gl: &GlPtr, shader_type: GLenum, source: &[&[u8]]) -> Option<GLuint> {
    let shader = gl.create_shader(shader_type);
    if shader == 0 {
        return None;
    }
    gl.shader_source(shader, source);
    gl.compile_shader(shader);
    let mut compiled = [0];
    unsafe {
        gl.get_shader_iv(shader, gl::COMPILE_STATUS, &mut compiled);
    }
    if compiled[0] == 0 {
        let log = gl.get_shader_info_log(shader);
        println!("{}", log);
        gl.delete_shader(shader);
        return None;
    }
    Some(shader)
}

fn init_buffer(gl: &GlPtr, program: GLuint) -> Option<GLuint> {
    let vertices: Vec<f32> = vec![
        -50.0, -50.0, -50.0, 0.0, 0.0, 0.0, 50.0, -50.0, -50.0, 1.0, 0.0, 0.0, 50.0, 50.0, -50.0,
        1.0, 1.0, 0.0, -50.0, 50.0, -50.0, 0.0, 1.0, 0.0, -50.0, -50.0, 50.0, 0.0, 0.0, 1.0, 50.0,
        -50.0, 50.0, 1.0, 0.0, 1.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, -50.0, 50.0, 50.0, 0.0, 1.0,
        1.0,
    ];
    let elements: Vec<u16> = vec![
        3, 2, 0, 2, 0, 1, 0, 1, 4, 1, 4, 5, 1, 2, 5, 2, 5, 6, 2, 3, 6, 3, 6, 7, 3, 0, 7, 0, 7, 4,
        4, 5, 7, 5, 7, 6,
    ];
    let buffers = gl.gen_buffers(2);
    let vertex_buffer = buffers[0];
    let element_buffer = buffers[1];
    let position_location = gl.get_attrib_location(program, "aPosition") as u32;
    let color_location = gl.get_attrib_location(program, "aColor") as u32;
    let array = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(array);
    gl.enable_vertex_attrib_array(position_location);
    gl.enable_vertex_attrib_array(color_location);
    gl.bind_buffer(gl::ARRAY_BUFFER, vertex_buffer);
    gl.buffer_data_untyped(
        gl::ARRAY_BUFFER,
        4 * vertices.len() as isize,
        vertices.as_ptr() as *const _,
        gl::STATIC_DRAW,
    );
    gl.vertex_attrib_pointer(position_location, 3, gl::FLOAT, false, 24, 0);
    gl.vertex_attrib_pointer(color_location, 3, gl::FLOAT, false, 24, 12);
    gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, element_buffer);
    gl.buffer_data_untyped(
        gl::ELEMENT_ARRAY_BUFFER,
        2 * elements.len() as isize,
        elements.as_ptr() as *const _,
        gl::STATIC_DRAW,
    );
    gl.bind_vertex_array(0);
    Some(array)
}

impl Context {
    fn new(gl: GlPtr) -> Context {
        let v_shader = load_shader(&gl, gl::VERTEX_SHADER, VS_SRC).unwrap();
        let f_shader = load_shader(&gl, gl::FRAGMENT_SHADER, FS_SRC).unwrap();
        let program = gl.create_program();
        gl.attach_shader(program, v_shader);
        gl.attach_shader(program, f_shader);
        gl.link_program(program);
        gl.use_program(program);
        let position_location = gl.get_attrib_location(program, "aPosition") as u32;
        let color_location = gl.get_attrib_location(program, "aColor") as u32;
        gl.enable_vertex_attrib_array(position_location);
        gl.enable_vertex_attrib_array(color_location);
        let buffer = init_buffer(&gl, program).unwrap();
        gl.clear_color(0.0, 0.0, 0.0, 1.0);
        gl.enable(gl::DEPTH_TEST);
        let (width, height) = get_canvas_size();
        Context {
            gl: gl,
            program: program,
            buffer: buffer,
            theta: 0.0,
            camera: viewing_matrix([0.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
            p_matrix: perspective_matrix(
                (45.0 as f32).to_radians(),
                width as f32 / height as f32,
                0.001,
                1000.0,
            ),
            width: width,
            height: height,
        }
    }

    fn draw(&self) {
        let gl = &self.gl;
        gl.viewport(0, 0, self.width as i32, self.height as i32);
        gl.clear(gl::COLOR_BUFFER_BIT);
        gl.use_program(self.program);
        let mv_location = gl.get_uniform_location(self.program, "uMVMatrix");
        let mv_matrix = matmul(
            matmul(rotate_x(self.theta), rotate_y(self.theta)),
            self.camera,
        );
        gl.uniform_matrix_4fv(mv_location, false, &mv_matrix);
        let p_location = gl.get_uniform_location(self.program, "uPMatrix");
        gl.uniform_matrix_4fv(p_location, false, &self.p_matrix);
        gl.bind_vertex_array(self.buffer);
        gl.draw_elements(gl::TRIANGLES, 36, gl::UNSIGNED_SHORT, 0);
        gl.bind_vertex_array(0);
    }
}

fn get_canvas_size() -> (u32, u32) {
    unsafe {
        let mut width = std::mem::uninitialized();
        let mut height = std::mem::uninitialized();
        emscripten_get_element_css_size(std::ptr::null(), &mut width, &mut height);
        (width as u32, height as u32)
    }
}

fn step(ctx: &mut Context) {
    ctx.theta += 0.01;
    ctx.draw();
}

extern "C" fn loop_wrapper(ctx: *mut std::os::raw::c_void) {
    unsafe {
        let mut ctx = &mut *(ctx as *mut Context);
        step(&mut ctx);
    }
}

fn main() {
    unsafe {
        let mut attributes: EmscriptenWebGLContextAttributes = std::mem::uninitialized();
        emscripten_webgl_init_context_attributes(&mut attributes);
        attributes.majorVersion = 2;
        let handle = emscripten_webgl_create_context(std::ptr::null(), &attributes);
        emscripten_webgl_make_context_current(handle);
        let gl = gl::GlesFns::load_with(|addr| {
            let addr = std::ffi::CString::new(addr).unwrap();
            emscripten_GetProcAddress(addr.into_raw() as *const _) as *const _
        });
        let mut ctx = Context::new(gl);
        let ptr = &mut ctx as *mut _ as *mut std::os::raw::c_void;
        emscripten_set_main_loop_arg(Some(loop_wrapper), ptr, 0, 1);
    }
}

const VS_SRC: &'static [&[u8]] = &[b"#version 300 es
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
out vec4 vColor;
void main() {
    gl_Position = uPMatrix * uMVMatrix * vec4(aPosition, 1.0);
    vColor = vec4(aColor, 1.0);
}"];

const FS_SRC: &'static [&[u8]] = &[b"#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 oFragColor;
void main() {
    oFragColor = vColor;
}"];
