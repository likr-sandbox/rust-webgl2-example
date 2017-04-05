extern crate gleam;
extern crate emscripten_sys;

use gleam::gl;
use gleam::gl::{GLenum, GLuint};
use emscripten_sys::{
    emscripten_set_main_loop_arg,
    emscripten_GetProcAddress,
    emscripten_webgl_init_context_attributes,
    emscripten_webgl_create_context,
    emscripten_webgl_make_context_current,
    EmscriptenWebGLContextAttributes,
};

type GlPtr = std::rc::Rc<gl::Gl>;
type GlMatrix = [f32; 16];

#[repr(C)]
struct Context {
    gl: GlPtr,
    program: GLuint,
    buffer: GLuint,
    theta: f32,
    mv_matrix: GlMatrix,
    p_matrix: GlMatrix,
}

fn load_shader(gl: &GlPtr, shader_type: GLenum, source: &[&[u8]]) -> Option<GLuint> {
    let shader = gl.create_shader(shader_type);
    if shader == 0 {
        return None;
    }
    gl.shader_source(shader, source);
    gl.compile_shader(shader);
    let compiled = gl.get_shader_iv(shader, gl::COMPILE_STATUS);
    if compiled == 0 {
        let log = gl.get_shader_info_log(shader);
        println!("{}", log);
        gl.delete_shader(shader);
        return None;
    }
    Some(shader)
}

fn init_buffer(gl: &GlPtr, program: GLuint) -> Option<GLuint> {
    let vertices: Vec<f32> = vec![
        -50.0, -50.0, -50.0, 0.0, 0.0, 0.0,
        50.0, -50.0, -50.0, 1.0, 0.0, 0.0,
        50.0, 50.0, -50.0, 1.0, 1.0, 0.0,
        -50.0, 50.0, -50.0, 0.0, 1.0, 0.0,
        -50.0, -50.0, 50.0, 0.0, 0.0, 1.0,
        50.0, -50.0, 50.0, 1.0, 0.0, 1.0,
        50.0, 50.0, 50.0, 1.0, 1.0, 1.0,
        -50.0, 50.0, 50.0, 0.0, 1.0, 1.0,
    ];
    let elements: Vec<u16> = vec![
        3, 2, 0,
        2, 0, 1,
        0, 1, 4,
        1, 4, 5,
        1, 2, 5,
        2, 5, 6,
        2, 3, 6,
        3, 6, 7,
        3, 0, 7,
        0, 7, 4,
        4, 5, 7,
        5, 7, 6,
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
    gl.buffer_data_untyped(gl::ARRAY_BUFFER, 4 * vertices.len() as isize, vertices.as_ptr() as *const _, gl::STATIC_DRAW);
    gl.vertex_attrib_pointer(position_location, 3, gl::FLOAT, false, 24, 0);
    gl.vertex_attrib_pointer(color_location, 3, gl::FLOAT, false, 24, 12);
    gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, element_buffer);
    gl.buffer_data_untyped(gl::ELEMENT_ARRAY_BUFFER, 2 * elements.len() as isize, elements.as_ptr() as *const _, gl::STATIC_DRAW);
    gl.bind_vertex_array(0);
    Some(array)
}

fn zeros() -> GlMatrix {
    [0f32; 16]
}

fn identity() -> GlMatrix {
    let mut matrix = zeros();
    matrix[0] = 1.0;
    matrix[5] = 1.0;
    matrix[10] = 1.0;
    matrix[15] = 1.0;
    matrix
}

fn rotate_x(theta: f32) -> GlMatrix {
    let mut matrix = identity();
    matrix[5] = theta.cos();
    matrix[6] = theta.sin();
    matrix[9] = -theta.sin();
    matrix[10] = theta.cos();
    matrix
}

fn rotate_y(theta: f32) -> GlMatrix {
    let mut matrix = identity();
    matrix[0] = theta.cos();
    matrix[2] = theta.sin();
    matrix[8] = -theta.sin();
    matrix[10] = theta.cos();
    matrix
}

fn translate(x: f32, y: f32, z: f32) -> GlMatrix {
    let mut matrix = identity();
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix
}

fn cross(v1: [f32; 3], v2: [f32; 3]) -> [f32; 3] {
    [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
}

fn normalize(v1: [f32; 3]) -> [f32; 3] {
    let sum = v1[0] + v1[1] + v1[2];
    [
        v1[0] / sum,
        v1[1] / sum,
        v1[2] / sum,
    ]
}

fn viewing_matrix(eye: [f32; 3], up: [f32; 3], target: [f32; 3]) -> GlMatrix {
    let d = [
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ];
    let r = cross(d, up);
    let f = cross(r, d);
    let d = normalize(d);
    let r = normalize(r);
    let f = normalize(f);
    let mut matrix = identity();
    matrix[0] = r[0];
    matrix[4] = r[1];
    matrix[8] = r[2];
    matrix[1] = f[0];
    matrix[5] = f[1];
    matrix[9] = f[2];
    matrix[2] = -d[0];
    matrix[6] = -d[1];
    matrix[10] = -d[2];
    matmul(matrix, translate(-eye[0], -eye[1], -eye[2]))
}

fn orthogonal_matrix(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> GlMatrix {
    let mut matrix = zeros();
    let w = right - left;
    let x = right + left;
    let h = top - bottom;
    let y = top + bottom;
    let d = far - near;
    let z = far + near;
    matrix[0] = 2.0 / w;
    matrix[5] = 2.0 / h;
    matrix[10] = -1.0 / d;
    matrix[12] = -x / w;
    matrix[13] = -y / h;
    matrix[14] = -z / d;
    matrix[15] = 1.0;
    matrix
}

fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> GlMatrix {
    let mut matrix = zeros();
    matrix[0] = 1.0 / fov.tan() / aspect;
    matrix[5] = 1.0 / fov.tan();
    matrix[10] = -(far + near) / (far - near);
    matrix[11] = -1.0;
    matrix[14] = -2.0 * far * near / (far - near);
    matrix
}

fn matmul(a: GlMatrix, b: GlMatrix) -> GlMatrix {
    let mut c = zeros();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
    c
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
        Context {
            gl: gl,
            program: program,
            buffer: buffer,
            theta: 0.0,
            mv_matrix: identity(),
            p_matrix: perspective_matrix((45.0 as f32).to_radians(), 1.0, 0.001, 1000.0),
        }
    }

    fn draw(&self) {
        let gl = &self.gl;
        gl.viewport(0, 0, 500, 500);
        gl.clear(gl::COLOR_BUFFER_BIT);
        gl.use_program(self.program);
        let mv_location = gl.get_uniform_location(self.program, "uMVMatrix");
        gl.uniform_matrix_4fv(mv_location, false, &self.mv_matrix);
        let p_location = gl.get_uniform_location(self.program, "uPMatrix");
        gl.uniform_matrix_4fv(p_location, false, &self.p_matrix);
        gl.bind_vertex_array(self.buffer);
        gl.draw_elements(gl::TRIANGLES, 36, gl::UNSIGNED_SHORT, 0);
        gl.bind_vertex_array(0);
    }
}

fn step(ctx: &mut Context) {
    let camera = viewing_matrix([0.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
    ctx.theta += 0.01;
    ctx.mv_matrix = matmul(matmul(rotate_x(ctx.theta), rotate_y(ctx.theta)), camera);
    ctx.draw();
}

extern fn loop_wrapper(ctx: *mut std::os::raw::c_void) {
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
