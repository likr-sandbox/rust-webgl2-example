pub type Vec3 = [f32; 3];
pub type Matrix44 = [f32; 16];

pub fn zeros() -> Matrix44 {
    [0f32; 16]
}

pub fn identity() -> Matrix44 {
    let mut matrix = zeros();
    matrix[0] = 1.0;
    matrix[5] = 1.0;
    matrix[10] = 1.0;
    matrix[15] = 1.0;
    matrix
}

pub fn rotate_x(theta: f32) -> Matrix44 {
    let mut matrix = identity();
    matrix[5] = theta.cos();
    matrix[6] = theta.sin();
    matrix[9] = -theta.sin();
    matrix[10] = theta.cos();
    matrix
}

pub fn rotate_y(theta: f32) -> Matrix44 {
    let mut matrix = identity();
    matrix[0] = theta.cos();
    matrix[2] = theta.sin();
    matrix[8] = -theta.sin();
    matrix[10] = theta.cos();
    matrix
}

pub fn translate(x: f32, y: f32, z: f32) -> Matrix44 {
    let mut matrix = identity();
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix
}

pub fn cross(v1: Vec3, v2: Vec3) -> Vec3 {
    [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
}

pub fn normalize(v: Vec3) -> Vec3 {
    let sum = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / sum, v[1] / sum, v[2] / sum]
}

pub fn viewing_matrix(eye: Vec3, up: Vec3, target: Vec3) -> Matrix44 {
    let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    let r = cross(up, d);
    let f = cross(d, r);
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
    matmul(translate(-eye[0], -eye[1], -eye[2]), matrix)
}

pub fn orthogonal_matrix(
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    near: f32,
    far: f32,
) -> Matrix44 {
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

pub fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> Matrix44 {
    let mut matrix = zeros();
    matrix[0] = 1.0 / fov.tan() / aspect;
    matrix[5] = 1.0 / fov.tan();
    matrix[10] = -(far + near) / (far - near);
    matrix[11] = -1.0;
    matrix[14] = -2.0 * far * near / (far - near);
    matrix
}

pub fn matmul(a: Matrix44, b: Matrix44) -> Matrix44 {
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
