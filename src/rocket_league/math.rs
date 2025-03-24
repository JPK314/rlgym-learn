use core::f32;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "rotation_to_quaternion")]
pub fn rotation_to_quaternion_py<'py>(
    rot: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let quat: Bound<'_, PyArray1<f32>> = unsafe { PyArray1::new(rot.py(), (4,), false) };
    rotation_to_quaternion(
        rot.readonly().as_slice()?.try_into()?,
        quat.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(quat)
}

#[pyfunction]
#[pyo3(name = "quaternion_to_rotation")]
pub fn quaternion_to_rotation_py<'py>(
    quat: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rot: Bound<'_, PyArray2<f32>> = unsafe { PyArray2::new(quat.py(), (3, 3), false) };
    quaternion_to_rotation(
        quat.readonly().as_slice()?.try_into()?,
        rot.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(rot)
}

#[pyfunction]
#[pyo3(name = "euler_to_rotation")]
pub fn euler_to_rotation_py<'py>(
    euler: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rot: Bound<'_, PyArray2<f32>> = unsafe { PyArray2::new(euler.py(), (3, 3), false) };
    euler_to_rotation(
        euler.readonly().as_slice()?.try_into()?,
        rot.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(rot)
}

#[pyfunction]
#[pyo3(name = "rotation_to_euler")]
pub fn rotation_to_euler_py<'py>(
    rot: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let euler: Bound<'_, PyArray1<f32>> = unsafe { PyArray1::new(rot.py(), (3,), false) };
    rotation_to_euler(
        rot.readonly().as_slice()?.try_into()?,
        euler.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(euler)
}

#[pyfunction]
#[pyo3(name = "quaternion_to_euler")]
pub fn quaternion_to_euler_py<'py>(
    quat: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let euler: Bound<'_, PyArray1<f32>> = unsafe { PyArray1::new(quat.py(), (3,), false) };
    quaternion_to_euler(
        quat.readonly().as_slice()?.try_into()?,
        euler.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(euler)
}

#[pyfunction]
#[pyo3(name = "euler_to_quaternion")]
pub fn euler_to_quaternion_py<'py>(
    euler: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rot: Bound<'_, PyArray2<f32>> = unsafe { PyArray2::new(euler.py(), (3, 3), false) };
    euler_to_quaternion(
        euler.readonly().as_slice()?.try_into()?,
        rot.readwrite().as_slice_mut()?.try_into()?,
    );
    Ok(rot)
}

pub fn rotation_to_quaternion(rot: &[f32; 9], quat: &mut [f32; 4]) {
    // rot is stored in row-major ordering, quat is stored as [w,x,y,z] = w + xi + yj + zk
    // Given the euler axis a = (a1, a2, a3) and rotation angle t, we want
    // w = cos(t/2), (x,y,z) = (a/||a||) * sin(t/2)
    // Logic taken from https://math.stackexchange.com/a/895033 which cites
    // https://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf
    // Note that the matrix in the link there is for RIGHT MULTIPLYING and so we want the transpose of it
    // Essentially, there are many ways of doing this but we want to pick one based on which
    // will be most numerically stable. The problematic operation is the square root and so we
    // pick the implementation which will have the largest square root for stability. Once we
    // have one of wxyz, we can calculate the other three without doing any other sqrt operations.
    // Here are our options:
    // w = 1/2 * sqrt(1 + a_{11} + a_{22} + a_{33}) => w^2 = 1/4 * (1 + a_{11} + a_{22} + a_{33})
    // x = 1/2 * sqrt(1 + a_{11} - a_{22} - a_{33}) => x^2 = 1/4 * (1 + a_{11} - a_{22} - a_{33})
    // y = 1/2 * sqrt(1 - a_{11} + a_{22} - a_{33}) => y^2 = 1/4 * (1 - a_{11} + a_{22} - a_{33})
    // z = 1/2 * sqrt(1 - a_{11} - a_{22} + a_{33}) => z^2 = 1/4 * (1 - a_{11} - a_{22} + a_{33})
    // Since x^2+y^2+z^2 = sin(t/2)^2, we have w^2+x^2+y^2+z^2=1 and so one of w^2,x^2,y^2,z^2
    // must be >= 1/4. Therefore we pick the first one we find which is >= 1/4
    let (w, x, y, z);
    // This is equivalent to checking w^2 >= 1/4
    let v0 = rot[0] + rot[4] + rot[8];
    if v0 >= 0.0 {
        w = (v0 + 1.0).sqrt() * 0.5;
        let coef = 0.25 / w;
        x = coef * (rot[7] - rot[5]);
        y = coef * (rot[2] - rot[6]);
        z = coef * (rot[3] - rot[1]);
    } else {
        // This is equivalent to checking x^2 >= 1/4
        let v1 = rot[0] - rot[4] - rot[8];
        if v1 >= 0.0 {
            x = (v1 + 1.0).sqrt() * 0.5;
            let coef = 0.25 / x;
            w = coef * (rot[7] - rot[5]);
            y = coef * (rot[3] + rot[1]);
            z = coef * (rot[6] + rot[2]);
        } else {
            // This is equivalent to checking y^2 >= 1/4
            let v2 = rot[4] - rot[0] - rot[8];
            if v2 >= 0.0 {
                y = (v2 + 1.0).sqrt() * 0.5;
                let coef = 0.25 / y;
                w = coef * (rot[2] - rot[6]);
                x = coef * (rot[1] + rot[3]);
                z = coef * (rot[7] + rot[5]);
            } else {
                // must be z^2 >= 1/4
                z = (rot[8] - rot[0] - rot[4] + 1.0).sqrt() * 0.5;
                let coef = 0.25 / z;
                w = coef * (rot[3] - rot[1]);
                x = coef * (rot[2] + rot[6]);
                y = coef * (rot[5] + rot[7]);
            }
        }
    }
    quat[0] = w;
    quat[1] = x;
    quat[2] = y;
    quat[3] = z;
}

pub fn quaternion_to_rotation(quat: &[f32; 4], rot: &mut [f32; 9]) {
    // Taken from https://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf p.462
    let [w2, x2, y2, z2] = quat.map(|v| v * v);
    let xy = quat[1] * quat[2];
    let wz = quat[0] * quat[3];
    let xz = quat[1] * quat[3];
    let wy = quat[0] * quat[2];
    let yz = quat[2] * quat[3];
    let wx = quat[0] * quat[1];
    rot[0] = w2 + x2 - y2 - z2;
    rot[1] = 2.0 * (xy - wz);
    rot[2] = 2.0 * (xz + wy);
    rot[3] = 2.0 * (xy + wz);
    rot[4] = w2 - x2 + y2 - z2;
    rot[5] = 2.0 * (yz - wx);
    rot[6] = 2.0 * (xz - wy);
    rot[7] = 2.0 * (yz + wx);
    rot[8] = w2 - x2 - y2 + z2;
}

pub fn euler_to_rotation(euler: &[f32; 3], rot: &mut [f32; 9]) {
    // Euler is stored as pitch, yaw, roll, but application order is yaw, pitch, roll
    let [sp, sy, sr] = euler.map(|v| v.sin());
    let [cp, cy, cr] = euler.map(|v| v.cos());
    // Row-major ordering
    // For derivation, see the conversation here: https://discord.com/channels/348658686962696195/535605770436345857/1351015237037592616 (link to RLBot discord)
    rot[0] = cp * cy;
    rot[1] = sp * sr * cy - cr * sy;
    rot[2] = -sp * cr * cy - sr * sy;
    rot[3] = cp * sy;
    rot[4] = sp * sr * sy + cr * cy;
    rot[5] = sr * cy - sp * cr * sy;
    rot[6] = sp;
    rot[7] = -cp * sr;
    rot[8] = cp * cr;
}

pub fn rotation_to_euler(rot: &[f32; 9], euler: &mut [f32; 3]) {
    // rot[6] = sp so p = asin(rot[6])
    // if |rot[6]| < 1:
    //    cp cancels out of -rot[7]/rot[8] = tan(r) and so atan2(-rot[7], rot[8]) = r,
    //    and similarly atan2(rot[3], rot[0]) = y
    // if rot[6] == 1:
    //    sp=1 so rot[2] = -cr * cy - sr * sy = -1/2(cos(r-y)+cos(r+y)) - 1/2(cos(r-y)-cos(r+y)) = -cos(r-y)
    //    and rot[5] = sr * cy - cr * sy = 1/2(sin(r+y)+sin(r-y)) - 1/2(sin(r+y)-sin(r-y)) = sin(r-y)
    //    so r-y = atan2(rot[5], -rot[2])
    //    We can just choose y=0 and r = atan2(rot[5], -rot[2])
    // if rot[6] == -1:
    //    sp=-1 so rot[2] = cr * cy - sr * sy = 1/2(cos(r-y)+cos(r+y)) - 1/2(cos(r-y)-cos(r+y)) = cos(r+y),
    //    and rot[5] = sr * cy + cr * sy = 1/2(sin(r+y)+sin(r-y)) + 1/2(sin(r+y)-sin(r-y)) = sin(r+y)
    //    so r+y = atan2(rot[5], rot[2]).
    //    we can just choose y=0 and r = atan2(rot[5], rot[2])
    let (p, y, r);
    if rot[6] < 1.0 {
        if rot[6] > -1.0 {
            p = rot[6].asin();
            y = rot[3].atan2(rot[0]);
            r = (-rot[7]).atan2(rot[8]);
        } else {
            p = -f32::consts::FRAC_PI_2;
            y = 0.0;
            r = rot[5].atan2(rot[2]);
        }
    } else {
        p = f32::consts::FRAC_PI_2;
        y = 0.0;
        r = rot[5].atan2(-rot[2]);
    }
    euler[0] = p;
    euler[1] = y;
    euler[2] = r;
}

pub fn quaternion_to_euler(quat: &[f32; 4], euler: &mut [f32; 3]) {
    // Simplification of quaternion -> rotation -> euler
    let (p, y, r);
    let xz = quat[1] * quat[3];
    let wy = quat[0] * quat[2];
    let rot6 = 2.0 * (xz - wy);
    if rot6 < 1.0 {
        if rot6 > -1.0 {
            let [w2, x2, y2, z2] = quat.map(|v| v * v);
            let xy = quat[1] * quat[2];
            let wz = quat[0] * quat[3];
            let yz = quat[2] * quat[3];
            let wx = quat[0] * quat[1];
            p = rot6.asin();
            y = (2.0 * (xy + wz)).atan2(w2 + x2 - y2 - z2);
            r = (-2.0 * (yz + wx)).atan2(w2 - x2 - y2 + z2);
        } else {
            let yz = quat[2] * quat[3];
            let wx = quat[0] * quat[1];
            p = -f32::consts::FRAC_PI_2;
            y = 0.0;
            r = (yz - wx).atan2(xz + wy);
        }
    } else {
        let yz = quat[2] * quat[3];
        let wx = quat[0] * quat[1];
        p = f32::consts::FRAC_PI_2;
        y = 0.0;
        r = (yz - wx).atan2(-(xz + wy));
    }
    euler[0] = p;
    euler[1] = y;
    euler[2] = r;
}

pub fn euler_to_quaternion(euler: &[f32; 3], quat: &mut [f32; 4]) {
    let (w, x, y, z);
    // direct logic for euler -> rotation -> quaternion
    let [sp, sy, sr] = euler.map(|v| v.sin());
    let [cp, cy, cr] = euler.map(|v| v.cos());
    let rot0 = cp * cy;
    let rot1 = sp * sr * cy - cr * sy;
    let rot2 = -sp * cr * cy - sr * sy;
    let rot3 = cp * sy;
    let rot4 = sp * sr * sy + cr * cy;
    let rot5 = sr * cy - sp * cr * sy;
    let rot6 = sp;
    let rot7 = -cp * sr;
    let rot8 = cp * cr;
    // This is equivalent to checking w^2 >= 1/4
    let v0 = rot0 + rot4 + rot8;
    if v0 >= 0.0 {
        w = (v0 + 1.0).sqrt() * 0.5;
        let coef = 0.25 / w;
        x = coef * (rot7 - rot5);
        y = coef * (rot2 - rot6);
        z = coef * (rot3 - rot1);
    } else {
        // This is equivalent to checking x^2 >= 1/4
        let v1 = rot0 - rot4 - rot8;
        if v1 >= 0.0 {
            x = (v1 + 1.0).sqrt() * 0.5;
            let coef = 0.25 / x;
            w = coef * (rot7 - rot5);
            y = coef * (rot1 + rot3);
            z = coef * (rot2 + rot6);
        } else {
            // This is equivalent to checking y^2 >= 1/4
            let v2 = rot4 - rot0 - rot8;
            if v2 >= 0.0 {
                y = (v2 + 1.0).sqrt() * 0.5;
                let coef = 0.25 / y;
                w = coef * (rot2 - rot6);
                x = coef * (rot3 + rot1);
                z = coef * (rot5 + rot7);
            } else {
                // must be z^2 >= 1/4
                z = (rot8 - rot0 - rot4 + 1.0).sqrt() * 0.5;
                let coef = 0.25 / z;
                w = coef * (rot3 - rot1);
                x = coef * (rot6 + rot2);
                y = coef * (rot7 + rot5);
            }
        }
    }
    quat[0] = w;
    quat[1] = x;
    quat[2] = y;
    quat[3] = z;
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrand;
    static TOL: f32 = 0.0001;
    fn generate_random_quaternion() -> Vec<f32> {
        // This is not uniformly random, but it works well enough for this
        let mut w = fastrand::f32();
        let mut x = fastrand::f32();
        let mut y = fastrand::f32();
        let mut z = fastrand::f32();
        let mut len_sq = w * w + x * x + y * y + z * z;
        while len_sq < 0.01 {
            w = fastrand::f32();
            x = fastrand::f32();
            y = fastrand::f32();
            z = fastrand::f32();
            len_sq = w * w + x * x + y * y + z * z;
        }
        let len = len_sq.sqrt();
        w = w / len;
        x = x / len;
        y = y / len;
        z = z / len;
        vec![w, x, y, z]
    }

    fn generate_random_euler() -> Vec<f32> {
        // This is also not uniformly random but it's good enough
        vec![
            fastrand::f32() * core::f32::consts::PI * 2.0,
            fastrand::f32() * core::f32::consts::PI * 2.0,
            fastrand::f32() * core::f32::consts::PI * 2.0,
        ]
    }

    fn generate_random_rotation() -> Vec<f32> {
        // This is actually uniformly random, see https://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
        // or more directly: https://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        let theta = fastrand::f32() * core::f32::consts::PI * 2.0;
        let phi = fastrand::f32() * core::f32::consts::PI * 2.0;
        let z = fastrand::f32() * 2.0;

        let r = z.sqrt();
        let (vx, vy, vz) = (phi.sin() * r, phi.cos() * r, (2.0 - z).sqrt());

        let st = theta.sin();
        let ct = theta.cos();
        let sx = vx * ct - vy * st;
        let sy = vx * st + vy * ct;

        vec![
            vx * sx - ct,
            vx * sy - st,
            vx * vz,
            vy * sx + st,
            vy * sy - ct,
            vy * vz,
            vz * sx,
            vz * sy,
            1.0 - z,
        ]
    }

    fn euler_sanity_check() {
        let euler_start = &generate_random_euler()[..].try_into().unwrap();
        let mut quat = vec![0_f32; 4];
        let mut rot = vec![0_f32; 9];
        let mut euler_quat_end = vec![0_f32; 3];
        let mut euler_rot_end = vec![0_f32; 3];
        euler_to_quaternion(euler_start, quat.as_mut_slice().try_into().unwrap());
        quaternion_to_euler(
            quat.as_slice().try_into().unwrap(),
            euler_quat_end.as_mut_slice().try_into().unwrap(),
        );
        euler_to_rotation(euler_start, rot.as_mut_slice().try_into().unwrap());
        rotation_to_euler(
            &rot[..].try_into().unwrap(),
            euler_rot_end.as_mut_slice().try_into().unwrap(),
        );
        // Euler angles are very difficult to compare directly, so we convert to something we haven't used yet
        let mut rot_euler_quat_end = vec![0_f32; 9];
        euler_to_rotation(
            euler_quat_end.as_slice().try_into().unwrap(),
            rot_euler_quat_end.as_mut_slice().try_into().unwrap(),
        );
        let quat_dist_sq = rot
            .iter()
            .zip(rot_euler_quat_end.iter())
            .map(|(r0, r1)| (r1 - r0) * (r1 - r0))
            .sum();
        assert!(
            TOL > quat_dist_sq,
            "Failure converting euler angles to and from quaternion starting with {:?} (ended with {:?}, distance squared: {})",
            euler_start, euler_quat_end, quat_dist_sq
        );
        let mut quat_euler_rot_end = vec![0_f32; 4];
        euler_to_quaternion(
            euler_rot_end.as_slice().try_into().unwrap(),
            quat_euler_rot_end.as_mut_slice().try_into().unwrap(),
        );
        let euler_dist_sq1: f32 = quat
            .iter()
            .zip(quat_euler_rot_end.iter())
            .map(|(q0, q1)| (q1 - q0) * (q1 - q0))
            .sum();
        let euler_dist_sq2 = quat
            .iter()
            .zip(quat_euler_rot_end.iter())
            .map(|(q0, q1)| (-q1 - q0) * (-q1 - q0))
            .sum();
        let euler_dist_sq = euler_dist_sq1.min(euler_dist_sq2);
        assert!(
            TOL > euler_dist_sq,
            "Failure converting quaternion to and from euler angles starting with {:?} (ended with {:?}, distance squared: {})",
            euler_start, euler_rot_end, euler_dist_sq
        );
    }

    fn rotation_sanity_check() {
        let rot_start = &generate_random_rotation()[..].try_into().unwrap();
        let mut quat = vec![0_f32; 4];
        let mut euler = vec![0_f32; 3];
        let mut rot_quat_end = vec![0_f32; 9];
        let mut rot_euler_end = vec![0_f32; 9];
        rotation_to_quaternion(rot_start, quat.as_mut_slice().try_into().unwrap());
        quaternion_to_rotation(
            quat.as_slice().try_into().unwrap(),
            rot_quat_end.as_mut_slice().try_into().unwrap(),
        );
        rotation_to_euler(rot_start, euler.as_mut_slice().try_into().unwrap());
        euler_to_rotation(
            euler.as_slice().try_into().unwrap(),
            rot_euler_end.as_mut_slice().try_into().unwrap(),
        );
        let quat_dist_sq = rot_start
            .iter()
            .zip(rot_quat_end.iter())
            .map(|(r0, r1)| (r1 - r0) * (r1 - r0))
            .sum();
        assert!(
            TOL > quat_dist_sq,
            "Failure converting rotation to and from quaternion starting with {:?} (ended with {:?}, distance squared: {})",
            rot_start, rot_quat_end, quat_dist_sq
        );
        let euler_dist_sq = rot_start
            .iter()
            .zip(rot_euler_end.iter())
            .map(|(r0, r1)| (r1 - r0) * (r1 - r0))
            .sum();
        assert!(
            TOL > euler_dist_sq,
            "Failure converting rotation to and from euler angles starting with {:?} (ended with {:?}, distance squared: {})",
            rot_start, rot_euler_end, euler_dist_sq
        );
    }

    fn quaternion_sanity_check() {
        let q_start = &generate_random_quaternion()[..].try_into().unwrap();
        let mut euler = vec![0_f32; 3];
        let mut rot = vec![0_f32; 9];
        let mut q_euler_end = vec![0_f32; 4];
        let mut q_rot_end = vec![0_f32; 4];
        quaternion_to_euler(q_start, euler.as_mut_slice().try_into().unwrap());
        euler_to_quaternion(
            &euler[..].try_into().unwrap(),
            q_euler_end.as_mut_slice().try_into().unwrap(),
        );
        quaternion_to_rotation(q_start, rot.as_mut_slice().try_into().unwrap());
        rotation_to_quaternion(
            &rot[..].try_into().unwrap(),
            q_rot_end.as_mut_slice().try_into().unwrap(),
        );
        let euler_dist_sq1: f32 = q_start
            .iter()
            .zip(q_euler_end.iter())
            .map(|(q0, q1)| (q1 - q0) * (q1 - q0))
            .sum();
        let euler_dist_sq2 = q_start
            .iter()
            .zip(q_euler_end.iter())
            .map(|(q0, q1)| (-q1 - q0) * (-q1 - q0))
            .sum();
        let euler_dist_sq = euler_dist_sq1.min(euler_dist_sq2);
        assert!(
            TOL > euler_dist_sq,
            "Failure converting quaternion to and from euler angles starting with {:?} (ended with {:?}, distance squared: {})",
            q_start, q_euler_end, euler_dist_sq
        );
        let rot_dist_sq1: f32 = q_start
            .iter()
            .zip(q_rot_end.iter())
            .map(|(q0, q1)| (q1 - q0) * (q1 - q0))
            .sum();
        let rot_dist_sq2 = q_start
            .iter()
            .zip(q_rot_end.iter())
            .map(|(q0, q1)| (-q1 - q0) * (-q1 - q0))
            .sum();
        let rot_dist_sq = rot_dist_sq1.min(rot_dist_sq2);
        assert!(
            TOL > rot_dist_sq,
            "Failure converting quaternion to and from rotation matrix starting with {:?} (ended with {:?}, distance squared: {})",
            q_start, q_rot_end, rot_dist_sq
        );
    }

    #[test]
    fn sanity_checks() {
        for _ in 0..1000 {
            quaternion_sanity_check();
            rotation_sanity_check();
            euler_sanity_check();
        }
    }
}
