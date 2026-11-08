from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mechanism_parameters import (
    CONFIG,
    SEPARATION_FITS,
    TEMPERATURE_FITS,
    K_parallel,
    K_perp,
    temperature_ratio,
    mechanism_K6,
    mechanism_C6,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeometryConfig:
    num_rings: int = 1
    hex_radius: float = 0.50
    gap: float = 0.010
    body_z: float = 1.0
    # Total physical mechanisms per interface. Field-cooling-fixed uses two parallel rows.
    mechanisms_per_edge: int = 4
    mechanism_rows: int = 2
    row_separation_m: float = 0.050
    # Half-width of the along-edge mechanism-center footprint, normalized by
    # the hexagon circumradius (which also equals the edge length).  The Field-cooling-fixed
    # baseline uses +/-0.30R; dedicated sensitivity studies may vary it.
    mechanism_edge_half_span_fraction: float = 0.30


@dataclass
class DynamicsConfig:
    timestep: float = 2e-4
    run_time_s: float = 15.0

    force_clip: float = 200.0
    separation_min_m: float = 0.005
    separation_max_m: float = 0.030
    separation_clamp_count: int = 0
    # Transient geometry diagnostics. These values do not alter K or C.
    max_normal_gap_excursion_m: float = 0.0
    max_local_translation_m: float = 0.0
    min_instantaneous_gap_m: float = float("inf")
    max_instantaneous_gap_m: float = 0.0

    # Numerical anchor to remove the global six-DOF gauge mode.
    anchor_k_trans: float = 500.0
    anchor_zeta_trans: float = 0.01
    anchor_mass: float = 1.0
    anchor_k_rot: float = 500.0
    anchor_zeta_rot: float = 0.01
    anchor_inertia: float = 0.05229166666666667

    impulse_time: float = 1.5
    impulse_duration: float = 0.02
    impulse_force: Tuple[float, float, float] = (0.25, 0.0, 0.15)
    impulse_point_offset: Tuple[float, float, float] = (0.10, 0.15, 0.10)

    temperature_k: float = 77.0
    temperature_model: str = "chiang"
    zeta_nominal: float = 0.03


GEOM = GeometryConfig(
    hex_radius=CONFIG.segment.hex_circumradius_m,
    gap=CONFIG.nominal_gap_m,
)
DYN = DynamicsConfig(
    anchor_inertia=CONFIG.segment.inertia_x_kgm2,
    temperature_k=CONFIG.reference_temperature_k,
    temperature_model=CONFIG.temperature_model,
    zeta_nominal=CONFIG.zeta_nominal,
)


# =============================================================================
# Cached literature models
# =============================================================================


def cached_empirical_translational_stiffness(d_m: float) -> np.ndarray:
    """Return [K11, K22, K33] for compatibility with older validators."""
    d = float(d_m)
    return np.array([
        K_parallel(d),
        float(SEPARATION_FITS["K22"].stiffness(d)),
        K_perp(d),
    ], dtype=float)


def cached_temperature_ratio(temp_k: float, model_name: str = "chiang") -> float:
    return temperature_ratio(temp_k, model_name)


def empirical_translational_stiffness(d_m: float) -> np.ndarray:
    return cached_empirical_translational_stiffness(float(d_m))


@dataclass
class Mechanism:
    a_i: np.ndarray
    a_j: np.ndarray
    r0: np.ndarray
    T: np.ndarray


@dataclass
class RunResult:
    t: np.ndarray
    pos_err: np.ndarray
    rot_err: np.ndarray
    dof_err: np.ndarray
    force_mag: np.ndarray
    torque_mag: np.ndarray
    clip_count: int
    force_evaluations: int
    max_force: float
    max_torque: float
    # Shape (n_steps, n_segments, 6): [dx,dy,dz,rx,ry,rz].  Retaining the
    # segment histories lets every damping formulation use exactly the same
    # post-processing from the saved simulation history.
    segment_dofs: np.ndarray


# =============================================================================
# Geometry
# =============================================================================


def hex_positions(num_rings: int, hex_radius: float, gap: float) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    scale = (math.sqrt(3.0) * hex_radius + gap) / (math.sqrt(3.0) * hex_radius)
    for q in range(-num_rings, num_rings + 1):
        for r in range(-num_rings, num_rings + 1):
            if abs(q + r) <= num_rings:
                x = 1.5 * hex_radius * q
                y = math.sqrt(3.0) * hex_radius * (r + 0.5 * q)
                coords.append((x * scale, y * scale))
    return coords


def hex_vertices(hex_radius: float, thickness: float) -> str:
    """Return a hexagonal prism centered on the body's reference point."""
    R = hex_radius
    z0 = -0.5 * thickness
    z1 = 0.5 * thickness
    verts = [
        (R, 0.0, z0),
        (0.5 * R, math.sqrt(3.0) * 0.5 * R, z0),
        (-0.5 * R, math.sqrt(3.0) * 0.5 * R, z0),
        (-R, 0.0, z0),
        (-0.5 * R, -math.sqrt(3.0) * 0.5 * R, z0),
        (0.5 * R, -math.sqrt(3.0) * 0.5 * R, z0),
        (R, 0.0, z1),
        (0.5 * R, math.sqrt(3.0) * 0.5 * R, z1),
        (-0.5 * R, math.sqrt(3.0) * 0.5 * R, z1),
        (-R, 0.0, z1),
        (-0.5 * R, -math.sqrt(3.0) * 0.5 * R, z1),
        (0.5 * R, -math.sqrt(3.0) * 0.5 * R, z1),
    ]
    return "\n".join(f"{x} {y} {z}" for x, y, z in verts)


def create_xml(positions: List[Tuple[float, float]]) -> str:
    """Build a complete MuJoCo XML model with explicit segment inertial properties."""
    I = CONFIG.segment.inertia_diag_kgm2

    # Explicit triangular faces for a closed hexagonal prism.
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5),
        (6, 8, 7), (6, 9, 8), (6, 10, 9), (6, 11, 10),
        (0, 6, 7), (0, 7, 1),
        (1, 7, 8), (1, 8, 2),
        (2, 8, 9), (2, 9, 3),
        (3, 9, 10), (3, 10, 4),
        (4, 10, 11), (4, 11, 5),
        (5, 11, 6), (5, 6, 0),
    ]
    face_text = " ".join(f"{a} {b} {c}" for a, b, c in faces)

    xml = f'''
<mujoco model="version4_8_two_row_pure_translational">
  <option gravity="0 0 0" timestep="{DYN.timestep}" integrator="RK4"/>
  <asset>
    <mesh
        name="hex"
        vertex="
{hex_vertices(GEOM.hex_radius, CONFIG.segment.thickness_m)}
        "
        face="{face_text}"/>
    <material name="mirror_mat" rgba="0.4 0.6 0.9 0.92"/>
  </asset>
  <worldbody>
    <geom type="plane" size="8 8 0.1" rgba="0.1 0.1 0.1 1"/>
    <light pos="0 0 5" dir="0 0 -1"/>
'''

    body_parts = []
    for i, (x, y) in enumerate(positions):
        body_parts.append(
            f'''
    <body name="h{i}" pos="{x:.9f} {y:.9f} {GEOM.body_z:.9f}">
      <freejoint/>
      <inertial
          pos="0 0 0"
          mass="{CONFIG.segment.mass_kg:.12g}"
          diaginertia="{I[0]:.12g} {I[1]:.12g} {I[2]:.12g}"/>
      <geom
          type="mesh"
          mesh="hex"
          material="mirror_mat"
          contype="0"
          conaffinity="0"/>
    </body>
'''
        )

    return xml + "".join(body_parts) + "  </worldbody>\n</mujoco>\n"


# =============================================================================
# Rigid-body helpers
# =============================================================================


def body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_id: int):
    pos = data.xpos[body_id].copy()
    R = data.xmat[body_id].reshape(3, 3).copy()
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
    omega = vel6[:3].copy()
    lin = vel6[3:].copy()
    return pos, R, lin, omega


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    x = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(x)
    if angle < 1e-10:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
    s = math.sin(angle)
    if abs(s) < 1e-8:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        if axis[0] > 1e-8:
            axis[1] = math.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = math.copysign(axis[2], R[0, 2] + R[2, 0])
        elif axis[1] > 1e-8:
            axis[2] = math.copysign(axis[2], R[1, 2] + R[2, 1])
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        return axis * angle
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * s) * angle


def apply_point_force(model, data, body_id: int, point_world: np.ndarray, force_world: np.ndarray):
    mujoco.mj_applyFT(
        model,
        data,
        np.asarray(force_world, dtype=float),
        np.zeros(3),
        np.asarray(point_world, dtype=float),
        body_id,
        data.qfrc_applied,
    )


# =============================================================================
# Pure-translational mechanism
# =============================================================================


def make_edge_mechanisms(center_i, center_j, count: int) -> List[Mechanism]:
    """Create two parallel rows of flux-pinning mechanisms per interface.

    ``count`` is the total number of physical mechanisms on the interface and
    must be even. Half are placed in each row. The row separation is measured
    in the local out-of-plane b direction, following the Gersh-Range concept
    that multiple rows generate bending stiffness from translational springs.
    """
    if GEOM.mechanism_rows != 2:
        raise ValueError("Field-cooling-fixed two-row model requires mechanism_rows=2.")
    if count < 2 or count % 2 != 0:
        raise ValueError("Field-cooling-fixed two-row model requires an even total mechanism count >= 2.")
    if GEOM.row_separation_m <= 0.0:
        raise ValueError("row_separation_m must be positive.")
    if not (0.0 < GEOM.mechanism_edge_half_span_fraction <= 0.5):
        raise ValueError(
            "mechanism_edge_half_span_fraction must lie in (0, 0.5]."
        )

    ci = np.asarray(center_i, dtype=float)[:2]
    cj = np.asarray(center_j, dtype=float)[:2]
    dvec = cj - ci
    dist = float(np.linalg.norm(dvec))
    if dist <= 1e-12:
        raise ValueError("Coincident body centers.")

    n2 = dvec / dist
    t2 = np.array([-n2[1], n2[0]], dtype=float)
    b3 = np.array([0.0, 0.0, -1.0], dtype=float)
    n3 = np.array([n2[0], n2[1], 0.0], dtype=float)
    t3 = np.array([t2[0], t2[1], 0.0], dtype=float)
    T = np.column_stack((t3, n3, b3))

    edge_offset = math.sqrt(3.0) * 0.5 * GEOM.hex_radius
    per_row = count // 2

    # The baseline value preserves the total tangential footprint of the old
    # in-line geometry.  Keep it configurable so placement can be studied
    # independently of mechanism count.
    if per_row == 1:
        s_vals = np.array([0.0], dtype=float)
    else:
        half_span = GEOM.mechanism_edge_half_span_fraction * GEOM.hex_radius
        s_vals = np.linspace(
            -half_span,
            half_span,
            per_row,
        )

    row_offsets = (-0.5 * GEOM.row_separation_m,
                    0.5 * GEOM.row_separation_m)

    mechanisms: List[Mechanism] = []
    for b_off in row_offsets:
        for s in s_vals:
            # Local mechanism placement: t = tangential along edge,
            # n = across the interface, b = out of plane.
            ai2 = n2 * edge_offset + t2 * s
            aj2 = -n2 * edge_offset + t2 * s
            ai = np.array([ai2[0], ai2[1], b_off], dtype=float)
            aj = np.array([aj2[0], aj2[1], b_off], dtype=float)
            pi0 = np.array([ci[0], ci[1], GEOM.body_z]) + ai
            pj0 = np.array([cj[0], cj[1], GEOM.body_z]) + aj
            mechanisms.append(
                Mechanism(
                    a_i=ai,
                    a_j=aj,
                    r0=pi0 - pj0,
                    T=T.copy(),
                )
            )
    return mechanisms


def build_adjacency(positions):
    pos = np.asarray(positions, dtype=float)
    threshold = 1.15 * (math.sqrt(3.0) * GEOM.hex_radius + GEOM.gap)
    pairs = []
    mechanism_sets: Dict[Tuple[int, int], List[Mechanism]] = {}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(pos[j] - pos[i]) < threshold:
                pairs.append((i, j))
                mechanism_sets[(i, j)] = make_edge_mechanisms(
                    pos[i], pos[j], GEOM.mechanisms_per_edge
                )
    return pairs, mechanism_sets


# =============================================================================
# Anchor and mechanism force application
# =============================================================================


def anchor_parameters():
    c_trans = 2.0 * DYN.anchor_zeta_trans * math.sqrt(
        DYN.anchor_k_trans * DYN.anchor_mass
    )
    c_rot = 2.0 * DYN.anchor_zeta_rot * math.sqrt(
        DYN.anchor_k_rot * DYN.anchor_inertia
    )
    return c_trans, c_rot


def apply_anchor(model, data, body_id, p_ref, R_ref):
    p, R, v, omega = body_pose(model, data, body_id)
    c_trans, c_rot = anchor_parameters()
    f = -DYN.anchor_k_trans * (p - p_ref) - c_trans * v
    rot_err = rotation_matrix_to_rotvec(R @ R_ref.T)
    tau = -DYN.anchor_k_rot * rot_err - c_rot * omega
    apply_point_force(model, data, body_id, p, f)
    mujoco.mj_applyFT(
        model,
        data,
        np.zeros(3),
        tau,
        p,
        body_id,
        data.qfrc_applied,
    )


def mechanism_step_force(
    model,
    data,
    body_i: int,
    body_j: int,
    mech: Mechanism,
    pose_i,
    pose_j,
    K_local: np.ndarray,
    C_local: np.ndarray,
):
    pi0, Ri, vi_lin, wi = pose_i
    pj0, Rj, vj_lin, wj = pose_j

    pi = pi0 + Ri @ mech.a_i
    pj = pj0 + Rj @ mech.a_j
    vi = vi_lin + np.cross(wi, pi - pi0)
    vj = vj_lin + np.cross(wj, pj - pj0)

    dp_world = pi - pj - mech.r0
    dv_world = vi - vj

    # Relative orientation is retained for diagnostics, but the primary
    # mechanism constitutive law has no direct rotational stiffness/damping.
    Rrel = Ri @ Rj.T
    drot_world = rotation_matrix_to_rotvec(Rrel)

    e_t = mech.T.T @ dp_world
    v_t = mech.T.T @ dv_world
    e6 = np.concatenate((e_t, mech.T.T @ drot_world))
    v6 = np.concatenate((v_t, mech.T.T @ (wi - wj)))

    instantaneous_separation = float(abs(np.dot(pi - pj, mech.T[:, 1])))
    field_cooling_separation = float(GEOM.gap)

    # Field-cooling convention: the small-displacement constitutive
    # matrices are selected from the separation at which the pair was field
    # cooled and remain fixed throughout the transient.  The instantaneous
    # gap is tracked only as a clearance/small-motion diagnostic.
    DYN.max_normal_gap_excursion_m = max(
        DYN.max_normal_gap_excursion_m,
        abs(instantaneous_separation - field_cooling_separation),
    )
    DYN.max_local_translation_m = max(
        DYN.max_local_translation_m,
        float(np.linalg.norm(e_t)),
    )
    DYN.min_instantaneous_gap_m = min(
        DYN.min_instantaneous_gap_m, instantaneous_separation
    )
    DYN.max_instantaneous_gap_m = max(
        DYN.max_instantaneous_gap_m, instantaneous_separation
    )
    if not (DYN.separation_min_m <= instantaneous_separation <= DYN.separation_max_m):
        DYN.separation_clamp_count += 1

    w_local = -(K_local @ e6 + C_local @ v6)

    F_world = mech.T @ w_local[:3]
    M_world = mech.T @ w_local[3:]

    fmag = float(np.linalg.norm(F_world))
    clipped = 0
    if fmag > DYN.force_clip:
        F_world *= DYN.force_clip / max(fmag, 1e-12)
        clipped = 1

    apply_point_force(model, data, body_i, pi, F_world)
    apply_point_force(model, data, body_j, pj, -F_world)

    # M_world is zero for the pure-translational constitutive model. The torque
    # on each segment still arises physically from the force moment arm.
    if np.linalg.norm(M_world) > 0.0:
        mujoco.mj_applyFT(model, data, np.zeros(3), M_world, pi, body_i, data.qfrc_applied)
        mujoco.mj_applyFT(model, data, np.zeros(3), -M_world, pj, body_j, data.qfrc_applied)

    tau_i = np.cross(pi - pi0, F_world) + M_world
    torque_mag = float(np.linalg.norm(tau_i))
    wrench_mag = float(np.linalg.norm(np.concatenate((F_world, M_world))))
    return fmag, torque_mag, clipped, fmag, wrench_mag


# =============================================================================
# Dynamic simulation
# =============================================================================


def run_case(
    temperature_k: Optional[float] = None,
    temperature_model: Optional[str] = None,
    show_viewer: bool = False,
    timestep: Optional[float] = None,
    impulse_times: Optional[List[float]] = None,
    impulse_force_scale: float = 1.0,
    stop_callback=None,
    log_stride: int = 1,
) -> RunResult:
    old_timestep = DYN.timestep
    old_temperature_k = DYN.temperature_k
    old_temperature_model = DYN.temperature_model

    if timestep is not None:
        DYN.timestep = float(timestep)
    if temperature_k is not None:
        DYN.temperature_k = float(temperature_k)
    if temperature_model is not None:
        DYN.temperature_model = str(temperature_model).lower()

    if impulse_times is None:
        impulse_times = [DYN.impulse_time]
    impulse_times = sorted(float(x) for x in impulse_times)

    try:
        # Validate temperature once at the start of the run.
        cached_temperature_ratio(DYN.temperature_k, DYN.temperature_model)

        positions = hex_positions(GEOM.num_rings, GEOM.hex_radius, GEOM.gap)
        pairs, mech_sets = build_adjacency(positions)

        # The small-displacement constitutive matrices are functions of the
        # field-cooling condition, not the instantaneous mechanism motion.
        # Evaluate them once per case.  Keeping these lookups inside run_case
        # also preserves the supplemental runners' ability to monkeypatch the
        # constitutive functions for alternate-matrix studies.
        case_K_local = np.asarray(
            mechanism_K6(
                GEOM.gap,
                DYN.temperature_k,
                DYN.temperature_model,
            ),
            dtype=float,
        )
        case_C_local = np.asarray(
            mechanism_C6(
                GEOM.gap,
                DYN.temperature_k,
                DYN.temperature_model,
                zeta=DYN.zeta_nominal,
            ),
            dtype=float,
        )
        model = mujoco.MjModel.from_xml_string(create_xml(positions))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_ids = {i: model.body(f"h{i}").id for i in range(len(positions))}
        center_idx = int(np.argmin([math.hypot(x, y) for x, y in positions]))
        anchor_id = body_ids[center_idx]

        p_refs = [
            np.array([x, y, GEOM.body_z], dtype=float)
            for x, y in positions
        ]
        R_refs = [np.eye(3) for _ in positions]

        t_log = []
        pos_log = []
        rot_log = []
        dof_log = []
        segment_dof_log = []
        force_log = []
        torque_log = []

        DYN.separation_clamp_count = 0
        DYN.max_normal_gap_excursion_m = 0.0
        DYN.max_local_translation_m = 0.0
        DYN.min_instantaneous_gap_m = float("inf")
        DYN.max_instantaneous_gap_m = 0.0
        clip_count = 0
        evaluations = 0
        max_force = 0.0
        max_torque = 0.0

        log_stride = max(int(log_stride), 1)
        step_index = 0
        with mujoco.viewer.launch_passive(model, data) if show_viewer else _null_context() as viewer:
            while data.time < DYN.run_time_s:
                data.qfrc_applied[:] = 0.0

                poses = [
                    body_pose(model, data, body_ids[i])
                    for i in range(len(positions))
                ]

                apply_anchor(
                    model,
                    data,
                    anchor_id,
                    p_refs[center_idx],
                    R_refs[center_idx],
                )

                step_force = 0.0
                step_torque = 0.0

                for i, j in pairs:
                    pose_i = poses[i]
                    pose_j = poses[j]
                    for mech in mech_sets[(i, j)]:
                        fmag, tmag, clips, raw_f, _raw_w = mechanism_step_force(
                            model,
                            data,
                            body_ids[i],
                            body_ids[j],
                            mech,
                            pose_i,
                            pose_j,
                            K_local=case_K_local,
                            C_local=case_C_local,
                        )
                        step_force += raw_f
                        step_torque += tmag
                        clip_count += clips
                        evaluations += 1
                        max_force = max(max_force, raw_f)
                        # tmag is the actual moment-arm torque applied to the
                        # segment. The wrench norm is not a torque diagnostic.
                        max_torque = max(max_torque, tmag)

                active_impulse = any(
                    t0 <= data.time < t0 + DYN.impulse_duration
                    for t0 in impulse_times
                )

                if active_impulse:
                    impulse_body = max(
                        body_ids.keys(),
                        key=lambda i: math.hypot(positions[i][0], positions[i][1]),
                    )
                    p, R, _, _ = poses[impulse_body]
                    local = np.asarray(DYN.impulse_point_offset, dtype=float)
                    p_imp = p + R @ local
                    f_imp = impulse_force_scale * np.asarray(DYN.impulse_force, dtype=float)
                    apply_point_force(model, data, body_ids[impulse_body], p_imp, f_imp)

                mujoco.mj_step(model, data)

                dofs = []
                for i in range(len(positions)):
                    bid = body_ids[i]
                    p, R, _, _ = body_pose(model, data, bid)
                    dp = p - p_refs[i]
                    dtheta = rotation_matrix_to_rotvec(R @ R_refs[i].T)
                    dofs.append(np.concatenate((dp, dtheta)))

                dofs = np.asarray(dofs, dtype=float)
                mean_dof = np.mean(np.abs(dofs), axis=0)

                current_pos = float(np.mean(np.linalg.norm(dofs[:, :3], axis=1)))
                current_rot = float(np.mean(np.linalg.norm(dofs[:, 3:], axis=1)))
                should_stop = bool(
                    stop_callback(data.time, dofs, current_pos, current_rot)
                ) if stop_callback is not None else False

                if step_index % log_stride == 0 or should_stop:
                    t_log.append(data.time)
                    dof_log.append(mean_dof)
                    segment_dof_log.append(dofs.copy())
                    pos_log.append(current_pos)
                    rot_log.append(current_rot)
                    force_log.append(step_force / max(len(pairs) * GEOM.mechanisms_per_edge, 1))
                    torque_log.append(step_torque / max(len(pairs), 1))
                step_index += 1

                if viewer is not None:
                    viewer.sync()
                if should_stop:
                    break

        dof_log = np.asarray(dof_log)
        return RunResult(
            t=np.asarray(t_log),
            pos_err=np.asarray(pos_log),
            rot_err=np.asarray(rot_log),
            dof_err=dof_log,
            force_mag=np.asarray(force_log),
            torque_mag=np.asarray(torque_log),
            clip_count=clip_count,
            force_evaluations=evaluations,
            max_force=max_force,
            max_torque=max_torque,
            segment_dofs=np.asarray(segment_dof_log),
        )
    finally:
        DYN.timestep = old_timestep
        DYN.temperature_k = old_temperature_k
        DYN.temperature_model = old_temperature_model


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main():
    positions = hex_positions(GEOM.num_rings, GEOM.hex_radius, GEOM.gap)
    pairs, _ = build_adjacency(positions)

    K6 = mechanism_K6(
        CONFIG.nominal_gap_m,
        CONFIG.reference_temperature_k,
        CONFIG.temperature_model,
    )
    C6 = mechanism_C6(
        CONFIG.nominal_gap_m,
        CONFIG.reference_temperature_k,
        CONFIG.temperature_model,
        zeta=CONFIG.zeta_nominal,
    )

    print("Version 4.9 field-cooling-fixed pure-translational mechanism model")
    print(f"segments={len(positions)}")
    print(f"interfaces={len(pairs)}")
    print(f"mechanisms_per_edge(total)={GEOM.mechanisms_per_edge}")
    print(f"mechanism_rows={GEOM.mechanism_rows} | row_separation={GEOM.row_separation_m*1e3:.1f} mm")
    print(f"segment mass={CONFIG.segment.mass_kg:.6f} kg")
    print(
        "segment inertia diag="
        f"[{CONFIG.segment.inertia_x_kgm2:.8f}, "
        f"{CONFIG.segment.inertia_y_kgm2:.8f}, "
        f"{CONFIG.segment.inertia_z_kgm2:.8f}] kg m^2"
    )
    print(
        f"nominal gap={CONFIG.nominal_gap_m*1e3:.3f} mm | "
        f"T={CONFIG.reference_temperature_k:.1f} K | "
        f"temperature model={CONFIG.temperature_model}"
    )
    print("Count normalization: OFF; each mechanism carries its full per-pair stiffness.")
    print("Mechanism rotational stiffness/coupling: OFF; segment rotation emerges from force moment arms.")
    print("Nominal mechanism K6 [N/m, N m/rad]:")
    print(np.array2string(K6, precision=8, suppress_small=True))
    print("Nominal mechanism C6 [SI units]:")
    print(np.array2string(C6, precision=8, suppress_small=True))

    result = run_case()
    print(f"peak position error = {np.max(result.pos_err):.6e} m")
    print(f"peak rotation error = {np.max(result.rot_err):.6e} rad")
    print(f"final position error = {result.pos_err[-1]:.6e} m")
    print(f"final rotation error = {result.rot_err[-1]:.6e} rad")
    print(f"clip count = {result.clip_count} / {result.force_evaluations}")
    print(f"separation clamp count = {DYN.separation_clamp_count}")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(result.t, result.pos_err)
    ax[0].set_ylabel("Mean position error (m)")
    ax[0].grid(True)

    ax[1].plot(result.t, result.rot_err)
    ax[1].set_ylabel("Mean rotation error (rad)")
    ax[1].grid(True)

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    ax[2].plot(result.t, result.dof_err)
    ax[2].set_ylabel("Mean abs DOF error")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(labels, ncol=3)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
