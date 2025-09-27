import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import matplotlib as mpl
    import imageio_ffmpeg
    from nsf_rl.dmp import DMPConfig, PlanarDMP, DMPParams

    # Ensure Matplotlib's FFMpegWriter uses imageio-ffmpeg's bundled ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    DMP_SUBSTEPS = 8  # Integrate DMP with this many substeps per env control step
    return DMPConfig, DMPParams, PlanarDMP, animation, np, plt, DMP_SUBSTEPS


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Normalized planar DMP explorer

    The end-effector workspace is normalized to \([-1, 1]^2\) so slider values
    stay in a reasonable range independent of pixels. Use the checkbox to toggle
    whether the forcing term is scaled by \((\mathbf{g}-\mathbf{x}_0)\), and
    explore how the same weights behave with and without goal-direction
    modulation.
    """
    )
    return


@app.cell
def _(mo, np):
    n_basis = 3
    start_x = mo.ui.slider(180, 340, step=1, value=260, label="start_x")
    start_y = mo.ui.slider(180, 340, step=1, value=260, label="start_y")
    duration = mo.ui.slider(1.0, 4.0, step=0.1, value=2.5, label="duration [s]")
    radius = mo.ui.slider(30.0, 180.0, step=5.0, value=120.0, label="goal radius")
    angle = mo.ui.slider(-np.pi, np.pi, step=0.1, value=0.0, label="goal angle (rad)")
    stiffness = mo.ui.slider(5.0, 35.0, step=1.0, value=25.0, label="stiffness α_z")
    scale_forcing = mo.ui.checkbox(False, label="scale forcing")

    # PushT simulation controls
    object_x = mo.ui.slider(0.0, 512.0, step=1.0, value=256.0, label="object_x")
    object_y = mo.ui.slider(0.0, 512.0, step=1.0, value=256.0, label="object_y")
    block_angle = mo.ui.slider(-np.pi, np.pi, step=0.1, value=0.0, label="block angle (rad)")

    weights_x = [
        mo.ui.slider(-30, 30, step=0.5, value=0.0, label=f"w_x{i}")
        for i in range(n_basis)
    ]
    weights_y = [
        mo.ui.slider(-30, 30, step=0.5, value=0.0, label=f"w_y{i}")
        for i in range(n_basis)
    ]

    col_x = mo.vstack([mo.md("### Weights (x)")] + weights_x)
    col_y = mo.vstack([mo.md("### Weights (y)")] + weights_y)
    mo.vstack([
        mo.hstack([start_x, start_y, duration]),
        mo.hstack([radius, angle, stiffness, scale_forcing]),
        mo.hstack([object_x, object_y, block_angle]),
        mo.hstack([col_x, col_y]),
    ])
    return (
        angle,
        block_angle,
        duration,
        n_basis,
        object_x,
        object_y,
        radius,
        scale_forcing,
        start_x,
        start_y,
        stiffness,
        weights_x,
        weights_y,
    )


@app.cell
def _(
    DMPConfig,
    DMPParams,
    PlanarDMP,
    angle,
    animation,
    block_angle,
    duration,
    mo,
    n_basis,
    np,
    DMP_SUBSTEPS,
    object_x,
    object_y,
    plt,
    radius,
    scale_forcing,
    start_x,
    start_y,
    stiffness,
    weights_x,
    weights_y,
):
    import tempfile

    workspace_low = 0.0
    workspace_high = 512.0
    span = workspace_high - workspace_low
    half_span = span / 2.0
    center = workspace_low + half_span

    def to_normalized(pix):
        arr = np.asarray(pix, dtype=np.float32)
        return (arr - center) / half_span

    def to_pixels(norm):
        arr = np.asarray(norm, dtype=np.float32)
        return arr * half_span + center

    start_pix = np.array([start_x.value, start_y.value], dtype=np.float32)
    start_pix = np.clip(start_pix, workspace_low, workspace_high)
    goal_direction = np.array([np.cos(angle.value), np.sin(angle.value)], dtype=np.float32)
    goal_pix = start_pix + radius.value * goal_direction
    goal_pix = np.clip(goal_pix, workspace_low, workspace_high)

    start = np.clip(to_normalized(start_pix), -1.0, 1.0)
    goal = np.clip(to_normalized(goal_pix), -1.0, 1.0)

    weights = np.stack(
        [
            np.array([slider.value for slider in weights_x], dtype=np.float32),
            np.array([slider.value for slider in weights_y], dtype=np.float32),
        ],
        axis=0,
    )
    cfg = DMPConfig(
        n_basis=n_basis,
        min_duration=duration.value,
        max_duration=duration.value,
        workspace_low=-1.0,
        workspace_high=1.0,
        weight_scale=0.0,
        goal_noise=0.0,
        start_noise=0.0,
        stiffness=stiffness.value,
        scale_forcing_by_goal_delta=bool(scale_forcing.value),
    )

    env_dt = 0.1  # PushT control period in seconds
    dmp = PlanarDMP(dt=env_dt / DMP_SUBSTEPS, config=cfg)

    params = DMPParams(
        duration=duration.value,
        start=start,
        goal=goal,
        weights=weights,
        stiffness=cfg.stiffness,
        damping=2.0 * np.sqrt(cfg.stiffness),
    )

    positions_norm_full, _ = dmp.rollout(params)
    positions_full = np.clip(to_pixels(positions_norm_full), workspace_low, workspace_high)
    env_indices = np.arange(0, positions_full.shape[0], DMP_SUBSTEPS, dtype=int)
    if env_indices.size == 0 or env_indices[-1] != positions_full.shape[0] - 1:
        env_indices = np.append(env_indices, positions_full.shape[0] - 1)
    positions = positions_full[env_indices]

    from nsf_rl.utils.pymunk_compat import ensure_add_collision_handler

    ensure_add_collision_handler()
    import gymnasium as gym  # type: ignore

    ensure_add_collision_handler()
    import gym_pusht  # type: ignore # noqa: F401
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    try:
        _state, _info = env.reset(
            options={
                "reset_to_state": [
                    float(start_pix[0]),
                    float(start_pix[1]),
                    float(object_x.value),
                    float(object_y.value),
                    float(block_angle.value),
                ]
            }
        )
    except Exception:
        _state, _info = env.reset()

    frames = []
    for i in range(len(positions)):
        action = positions[i].astype(np.float32)
        action = np.clip(action, workspace_low, workspace_high)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if bool(terminated) or bool(truncated):
            break

    env.close()

    if not frames:
        mo.md("No frames rendered from gym-pusht.")
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        im = ax.imshow(frames[0])

        def init():
            im.set_data(frames[0])
            return (im,)

        def update(i):
            im.set_data(frames[i])
            return (im,)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            init_func=init,
            blit=True,
            interval=100,
        )

        writer = animation.FFMpegWriter(fps=10)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            anim.save(tmp.name, writer=writer)
            with open(tmp.name, "rb") as f:
                video_bytes = f.read()
        plt.close(fig)
    return (video_bytes,)


@app.cell
def _(mo, video_bytes):
    mo.video(video_bytes, controls=True, loop=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
