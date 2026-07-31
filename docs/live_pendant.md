# UR10e live pendant

The project includes a local, read-only browser view for the real UR10e. It is
not a remote copy of PolyScope: PolyScope 5 does not expose its teach-pendant UI
as a web application. Instead, this page visualizes the live ROS 2 driver data
that is useful during supervised feeding work.

The page shows:

- robot, safety, program, and remote-control state;
- controller speed scaling;
- TCP position in millimetres and orientation as a PolyScope-style rotation
  vector in radians;
- all reported joint positions and velocities;
- digital I/O and tool electrical/temperature data;
- ROS 2 controller states; and
- the wrist RGB-D image when `/wrist_rgbd/image` is publishing.

It has only HTTP `GET` endpoints and makes only subscriptions plus three
read-only queries (`list_controllers`, `is_in_remote_control`, and
`program_state`). It cannot play or stop programs, switch controllers, write
I/O, change speed, or command motion.

## Start it

With the real driver already running:

```bash
scripts/start_live_pendant.sh
```

Then open <http://127.0.0.1:8088>. The server binds to localhost by default.
To view it from another trusted device on the robot LAN, opt in explicitly:

```bash
scripts/start_live_pendant.sh --host 0.0.0.0
```

The daily startup helper, `scripts/daily_log_and_polyscope.sh`, starts the
localhost page in the background when needed and opens it in the workstation's
browser. Its PID and output are written under `logs/`.

For UI development without ROS or a connected robot, use:

```bash
python3 -m live_pendant.server --demo
```

The health and telemetry endpoints are `/api/health` and `/api/status`.
