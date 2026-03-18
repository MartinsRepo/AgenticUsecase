# AgenticUsecase

The purpose of this repo was to show the capabilities of autogenerating a complete github project 
with the Gihub Copilot in **agent/planning**-mode. The project and it's branches are runnable. 
There will be no further development in the project.
To reproduce the result or similar, create an empty git project in your local workspace and
copy the **agents.md** inside. Then open VSCode, add the folder to your workspace and let 
Copilot play. The result would be similar, but not the same.

## ADAS Highway Simulation

A 2-D top-down highway simulation in Python featuring Advanced Driver Assistance Systems (ADAS):
**Lane Keeping Assist (LKA)** and **Adaptive Cruise Control (ACC)**.

### Features

- Real-time 60 FPS Pygame visualisation (top-down highway view)
- Kinematic bicycle model vehicle physics
- ACC: PID-based longitudinal speed control with time-headway gap regulation
- LKA: PID-based lateral steering control to keep the ego vehicle centred in its lane
- Forward radar sensor simulation using raycasting (AABB slab method)
- NPC traffic vehicles with constant-velocity model
- Interactive: place obstacles with the mouse, adjust target speed with arrow keys
- Transparent radar cone visualisation, proximity warning overlay, live telemetry HUD

### Project Structure

```
src/
    config.py               Global constants and colour scheme
    main.py                 Entry point and main game loop (MVC integration)
    models/
        vehicle.py          Kinematic bicycle model
        obstacle.py         Static obstacles and NPC vehicles
        world.py            World state (ego, NPCs, obstacles, road geometry)
    controllers/
        pid.py              Generic discrete-time PID controller
        acc.py              Adaptive Cruise Control
        lka.py              Lane Keeping Assist
    sensors/
        radar.py            Raycasting radar sensor
    views/
        renderer.py         Pygame renderer (road, vehicles, radar cone)
        hud.py              Telemetry HUD panel
tests/
    test_models.py          Vehicle physics unit tests
    test_controllers.py     PID / ACC / LKA unit tests
    test_sensors.py         Radar sensor unit tests
```

### Requirements

- Python 3.10+
- Pygame >= 2.5.0
- NumPy >= 1.24.0

```bash
pip install -r requirements.txt
```

### Running the simulation

```bash
python -m src.main
```

### Controls

| Key / Action         | Effect                        |
|----------------------|-------------------------------|
| Arrow UP             | Increase ACC target speed     |
| Arrow DOWN           | Decrease ACC target speed     |
| Left mouse click     | Place obstacle on road        |
| R                    | Remove all obstacles          |
| ESC                  | Quit                          |

### Running tests

```bash
python -m pytest tests/ -v
```

### Colour Scheme

| Element               | Hex       |
|-----------------------|-----------|
| Asphalt background    | `#2c3e50` |
| Lane markings         | `#ffffff` |
| Ego vehicle           | `#209dd7` |
| Radar cone            | `#ecad0a` |
| Warning / danger      | `#e74c3c` |
| UI telemetry text     | `#888888` |
