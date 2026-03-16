# ADAS Simulation - Lane Keeping Assist & Adaptive Cruise Control

A 2D top-down highway simulation in Python demonstrating two Advanced Driver
Assistance Systems (ADAS):

- **LKA** - Lane Keeping Assist: keeps the ego vehicle centred in its lane using a PID controller on the lateral offset.
- **ACC** - Adaptive Cruise Control: maintains a target speed and automatically brakes when a leading vehicle or obstacle is detected.

![ADAS Simulation Screenshot](https://github.com/user-attachments/assets/85b1c1b6-e45d-4c38-882f-1b5fa8a07e76)

---

## Features

- Kinematic bicycle model for realistic vehicle dynamics
- Ray-casting radar sensor (forward detection cone)
- Lane offset sensor for lateral positioning
- PID controllers with anti-windup for both ACC and LKA
- 60 FPS Pygame rendering with semi-transparent sensor cone
- Proximity collision warning banner
- Interactive mouse-click obstacle placement and removal
- Real-time telemetry HUD (speed, distance, offset, system status)
- NPC traffic that wraps from top to bottom continuously

---

## Architecture

The project follows a strict **MVC** pattern:

```
src/
  model/
    vehicle.py    - Kinematic bicycle model (Model)
    sensor.py     - RadarSensor (ray-casting) and LaneSensor
  controller/
    pid.py        - Generic discrete-time PID controller
    adas.py       - AdaptiveCruiseControl and LaneKeepingAssist
  view/
    renderer.py   - Pygame rendering layer (View)
  main.py         - Simulation coordinator and entry point (Controller)
tests/
  test_vehicle.py
  test_sensor.py
  test_pid.py
  test_adas.py
```

---

## Requirements

- Python 3.10+
- pygame >= 2.5.0
- numpy >= 1.24.0

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python -m src.main
```

---

## Controls

| Input | Action |
|---|---|
| Left mouse click (on road) | Place a static obstacle |
| Right mouse click | Remove nearest obstacle |
| `+` / `-` | Increase / decrease ACC target speed |
| `A` | Toggle Adaptive Cruise Control |
| `L` | Toggle Lane Keeping Assist |
| `R` | Reset ego vehicle and clear obstacles |
| `ESC` | Quit |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Color Scheme

| Element | Color | Hex |
|---|---|---|
| Road (asphalt) | Dark blue-gray | `#2c3e50` |
| Lane markings | White | `#ffffff` |
| Ego vehicle | Blue | `#209dd7` |
| Radar sensor cone | Accent yellow | `#ecad0a` |
| Proximity warning | Red | `#e74c3c` |
| HUD telemetry | Gray | `#888888` |
| NPC vehicles | Green | `#27ae60` |
| Static obstacles | Dark red | `#c0392b` |

---

## Physics

The vehicle uses a **kinematic bicycle model**:

```
x(t+dt)       = x(t) + v * cos(psi) * dt
y(t+dt)       = y(t) + v * sin(psi) * dt
psi(t+dt)     = psi(t) + (v / L) * tan(delta) * dt
v(t+dt)       = v(t) + a * dt
```

Where `L` is the wheelbase, `delta` is the front-wheel steering angle, and `a` is
the longitudinal acceleration commanded by the ACC PID loop.

The LKA computes a steering correction via a PID controller whose process
variable is the lateral offset measured by the lane sensor.
