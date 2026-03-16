"""ADAS Highway Simulation - global configuration and constants."""

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
WINDOW_WIDTH: int = 1200
WINDOW_HEIGHT: int = 800
FPS: int = 60
WINDOW_TITLE: str = "ADAS Highway Simulation - LKA & ACC"

# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
PIXELS_PER_METER: float = 22.0  # 1 metre = 22 pixels

# ---------------------------------------------------------------------------
# Road geometry (metres)
# ---------------------------------------------------------------------------
LANE_WIDTH_M: float = 3.7
NUM_LANES: int = 3
SHOULDER_WIDTH_M: float = 2.0
ROAD_WIDTH_M: float = NUM_LANES * LANE_WIDTH_M + 2.0 * SHOULDER_WIDTH_M

# In pixels
LANE_WIDTH_PX: int = int(LANE_WIDTH_M * PIXELS_PER_METER)   # ~81 px
ROAD_WIDTH_PX: int = int(ROAD_WIDTH_M * PIXELS_PER_METER)   # ~287 px

# Horizontal centre of the road on screen
ROAD_CENTER_X: int = WINDOW_WIDTH // 2 + 80  # 680 px – gives room for left HUD

# Left / right road edges in screen coords
ROAD_LEFT_X: int = ROAD_CENTER_X - ROAD_WIDTH_PX // 2
ROAD_RIGHT_X: int = ROAD_LEFT_X + ROAD_WIDTH_PX

# ---------------------------------------------------------------------------
# Vehicle dimensions (metres)
# ---------------------------------------------------------------------------
EGO_WIDTH_M: float = 2.0
EGO_LENGTH_M: float = 4.5
NPC_WIDTH_M: float = 2.0
NPC_LENGTH_M: float = 4.5

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
WHEELBASE_M: float = 2.7
MAX_STEER_RAD: float = 0.45
MAX_ACCEL_MS2: float = 4.0
MAX_DECEL_MS2: float = 9.0
MAX_SPEED_MS: float = 50.0

# ---------------------------------------------------------------------------
# ACC (Adaptive Cruise Control)
# ---------------------------------------------------------------------------
ACC_DEFAULT_TARGET_SPEED_MS: float = 28.0   # ~100 km/h
ACC_MIN_TARGET_SPEED_MS: float = 5.0
ACC_MAX_TARGET_SPEED_MS: float = 44.0       # ~160 km/h
ACC_SAFE_DISTANCE_M: float = 30.0           # desired following gap
ACC_WARNING_DISTANCE_M: float = 12.0        # critical – display warning colour
ACC_TIME_HEADWAY_S: float = 1.8             # time-headway for gap control

# PID gains for ACC speed controller
ACC_PID_KP: float = 1.2
ACC_PID_KI: float = 0.05
ACC_PID_KD: float = 0.3

# ---------------------------------------------------------------------------
# LKA (Lane Keeping Assist)
# ---------------------------------------------------------------------------
LKA_MAX_STEER_CORRECTION_RAD: float = 0.18

# PID gains for LKA lateral-error controller
LKA_PID_KP: float = 0.55
LKA_PID_KI: float = 0.01
LKA_PID_KD: float = 0.25

# ---------------------------------------------------------------------------
# Radar / sensor
# ---------------------------------------------------------------------------
RADAR_RANGE_M: float = 80.0
RADAR_FOV_DEG: float = 14.0   # half-angle of the forward cone

# ---------------------------------------------------------------------------
# Ego vehicle fixed vertical position on screen
# ---------------------------------------------------------------------------
EGO_SCREEN_Y: int = int(WINDOW_HEIGHT * 0.68)   # 544 px from top

# ---------------------------------------------------------------------------
# NPC traffic
# ---------------------------------------------------------------------------
NPC_COUNT: int = 6
NPC_BASE_SPEED_MS: float = 22.0   # base speed for NPC vehicles (~80 km/h)
NPC_SPEED_SPREAD_MS: float = 5.0  # ± spread around base speed
NPC_SPAWN_AHEAD_M: float = 120.0  # how far ahead to spawn NPCs
NPC_CULL_BEHIND_M: float = 60.0   # cull NPCs further than this behind ego

# ---------------------------------------------------------------------------
# Road markings
# ---------------------------------------------------------------------------
DASH_LENGTH_M: float = 3.0
DASH_GAP_M: float = 6.0

# ---------------------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------------------
SPEED_INCREMENT_MS: float = 2.0   # speed change per key press

# ---------------------------------------------------------------------------
# Colours  (R, G, B)
# ---------------------------------------------------------------------------
COLOR_ASPHALT: tuple[int, int, int] = (44, 62, 80)        # #2c3e50
COLOR_MARKING: tuple[int, int, int] = (255, 255, 255)     # #ffffff
COLOR_EGO: tuple[int, int, int] = (32, 157, 215)          # #209dd7
COLOR_RADAR: tuple[int, int, int] = (236, 173, 10)        # #ecad0a
COLOR_WARNING: tuple[int, int, int] = (231, 76, 60)       # #e74c3c
COLOR_UI_TEXT: tuple[int, int, int] = (136, 136, 136)     # #888888
COLOR_NPC: tuple[int, int, int] = (108, 122, 137)         # medium grey
COLOR_OBSTACLE: tuple[int, int, int] = (192, 57, 43)      # dark red
COLOR_OFFROAD: tuple[int, int, int] = (34, 49, 63)        # near-black green
COLOR_SHOULDER: tuple[int, int, int] = (55, 73, 87)       # slightly lighter asphalt
COLOR_PANEL_BG: tuple[int, int, int] = (20, 30, 40)       # HUD background
COLOR_PANEL_BORDER: tuple[int, int, int] = (60, 80, 95)   # HUD border

# Radar cone alpha (0-255)
RADAR_ALPHA: int = 70
