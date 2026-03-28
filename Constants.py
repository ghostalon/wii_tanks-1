import pygame
FPS = 60

WIDTH, HEIGHT = 1500,900


screen = pygame.display.set_mode((WIDTH,HEIGHT))

#RGB
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (0, 128, 0)
CADETBLUE1 = (152,245,255)

TANK_SPEED = 7
MAX_AMMUNITION = 5
BULLET_SPEED = 10
ENEMY_SPEED = 2

PLAYER_URL = "Data/Player.png"
ENEMY_URL = "Data/Enemy.png"
BULLET_URL = "Data/Bullet.png"

PLAYER_IMAGE = pygame.image.load(PLAYER_URL)
ENEMY_IMAGE = pygame.image.load(ENEMY_URL)
ENEMY_IMAGE = pygame.transform.rotate(ENEMY_IMAGE, 180)

epsilon_start, epsilon_final, epsiln_decay = 1, 0.01, 5
epochs = 10000
batch_size = 36
MIN_BUFFER = 50
batch_size = 36

gamma = 0.95  # Discount factor

# ── Tunable constants ────────────────────────────────────────────
REWARD_WIN          =  100.0  # terminal: enemy destroyed
REWARD_LOSE         = -100.0  # terminal: own tank destroyed
STEP_PENALTY        =  -0.02  # small cost per step (anti-stall)
K_AIM               =   0.1   # scale for angular-error improvement
AIM_BONUS_NEAR      =   0.05  # bonus when aim error < AIM_THRESH_NEAR
AIM_THRESH_NEAR     =   8     # degrees
AIM_BONUS_LOCK      =   0.1   # extra bonus when aim error < AIM_THRESH_LOCK
AIM_THRESH_LOCK     =   5     # degrees
SHOOT_PENALTY       =  -0.2   # base cost for firing (anti-spam)
SHOOT_BONUS_AIMED   =   1.0   # added back when shooting while well-aimed
K_DANGER            =   0.5   # scale for danger reduction
# ────────────────────────────────────────────────────────────────

# ── Training constants ───────────────────────────────────────────
LEARNING_RATE       = 0.0001
TARGET_UPDATE_FREQ  = 1000    # gradient steps between target-net copies
WIN_RATE_WINDOW     = 20      # episodes for rolling win-rate
CHECKPOINT_INTERVAL = 100     # save checkpoint every N epochs
OPPONENT_EPSILON    = 0.5     # Advanced_Random_Agent randomness
# ────────────────────────────────────────────────────────────────
