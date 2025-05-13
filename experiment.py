"""
Graded-Awareness Toy Demo
-------------------------
A lean PsychoPy implementation of:
mask → stimulus (SOA) → mask → orientation + confidence
→ mask → stimulus (same SOA) → mask → prime → reverse-mask → RT  (← / →)

Author : you
Version: 2025-05-12  (toy settings)
"""

from psychopy import core, visual, event, data, gui, logging, monitors
from psychopy.hardware import keyboard
import numpy as np, random, csv, os, math, datetime

# ===============  HYPER-PARAMETERS (Toy)  ========================
N_PRACTICE            = 2
N_STAIR_TRIALS        = 6
STAIR_STEP_MS         = 20           # ms
MIN_SOA_MS            = 50
MAX_SOA_MS            = 150
N_SOAS                = 3
TRIALS_PER_SOA        = 3            # 3 × 3  = 9 main trials
PRIME_MATCH_PROB      = 0.50         # congruent vs. incongruent
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

FIX_MS      = 500
MASK_MS     = 100
PRIME_MS    = 33                     # one 30-Hz frame ≈ 33 ms
REV_MASK_MS = 100
FPS         = 60                     # change if needed
# ================================================================

# ----------  Basic dialog & files ----------
exp_info = {
    "Participant": "",
    "Run": "001",
    "Date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
}
if not gui.DlgFromDict(exp_info, title="Graded-Awareness Toy").OK:
    core.quit()

this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir,
    f"{exp_info['Participant']}_{exp_info['Run']}_{exp_info['Date']}.csv")

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["subj","phase","trial",
                     "soa_ms","target_ori","estimate",
                     "confidence","rt_prime_ms",
                     "prime_congruent","seen_in_stair"])

# ----------  Window / stimuli ----------
mon = monitors.Monitor('testMonitor')  
mon.setWidth(73)          # physical width of your display in cm
mon.setDistance(75)       # your viewing distance in cm
mon.setSizePix([SCREEN_WIDTH,SCREEN_HEIGHT])
# Window
win = visual.Window(
    size=(SCREEN_WIDTH,SCREEN_HEIGHT), 
    fullscr=False, 
    monitor=mon,
    units='deg'
)
mouse = event.Mouse(win=win)
kb    = keyboard.Keyboard()

fix = visual.TextStim(win, text="+", height=0.8, color="white")
noise_array = (np.random.rand(256,256) * 255).astype('uint8')
mask = visual.ImageStim(win,
                        image=noise_array,
                        size=10,
                        units="deg",
                        interpolate=False)
gabor = visual.GratingStim(win, tex="sin", mask="gauss",
                           size=5, sf=3, units="deg", contrast=1.0)
prime_prompt = visual.TextStim(win, text="Tilt?  ←  /  →",
                               height=0.8, color="white")
line_probe = visual.Line(win, start=(0,0), end=(0,4),
                         lineWidth=4, lineColor="white", units="deg")

conf_bar = visual.Rect(win, width=8, height=0.6, lineColor="white",
                       fillColor=None, pos=(0,-4))
conf_marker = visual.Rect(win, width=0.2, height=0.8,
                          lineColor=None, fillColor="white")

# ----------  Helper functions ----------
def wait_ms(ms):
    core.wait(ms/1000.0)

def angle_from_mouse():
    x,y = mouse.getPos()
    angle = (math.degrees(math.atan2(y,x)) + 360) % 180  # 0-180
    return angle

def orientation_estimate():
    mouse.clickReset()
    while True:
        line_probe.ori = angle_from_mouse()
        line_probe.draw(); win.flip()
        if mouse.getPressed()[0]:  # left click to confirm
            core.wait(0.2)
            return line_probe.ori

def confidence_estimate():
    mouse.clickReset()
    while True:
        x,_ = mouse.getPos()
        x = np.clip(x, -4, 4)
        pct = int(np.interp(x, (-4,4), (0,100))//10*10)
        conf_marker.pos = (x,-4)
        conf_bar.draw(); conf_marker.draw()
        visual.TextStim(win,
            text=f"Confidence you saw ANY stimulus: {pct} %",
            pos=(0,-2.5), height=0.6).draw()
        win.flip()
        if mouse.getPressed()[0]:
            core.wait(0.2)
            return pct

def draw_and_wait(stim, ms):
    stim.draw(); win.flip(); wait_ms(ms)

# ----------  Practice ----------
for p in range(N_PRACTICE):
    true_ori = random.uniform(0,180)
    fix.draw(); win.flip(); wait_ms(FIX_MS)
    draw_and_wait(mask, MASK_MS)
    gabor.ori = true_ori
    draw_and_wait(gabor, 150)
    draw_and_wait(mask, MASK_MS)
    est = orientation_estimate()
    conf = confidence_estimate()

# ----------  Simple staircase (1-up / 1-down) ----------
soa_ms = (MIN_SOA_MS+MAX_SOA_MS)/2
for s in range(N_STAIR_TRIALS):
    true_ori = random.uniform(0,180)
    fix.draw(); win.flip(); wait_ms(FIX_MS)
    draw_and_wait(mask, MASK_MS)
    gabor.ori = true_ori
    draw_and_wait(gabor, soa_ms)
    draw_and_wait(mask, MASK_MS)
    est = orientation_estimate()
    err = abs((est-true_ori+90)%180-90)
    seen = err < 30  # arbitrary “saw” criterion
    # staircase update
    if seen: soa_ms = max(MIN_SOA_MS, soa_ms-STAIR_STEP_MS)
    else:    soa_ms = min(MAX_SOA_MS, soa_ms+STAIR_STEP_MS)

# ----------  Build main trial list ----------
SOAS = np.linspace(MIN_SOA_MS, MAX_SOA_MS, N_SOAS)
trials = []
for soa in SOAS:
    for _ in range(TRIALS_PER_SOA):
        trials.append({
            "soa": soa,
            "prime_congruent": random.random() < PRIME_MATCH_PROB
        })
random.shuffle(trials)

# ----------  MAIN LOOP ----------
for tidx, trial in enumerate(trials):
    true_ori = random.uniform(0,180)

    # first presentation
    fix.draw(); win.flip(); wait_ms(FIX_MS)
    draw_and_wait(mask, MASK_MS)
    gabor.ori = true_ori
    draw_and_wait(gabor, trial["soa"])
    draw_and_wait(mask, MASK_MS)

    # orientation + confidence
    est = orientation_estimate()
    conf = confidence_estimate()

    # prime + RT
    draw_and_wait(mask, MASK_MS)
    gabor.ori = true_ori
    draw_and_wait(gabor, trial["soa"])
    draw_and_wait(mask, MASK_MS)
    prime_ori = true_ori if trial["prime_congruent"] else (true_ori+90)%180
    gabor.ori = prime_ori
    draw_and_wait(gabor, PRIME_MS)
    draw_and_wait(mask, REV_MASK_MS)
    prime_prompt.draw(); win.flip(); kb.clock.reset()
    key = kb.waitKeys(keyList=["left","right"])[0]
    rt_ms = kb.clock.getTime()*1000

    # log immediately
    csv_writer.writerow([exp_info["Participant"],
        "main", tidx, trial["soa"], true_ori, est, conf,
        rt_ms, trial["prime_congruent"], ""])
    csv_file.flush()

    # inter-trial pause
    wait_ms(300)

# ----------  tidy up ----------
csv_file.close()
visual.TextStim(win, text="Done!  Press ESC to quit.",
                height=1).draw(); win.flip()
event.waitKeys(keyList=["escape"])
core.quit()
