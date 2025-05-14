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
N_STAIR_TRIALS        = 8
STAIR_STEP_MS         = 20           # ms
MIN_SOA_MS            = 7
MAX_SOA_MS            = 250
N_SOAS                = 9
TRIALS_PER_SOA        = 3            # 3 × 3  = 9 main trials
PRIME_MATCH_PROB      = 0.50         # congruent vs. incongruent
CATCH_PROB            = 0.1
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
ORIENTATIONS = [45, 135]

FIX_MS      = 500
MASK_MS     = 100
PRIME_MS    = 33                     # one 30-Hz frame ≈ 33 ms
REV_MASK_MS = 100
FPS         = 144                     # change if needed
# ================================================================

# ----------  Basic dialog & files ----------
exp_info = {
    "Participant": "",
    "Run": "001",
    "Date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
}
if not gui.DlgFromDict(exp_info, title="Graded-Awareness Test").OK:
    core.quit()

this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir,
    f"{exp_info['Participant']}_{exp_info['Run']}_{exp_info['Date']}.csv")

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["subj","phase","trial",
                     "soa_ms","catch_trial","detect_YN",
                     "confidence","rt_prime_ms",
                     "prime_congruent"])

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
frame_dur = win.getMsPerFrame(nIdentical=10, nWarmUp=10)  # returns ms per frame
print(f"Measured frame duration: {frame_dur:.3f} ms")
mouse = event.Mouse(win=win)
kb    = keyboard.Keyboard()

fix = visual.TextStim(win, text="+", height=0.8, color="white")
gabor = visual.GratingStim(win, tex="sin", mask="gauss",
                           size=5, sf=3, units="deg", contrast=1.0)
detect_prompt = visual.TextStim(win, text="Did you see any stimulus?  ↑ YES /  ↓ NO",
                               height=0.8, color="white")
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
        
def detection_yn():
    detect_prompt.draw(); win.flip(); kb.clock.reset()
    return kb.waitKeys(keyList=["up","down"])[0]

def prime_rt():
    prime_prompt.draw(); win.flip(); kb.clock.reset()
    prime_key = kb.waitKeys(keyList=["left","right"])[0]
    rt_ms = kb.clock.getTime()*1000
    return prime_key, rt_ms

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

def draw_mask():
    noise_array = (np.random.rand(256,256) * 2 - 1).astype('float32')
    mask = visual.ImageStim(win,
                            image=noise_array,
                            colorSpace='rgb',
                            size=10,
                            units="deg",
                            interpolate=False)
    draw_and_wait(mask, MASK_MS)

def text_and_wait(text):
    transition_text = visual.TextStim(
        win,
        text="Great job!  Press SPACE to continue.",
        color="white",
        height=1.0,
        wrapWidth=20
    )
    transition_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

# ----------  Practice ----------
PRACTICE_TEXT = "You are now \n beginning practice trials. 2 trials will have no prime RT, then 2 will have RT.\nPress space bar to continue"
text_and_wait(PRACTICE_TEXT)
for p in range(N_PRACTICE):
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, FIX_MS)
    draw_mask()
    gabor.ori = true_ori
    draw_and_wait(gabor, 150)
    draw_mask()
    detect_key = detection_yn()
    conf = confidence_estimate()
    print(detect_key, conf)
for p in range(N_PRACTICE):
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, FIX_MS)
    draw_mask()
    gabor.ori = true_ori
    draw_and_wait(gabor, 150)
    draw_mask()
    prime_ori = true_ori if random.random()<PRIME_MATCH_PROB else (true_ori + 90) % 180
    gabor.ori = prime_ori
    draw_and_wait(gabor, PRIME_MS)
    draw_mask()
    prime_key, rt_ms = prime_rt()
    detect_key = detection_yn()
    conf = confidence_estimate()
    print(detect_key, conf, prime_key, rt_ms, prime_ori==true_ori)
    

# ----------  Simple staircase (1-up / 1-down) ----------
STAIRCASE_TEXT = "We will now staircase your SOA to 50% detection.\nPress space bar to continue"
text_and_wait(STAIRCASE_TEXT)
soa_ms = (MIN_SOA_MS+MAX_SOA_MS)/2
for s in range(N_STAIR_TRIALS):
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, FIX_MS)
    draw_mask()
    gabor.ori = true_ori
    draw_and_wait(gabor, soa_ms)
    draw_mask()
    detect_key = detection_yn()
    seen = (detect_key == "up")
    print(seen, soa_ms)
    if seen: soa_ms = max(MIN_SOA_MS, soa_ms-STAIR_STEP_MS)
    else:    soa_ms = min(MAX_SOA_MS, soa_ms+STAIR_STEP_MS)

SC_SOA = soa_ms

# ----------  Build main trial list ----------
if N_SOAS == 9:
    SOA_OFFSETS_MS = np.array([-50, -35, -20, -10, 0, 10, 20, 35, 50])
else:
    SOA_OFFSETS_MS = np.array([-40, -25, -10, 0, 10, 25, 40])
SOAS = SC_SOA + SOA_OFFSETS_MS
SOAS = np.clip(SOAS, MIN_SOA_MS, MAX_SOA_MS)
print(SOAS)
trials = []
for soa in SOAS:
    for _ in range(TRIALS_PER_SOA):
        trials.append({
            "soa": soa,
            "catch": False,
            "prime_congruent": random.random()<PRIME_MATCH_PROB
        })

# add catch trials explicitly
n_catch = int(TRIALS_PER_SOA * len(SOAS) * CATCH_PROB)
for _ in range(n_catch):
    trials.append({
        "soa": SC_SOA,        # or you can reuse SC_SOA if you like
        "catch": True,
        "prime_congruent": False
    })
random.shuffle(trials)

# ----------  MAIN LOOP ----------
STAIRCASE_TEXT = "We will now begin the experiment.\nPress space bar to continue"
text_and_wait(STAIRCASE_TEXT)
for tidx, trial in enumerate(trials):
    true_ori = random.choice(ORIENTATIONS)

    # first presentation
    draw_and_wait(fix, FIX_MS)
    draw_mask()
    if not trial["catch"]:
        gabor.ori = true_ori
        draw_and_wait(gabor, trial["soa"])
        draw_mask()
    else:
        wait_ms(trial["soa"] or SC_SOA)
        draw_mask()
    prime_ori = true_ori if trial["prime_congruent"] else (true_ori+90)%180
    gabor.ori = prime_ori
    draw_and_wait(gabor, PRIME_MS)
    draw_mask()
    prime_key, rt_ms = prime_rt()
    detect_key = detection_yn()
    conf = confidence_estimate()
    print(detect_key, conf, prime_key, rt_ms, trial["prime_congruent"], trial['catch'])

    csv_writer.writerow([exp_info["Participant"],
        "main", tidx, trial["soa"], trial["catch"], detect_key, conf,
        rt_ms, trial["prime_congruent"]])
    csv_file.flush()

    # inter-trial pause
    wait_ms(300)

# ----------  tidy up ----------
csv_file.close()
visual.TextStim(win, text="Done!  Press ESC to quit.",
                height=1).draw(); win.flip()
event.waitKeys(keyList=["escape"])
core.quit()
