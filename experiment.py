# Graded Awareness Experiment with Implicit Priming
# ----------------------------------------------------
# Author: (your name here)
# PsychoPy version: 2025.1
# Description:
#   Continuous‑report Gabor orientation task with individualized
#   SOA staircasing, confidence ratings, catch trials, and optional
#   implicit‐priming RT probe. Tuned for a 144 Hz monitor.
# ----------------------------------------------------

from psychopy import core, visual, event, data, gui, prefs, logging, monitors
from psychopy.hardware import keyboard
import numpy as np
import random
import os

# -----------------------
# EXPERIMENT PARAMETERS
# -----------------------
MONITOR_REFRESH_HZ = 144               # set to your display
FRAME_DUR = 1.0 / MONITOR_REFRESH_HZ   # seconds per frame
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

FIXATION_DUR   = 0.500  # s
MASK_DUR       = 0.100  # s (forward & backward)
PRIME_DUR      = 0.016  # s (one frame @ 60 Hz, ~2 frames @144 Hz)
ITI_MEAN       = 0.800  # mean ITI (jitter added)

# Staircase settings
STAIR_INITIAL_SOA = 0.100             # initial guess 100 ms
STAIR_MIN_SOA     = 0.016             # minimum 1 frame
STAIR_MAX_SOA     = 0.300             # maximum 300 ms
STAIR_N_REVERSALS = 8
STAIR_N_TRIALS    = 40
STAIR_STEP        = 0.016             # step size 1 frame
TARGET_ACC        = 0.75

# Main‑block settings
SOA_STEPS_AROUND = [-0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05]  # relative to thresh
TRIALS_PER_SOA   = 30                 # total ~270 trials
CATCH_PROB       = 0.10               # 10 % catch trials
PRIME_PROB       = 0.50               # 50 % of (non‑catch) trials

CONF_LEVELS      = [0,10,20,30,40,50,60,70,80,90,100]  # confidence scale

# -----------------------
# INITIAL SETUP
# -----------------------

mon = monitors.Monitor('testMonitor')  
mon.setWidth(73)          # physical width of your display in cm
mon.setDistance(75)       # your viewing distance in cm
mon.setSizePix([SCREEN_WIDTH,SCREEN_HEIGHT])

exp_info = {"Participant":"", "Run":"001"}
dlg = gui.DlgFromDict(exp_info, title="Graded Awareness Study")
if not dlg.OK:
    core.quit()

filename = f"data/{exp_info['Participant']}_{exp_info['Run']}".replace(" ", "_")
this_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(this_dir, 'data')):
    os.makedirs(os.path.join(this_dir, 'data'))

# Window
win = visual.Window(
    size=(SCREEN_WIDTH,SCREEN_HEIGHT), 
    fullscr=True, 
    monitor=mon,
    units='deg'
)
win.setRecordFrameIntervals(True)
kb = keyboard.Keyboard()

# Stimuli
fixation = visual.TextStim(win, text='+', color='white', height=0.8)
noise_tex = np.random.rand(256,256)*2-1
mask_tex  = visual.GratingStim(win, tex=noise_tex, size=10, sf=0, units='deg')

# Gabor template (orientation set later each trial)
gabor = visual.GratingStim(win, tex='sin', mask='gauss', size=5, sf=3, units='deg', contrast=1.0)

# Probe for continuous‑report
probe_line = visual.Line(win, start=(0,0), end=(0,4), lineColor='white', lineWidth=4, units='deg')

# Confidence text
conf_text = visual.TextStim(win, text='', color='white', height=0.8, wrapWidth=20)

# RT Choice prompt
rt_prompt = visual.TextStim(win, text='Tilt?  ← / →', color='white', height=0.8)

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def draw_and_wait(stim, dur):
    stim.draw()
    win.flip()
    core.wait(dur)


def orientation_error(resp_angle, true_angle):
    diff = (resp_angle - true_angle + 90) % 180 - 90
    return diff


def get_confidence():
    idx = 0
    while True:
        conf_text.text = f"Confidence you saw anything? {CONF_LEVELS[idx]}%\n[left/right to adjust, space to confirm]"
        conf_text.draw()
        win.flip()
        keys = event.waitKeys(keyList=['left','right','space','escape'])
        if 'escape' in keys: core.quit()
        if 'left' in keys and idx>0: idx -= 1
        if 'right' in keys and idx<len(CONF_LEVELS)-1: idx += 1
        if 'space' in keys: return CONF_LEVELS[idx]


def continuous_report():
    angle = 0
    while True:
        probe_line.ori = angle
        probe_line.draw(); win.flip()
        keys = event.getKeys()
        if 'escape' in keys: core.quit()
        if 'left' in keys:  angle -= 1
        if 'right' in keys: angle += 1
        if 'space' in keys: return angle % 180


def run_rt_probe(prime_ori, match):
    # prime orientation: either matches target or orthogonal (prime_od)
    gabor.ori = prime_ori if match else (prime_ori+90) % 180
    draw_and_wait(gabor, PRIME_DUR)
    draw_and_wait(mask_tex, MASK_DUR)
    rt_prompt.draw(); win.flip(); kb.clock.reset();
    key = kb.waitKeys(keyList=['left','right','escape'])[0]
    rt = kb.clock.getTime()
    if key.name == 'escape': core.quit()
    correct = (key.name == 'left' and gabor.ori<90) or (key.name=='right' and gabor.ori>=90)
    return rt, correct

# -----------------------
# INSTRUCTIONS & TRAINING
# -----------------------

instruction_pages = [
    "Welcome!\n\nYou will briefly see a grating (striped pattern).\nYour tasks:\n1. Adjust the white line to match the grating’s tilt.\n2. Rate how confident you are that ANY stimulus was present.\n3. Occasionally respond LEFT or RIGHT as quickly as possible.\n\nPress SPACE to continue.",
    "During the task, some trials will have NO stimulus.\nRate 0% confidence when you see nothing.\n\nAdjust line: ← →\nConfirm line: SPACE\nRate confidence: ← → then SPACE\nTilt RT: ← (left)  → (right)\n\nPress SPACE to begin a short practice."
]
for page in instruction_pages:
    visual.TextStim(win, text=page, color='white', height=0.8, wrapWidth=20).draw(); win.flip(); event.waitKeys(keyList=['space'])

# Practice trials (5 demo trials, fixed easy SOA)
for _ in range(5):
    ori = random.uniform(0,180)
    fixation.draw(); win.flip(); core.wait(FIXATION_DUR)
    draw_and_wait(mask_tex, MASK_DUR)
    gabor.ori = ori
    draw_and_wait(gabor, 0.150)
    draw_and_wait(mask_tex, MASK_DUR)
    resp_angle = continuous_report()
    conf_val   = get_confidence()
    core.wait(0.3)

# -----------------------
# STAIRCASE (2‑up/1‑down)
# -----------------------

dlg = gui.Dlg(title='Staircase starting...'); dlg.addText('Press OK'); dlg.show()

stair_handler = data.StairHandler(startVal=STAIR_INITIAL_SOA, stepSizes=STAIR_STEP,
                                 stepType='lin', nTrials=STAIR_N_TRIALS, nUp=2, nDown=1,
                                 minVal=STAIR_MIN_SOA, maxVal=STAIR_MAX_SOA, targetVal=TARGET_ACC)

for soa in stair_handler:
    ori = random.uniform(0,180)
    # --- Trial sequence ---
    fixation.draw(); win.flip(); core.wait(FIXATION_DUR)
    draw_and_wait(mask_tex, MASK_DUR)
    gabor.ori = ori
    draw_and_wait(gabor, soa)
    draw_and_wait(mask_tex, MASK_DUR)
    resp_angle = continuous_report()
    err = abs(orientation_error(resp_angle, ori))
    correct = err < 15  # coarse 30° window for staircase
    stair_handler.addResponse(correct)

# threshold estimate
thresh_soa = np.median(stair_handler.reversals) if stair_handler.reversals else stair_handler.mean()
print(f"Estimated 75% SOA: {thresh_soa*1000:.1f} ms")

# Generate SOA list for main trials
soa_list = [max(STAIR_MIN_SOA, min(STAIR_MAX_SOA, thresh_soa+delta)) for delta in SOA_STEPS_AROUND]
trials = []
for soa in soa_list:
    for _ in range(TRIALS_PER_SOA):
        trials.append({
            'soa': soa,
            'catch': random.random() < CATCH_PROB,
            'prime': False,
            'prime_match': False
        })
random.shuffle(trials)

# Assign primes
for t in trials:
    if (not t['catch']) and random.random() < PRIME_PROB:
        t['prime'] = True
        t['prime_match'] = random.choice([True, False])

trial_handler = data.TrialHandler(trials, 1, method='sequential', name='main')
trial_handler.data.addDataType('orientation')
trial_handler.data.addDataType('resp_angle')
trial_handler.data.addDataType('conf')
trial_handler.data.addDataType('err')
trial_handler.data.addDataType('rt')
trial_handler.data.addDataType('rt_correct')

# -----------------------
# MAIN LOOP
# -----------------------

for trial in trial_handler:
    # stimulus orientation
    ori = random.uniform(0,180)

    # fixation
    fixation.draw(); win.flip(); core.wait(FIXATION_DUR)

    # forward mask
    draw_and_wait(mask_tex, MASK_DUR)

    # target presentation (or catch)
    if not trial['catch']:
        gabor.ori = ori
        draw_and_wait(gabor, trial['soa'])
    else:
        core.wait(trial['soa'])  # gap of equal duration

    # backward mask
    draw_and_wait(mask_tex, MASK_DUR)

    # first‑order report
    resp_angle = continuous_report()

    # second‑order confidence
    conf_val = get_confidence()

    # optional implicit prime
    rt = np.nan; rt_correct = np.nan
    if trial['prime']:
        rt, rt_correct = run_rt_probe(ori, trial['prime_match'])

    # log
    err_val = orientation_error(resp_angle, ori) if not trial['catch'] else np.nan
    trial_handler.addData('orientation', ori)
    trial_handler.addData('resp_angle', resp_angle)
    trial_handler.addData('conf', conf_val)
    trial_handler.addData('err', err_val)
    trial_handler.addData('rt', rt)
    trial_handler.addData('rt_correct', rt_correct)

    # ITI
    iti = np.random.exponential(ITI_MEAN)
    core.wait(iti)

# Save data
trial_handler.saveAsWideText(filename + '.csv')
trial_handler.saveAsPickle(filename)

# End screen
visual.TextStim(win, text='Thank you!\n\nPress ESC or close window to exit.', color='white', height=1.0).draw(); win.flip()
keys = event.waitKeys(keyList=['escape']);
core.quit()
