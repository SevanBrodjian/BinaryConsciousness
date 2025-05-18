"""
Graded-Awareness
-------------------------
A lean PsychoPy implementation of:
mask → stimulus (SOA) → mask → orientation + confidence
→ mask → stimulus (same SOA) → mask → prime → reverse-mask → RT  (← / →)

Author : Sevan Brodjian
Caltech CNS 176 - Cognition - Prof. Shimojo
Spring 2025
"""

from psychopy import core, visual, event, data, gui, logging, monitors
from psychopy.data import StairHandler, QuestHandler
from psychopy.hardware import keyboard
import numpy as np, random, csv, os, math, datetime
import numpy.fft as fft
import math


# ====================  HYPER-PARAMETERS  ========================
N_PRACTICE            = 8
N_DETECT_PRACTICE     = 23
N_STAIR_TRIALS        = 70
STAIR_STEP_MS         = 7
STAIR_START_MS        = 100
MIN_SOA_MS            = 7
MAX_SOA_MS            = 500
N_SOAS                = 9
TRIALS_PER_SOA        = 16
PRIME_CONTRAST        = 0.35
PRIME_OPACITY         = 0.5
PRIME_MATCH_PROB      = 0.50
CATCH_PER_SOA         = 2
ORIENTATIONS          = [45, 135]
MASK_MS               = 100
PRIME_SOA             = 5
REV_MASK_MS           = 100
# ================================================================

# ----------  Basic dialog & files ----------
exp_info = {
    "Participant": "",
    "Run": "001",
    "Date": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
}
if not gui.DlgFromDict(exp_info, title="Graded-Awareness Experiment").OK:
    core.quit()

this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir,
    f"{exp_info['Participant']}_{exp_info['Run']}_{exp_info['Date']}.csv")

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["subj","trial","soa_ms","catch_trial",
                     "detect_yn","confidence","rt_prime_ms", "prime_shown",
                     "prime_congruent", "prime_correct", "threshold"])
# subj: Subject Name
# trial: Trial Number (Usually 001)
# soa_ms: Stimulus display time before dynamic noise mask
# catch_trial: T/F whether this was a catch (no stimulus)
# detect_yn: T/F Did the participant report seeing a stimulus
# confidence: How confident is the participant in their detection response
# rt_prime_ms: Response time to guess the tilt of the prime
# prime_congruent: T/F Was the prime tilted in the same way as the stimulus
# prime_correct: Did the participant correctly guess the prime tilt

# ----------  Window Set-up ----------
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
mon = monitors.Monitor('testMonitor')  
mon.setWidth(73)          # physical width of your display in cm
mon.setDistance(75)       # your viewing distance in cm
mon.setSizePix([SCREEN_WIDTH,SCREEN_HEIGHT])
win = visual.Window(
    size=(SCREEN_WIDTH,SCREEN_HEIGHT), 
    fullscr=True, 
    monitor=mon,
    units='deg'
)
FRAME_MS, msPFstd, msPFmed = win.getMsPerFrame(nFrames=144)
prep_text = visual.TextStim(
    win,
    text="Preparing experiment...",
    color="white",
    height=1.0,
    wrapWidth=45
)
prep_text.draw()
win.flip()
win.recordFrameIntervals = True 
win.refreshThreshold    = FRAME_MS*1.5/1000
logging.console.setLevel(logging.WARNING)
mouse = event.Mouse(win=win)
kb    = keyboard.Keyboard()
print(f"Measured frame duration: {FRAME_MS:.3f} ms")
win.flip()

# ----------  Stimuli Initialization ----------
fix = visual.TextStim(win, text="+", height=0.8, color="white")
gabor = visual.GratingStim(win, tex="sin", mask="gauss",
                           size=5, sf=3, units="deg", contrast=PRIME_CONTRAST, opacity=PRIME_OPACITY)
detect_prompt = visual.TextStim(win, text="Did you see any stimulus?",
                               height=0.8, color="white")
prime_prompt = visual.TextStim(win, text="Tilt?  (a) ←  /  → (d)",
                               height=0.8, color="white")

conf_bar = visual.Rect(win, width=8, height=0.6, lineColor="white",
                       fillColor=None, pos=(0,-4))
conf_marker = visual.Rect(win, width=0.2, height=0.8,
                          lineColor=None, fillColor="white")

mask_stim = visual.ImageStim(win, image=None, units="deg", texRes=256,
                            size=10, interpolate=False, colorSpace='rgb')

# ----------  Helper functions ----------
def wait_ms(ms): core.wait(ms/1000.0)

def text_and_wait(text, ms=0):
    if not ms:
        text += "\n\nPress any key to continue."
    transition_text = visual.TextStim(
        win,
        text=text,
        color="white",
        height=1.0,
        wrapWidth=45
    )
    transition_text.draw()
    win.flip()
    if ms:
        wait_ms(ms)
    else:
        event.waitKeys()
        
def detection_yn():
    yes_button = visual.Rect(
        win=win, width=4, height=1.5, pos=(-3, -4),
        fillColor='darkgrey', lineColor='white'
    )
    no_button = visual.Rect(
        win=win, width=4, height=1.5, pos=(3, -4),
        fillColor='darkgrey', lineColor='white'
    )
    yes_text = visual.TextStim(
        win=win, text='Yes', pos=yes_button.pos,
        color='white', height=0.8
    )
    no_text = visual.TextStim(
        win=win, text='No', pos=no_button.pos,
        color='white', height=0.8
    )
    mouse.clickReset()

    while True:
        detect_prompt.draw()
        yes_button.fillColor = 'lightgrey' if yes_button.contains(mouse) else 'darkgrey'
        no_button.fillColor  = 'lightgrey' if no_button.contains(mouse)  else 'darkgrey'

        yes_button.draw(); yes_text.draw()
        no_button.draw();  no_text.draw()
        win.flip()

        if mouse.getPressed()[0]:
            if yes_button.contains(mouse):
                core.wait(0.2)
                return True
            if no_button.contains(mouse):
                core.wait(0.2)
                return False

def prime_rt():
    win.callOnFlip(kb.clock.reset)
    prime_prompt.draw()
    win.flip()
    prime_key = kb.waitKeys(keyList=["a","d"])[0]
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
            text=f"How confident are you in this response?: {pct} %",
            pos=(0,-2.5), height=0.6).draw()
        win.flip()
        if mouse.getPressed()[0]:
            core.wait(0.2)
            return pct

def draw_and_wait(stim, ms):
    n_frames = int(round(ms/FRAME_MS))
    for _ in range(n_frames):
        stim.draw()
        win.flip()

pix_per_cm = SCREEN_WIDTH / mon.getWidth()
cm_per_deg = 2 * mon.getDistance() * math.tan(math.radians(0.5))
pix_per_deg = pix_per_cm * cm_per_deg
SF = 3.0
BW = 6.0
def make_bandpass_noise(size, sf=SF, bw=BW, ppd=pix_per_deg):
    """
    Generate a single patch of band-pass noise:
    - size: in pixels (e.g. 256)
    - sf: center freq in cycles/deg
    - bw: full width at half max in c/deg
    - ppd: pixels per visual degree
    """
    # 1) white noise
    white = np.random.randn(size, size)
    F = fft.fftshift(fft.fft2(white))
    # 2) frequency grid (in cycles/deg)
    fx = fft.fftshift(fft.fftfreq(size, d=1/ppd))
    fy = fft.fftshift(fft.fftfreq(size, d=1/ppd))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    # 3) band-pass filter (Gaussian)
    sigma = bw / 2.0
    H = np.exp(-0.5 * ((R - sf) / sigma)**2)
    # 4) apply & invert
    F_bp = F * H
    noise_bp = np.real(fft.ifft2(fft.ifftshift(F_bp)))
    # 5) normalize to [–1,1]
    noise_bp /= np.max(np.abs(noise_bp))
    return noise_bp.astype('float32')

def draw_blank(ms):
    n_frames = max(1, int(round(ms / FRAME_MS)))
    for _ in range(n_frames):
        win.flip()

def draw_gabor(orientation, soa):
    gabor.ori = orientation
    gabor_frames = 1
    for _ in range(gabor_frames):
        gabor.draw()
        win.flip()
    blank_time = max(0, soa - FRAME_MS * gabor_frames)
    if blank_time:
        draw_blank(blank_time)

rng = np.random.default_rng(42)
N_NOISE_FRAMES = 200
NOISE_FRAMES = np.stack([make_bandpass_noise(256) for _ in range(N_NOISE_FRAMES)])
frame_counter = 0
def draw_dynamic_mask(ms):
    global frame_counter
    n_frames = max(1, int(round(ms/FRAME_MS)))
    for _ in range(n_frames):
        idx = frame_counter
        frame_counter = (frame_counter + 1) % N_NOISE_FRAMES
        # idx = rng.integers(N_NOISE_FRAMES) 
        mask_stim.image = NOISE_FRAMES[idx]
        mask_stim.draw()
        win.flip(clearBuffer=False)


# ----------  Practice ----------
PRACTICE_TEXT = (
    "~~ Welcome ~~\n\n"
    "You will complete 180 rapid trials over about 15-20 minutes.\n"
    "On each trial, keep your eyes on the center cross."
)
text_and_wait(PRACTICE_TEXT)
fix.draw()
PRACTICE_TEXT = visual.TextStim(
    win,
    text="\n\n\n\n\nThis is the fixation point, keep your eyes here for the experiment duration.",
    color="white",
    height=1.0,
    wrapWidth=45
)
PRACTICE_TEXT.draw()
win.flip()
while True:
    if event.getKeys():
        break
PRACTICE_TEXT = (
    "First, you'll see a field of dynamic noise.\n\n"
    "Press any key to see an example."
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = "\n\n\n\n\n\n\n\n\n\n\nPress any key to continue."
while True:
    if event.getKeys():  # any key
        break
    bp = make_bandpass_noise(256)
    mask_stim.image = bp
    mask_stim.draw()
    transition_text = visual.TextStim(
        win,
        text=PRACTICE_TEXT,
        color="white",
        height=1.0,
        wrapWidth=75
    )
    transition_text.draw()
    win.flip()
PRACTICE_TEXT = (
    "Sometimes a very brief tilted grating is hidden in that noise.\n\n"
    "Press any key to see what the stimulus looks like."
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = "\n\n\n\n\n\n\n\n\n\n\nPress any key to continue."
gabor.ori = 45 #random.choice(ORIENTATIONS)
gabor.draw()
transition_text = visual.TextStim(
    win,
    text=PRACTICE_TEXT,
    color="white",
    height=1.0,
    wrapWidth=75
)
transition_text.draw()
win.flip()
event.waitKeys()
PRACTICE_TEXT = (
    "Each trial has two questions:\n\n"
    "1.  Tilt choice (speeded):\n"
    "    • As soon as the arrows appear, press 'a' if you THINK the tilt was left, 'd' if right.\n"
    "    • Even if you saw nothing, guess quickly.\n\n"
    "2.  Visibility & confidence:\n"
    "    • Then report Yes/No if you saw any grating at all,\n"
    "    • and rate how sure you are."
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = (
    "Now we'll do a few practice trials to get you familiar with the timing.\n"
    "You will not receive any feedback about your responses."
)
text_and_wait(PRACTICE_TEXT)
    
PRACTICE_SOA = 91
draw_and_wait(fix, 750)
for p in range(N_PRACTICE):
    true_ori = random.choice(ORIENTATIONS)
    congruent = random.random() < PRIME_MATCH_PROB
    prime_ori = true_ori if congruent else (true_ori + 90) % 180
    draw_dynamic_mask(MASK_MS)
    draw_gabor(true_ori, PRACTICE_SOA)
    draw_dynamic_mask(MASK_MS)
    draw_gabor(prime_ori, PRIME_SOA)
    draw_dynamic_mask(REV_MASK_MS)
    draw_blank(np.random.randint(250, 550))
    prime_key, rt_ms = prime_rt()
    detected = detection_yn()
    conf = confidence_estimate()

    prime_correct = (prime_ori == 45 and prime_key == "d") or (prime_ori == 135 and prime_key == "a")
    print(PRACTICE_SOA, False, detected, conf, rt_ms, congruent, prime_correct, prime_ori)

    ITI = np.random.randint(650, 950)
    draw_and_wait(fix, ITI)


# ----------  Detection Practice ----------
PRACTICE_TEXT = (
    "Now, to help you calibrate your detection, you will be presented some trials which always have a stimulus present.\n"
    "Select whether the stimulus was tilted left (a) or right (d). You will be informed if you are incorrect.\n"
    "The task will get more challenging as it goes on."
)
text_and_wait(PRACTICE_TEXT)
soa_ms = MAX_SOA_MS
for s in range(N_DETECT_PRACTICE):
    catch_test = random.random() < 0.0
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, 750)
    draw_dynamic_mask(MASK_MS)
    draw_gabor(true_ori, soa_ms)
    draw_dynamic_mask(MASK_MS)
    # detected = detection_yn()
    key, rt_ms = prime_rt()
    correct = ((true_ori == 45 and key == 'd') or
               (true_ori == 135 and key == 'a'))
    if not correct:
        text_and_wait("Incorrect.", 500)
    soa_ms = max(MIN_SOA_MS, soa_ms-21)
    # print("Seen:", detected, "Stimulus:", not catch_test, "soa ms:", soa_ms)


# ----------  Simple staircase (1-up / 1-down) ----------
STAIRCASE_TEXT = (
    "We will now calibrate the difficulty.\n"
    "Each trial contains a very brief tilted grating: guess left (a) or right (d).\n"
    "The staircase will adjust the blank interval after the 1-frame target."
)
text_and_wait(STAIRCASE_TEXT)
start_frames = round(STAIR_START_MS / FRAME_MS)        #  ≈15 frames
quest = data.QuestHandler(
    startVal     = np.log10(start_frames),
    startValSd   = 0.5,
    pThreshold   = .75,    gamma=.50, beta=3.5, delta=.02,
    nTrials      = N_STAIR_TRIALS,
    minVal       = np.log10(max(1, round(MIN_SOA_MS / FRAME_MS))),
    maxVal       = np.log10(round(MAX_SOA_MS / FRAME_MS)),
)
soas_collected = []
performance = []
for log_frames in quest:
    frames   = max(1, int(round(10**log_frames)))      # 3.  Discretise
    soa = frames * FRAME_MS
    soas_collected.append(soa)
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, 750)
    draw_dynamic_mask(MASK_MS)
    draw_gabor(true_ori, soa)
    draw_dynamic_mask(MASK_MS)
    key, rt_ms = prime_rt()
    correct = ((true_ori == 45 and key == 'd') or
               (true_ori == 135 and key == 'a'))
    performance.append(correct)
    quest.addResponse(int(correct))

print(soas_collected)
print(performance)
logT = quest.quantile()
SC_FRAMES = max(1, int(round(10**logT)))
SC_SOA = SC_FRAMES * FRAME_MS
print("Threshold:", SC_SOA)


# ----------  Build main trial list ----------
offset_frames = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
SOA_frames = np.clip(SC_FRAMES + offset_frames, 1, None)
SOAS = SOA_frames * FRAME_MS
SOAS = np.clip(SOAS, MIN_SOA_MS, MAX_SOA_MS)
print(SOAS)
trials = []
for soa in SOAS:
    for _ in range(TRIALS_PER_SOA):
        trials.append({
            "soa": soa,
            "catch": False,
            "prime_congruent": random.random()<PRIME_MATCH_PROB,
            "prime": True
        })
    for _ in range(CATCH_PER_SOA):
        trials.append({
            "soa": soa,
            "catch": True,
            "prime_congruent": False,
            "prime": True
        })
        trials.append({
            "soa": soa,
            "catch": False,
            "prime_congruent": False,
            "prime": False
        })
random.shuffle(trials)

# ----------  MAIN LOOP ----------
BREAK_TEXT = (
    "You are halfway done! Please take a break, and continue when you are ready."
)
EXPERIMENT_TEXT = (
    "We will now begin the experiment.\n"
    "You must quickly select the tilt using 'a'/'d', and then answer whether you saw a stimulus along with your confidence.\n"
    "Not every trial will have a stimulus present. Please click NO if you did not see any.\n"
    "You will not receive any feedback about your responses.\n"
    "There will be a break halfway through."
)
draw_and_wait(fix, 750)
text_and_wait(EXPERIMENT_TEXT)
text_and_wait("Fixate on the cross for the entire experiment duration.")
for tidx, trial in enumerate(trials):
    true_ori = random.choice(ORIENTATIONS)
    prime_ori = true_ori if trial["prime_congruent"] else (true_ori+90)%180
    draw_dynamic_mask(MASK_MS)
    if not trial["catch"]:
        draw_gabor(true_ori, trial['soa'])
    else:
        draw_blank(trial["soa"])
    draw_dynamic_mask(MASK_MS)
    if trial["prime"]:
        draw_gabor(prime_ori, PRIME_SOA)
    else:
        draw_blank(FRAME_MS)
    draw_dynamic_mask(REV_MASK_MS)
    draw_blank(np.random.randint(250, 550))
    prime_key, rt_ms = prime_rt()
    detected = detection_yn()
    conf = confidence_estimate()
    
    if trial["prime"]: 
        prime_correct = ((prime_ori==45 and prime_key=="d") or (prime_ori==135 and prime_key=="a")) 
    else: 
        prime_correct = None
    print(trial["soa"], trial["catch"], detected, conf, rt_ms, trial["prime_congruent"], prime_correct)

    csv_writer.writerow([exp_info["Participant"], tidx, trial["soa"], 
                         trial["catch"], detected, conf, rt_ms, trial["prime"],
                         trial["prime_congruent"], prime_correct, SC_SOA])
    csv_file.flush()

    ITI = np.random.randint(650, 950)
    draw_and_wait(fix, ITI)

    if tidx == (len(trials) // 2):
        text_and_wait(BREAK_TEXT)
        draw_and_wait(fix, 750)

# ----------  tidy up ----------
csv_file.close()
visual.TextStim(win, text="Done!  Press ESC to quit.",
                height=1).draw(); win.flip()
event.waitKeys(keyList=["escape"])
dropped = win.nDroppedFrames
total   = len(win.frameIntervals)
if dropped:
    pct = dropped / total * 100
    print(f"\nWARNING: {dropped} dropped frames "
          f"({pct:0.2f}% of {total}) – check timing!")
else:
    print("\nNo dropped frames detected – timing OK.")
core.quit()
