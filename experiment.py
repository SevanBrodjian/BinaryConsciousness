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
from tqdm import tqdm
import pickle
from scipy.optimize import curve_fit


# ====================  HYPER-PARAMETERS  ========================
N_PRACTICE            = 12
N_DETECT_PRACTICE     = 23
N_STAIR_TRIALS        = 70
STAIR_STEP_MS         = 7
STAIR_START_MS        = 80
MIN_SOA_MS            = 0
MAX_SOA_MS            = 250
N_SOAS                = 9
GABOR_FRAMES          = 2
TRIALS_PER_SOA        = 16
PRIME_CONTRAST        = 0.175
PRIME_OPACITY         = 0.75
PRIME_SIZE            = 6
PRIME_SF              = 4
PRIME_MATCH_PROB      = 0.50
CATCH_PER_SOA         = 4
ORIENTATIONS          = [45, 135]
N_FRAMES_PER_NOISE    = 3
MASK_MS               = 100
PRIME_SOA             = 5
REV_MASK_MS           = 50
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
                     "detect_yn","confidence", "threshold"])
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
gabor = visual.GratingStim(win, tex="sin", mask="gauss", pos=(0,0),
                           size=PRIME_SIZE, sf=PRIME_SF, units="deg", contrast=PRIME_CONTRAST, opacity=PRIME_OPACITY)
detect_prompt = visual.TextStim(win, text="Did you see any stimulus?",
                               height=0.8, color="white")
prime_prompt = visual.TextStim(win, text="Tilt?  (a) ←  /  → (d)",
                               height=0.8, color="white")

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
    win.flip()
    win.callOnFlip(kb.clock.reset)
    prime_prompt.draw()
    win.flip()
    prime_key = kb.waitKeys(keyList=["a","d"])[0]
    rt_ms = kb.clock.getTime()*1000
    return prime_key, rt_ms

def confidence_estimate():
    mouse.clickReset()
    labels    = ['1: Guessing', '2: Somewhat\nUnsure', '3: Somewhat\nSure', '4: Completely\nSure']
    positions = [(-5, -4), (-1.67, -4), (1.67, -4), (5, -4)]
    buttons = []
    texts   = []
    for label, pos in zip(labels, positions):
        btn = visual.Rect(
            win=win,
            width=3, height=1.5,
            pos=pos,
            fillColor='darkgrey',
            lineColor='white'
        )
        txt = visual.TextStim(
            win=win,
            text=label,
            wrapWidth=5,
            pos=pos,
            color='white',
            height=0.35
        )
        buttons.append(btn)
        texts.append(txt)
    prompt = visual.TextStim(
        win=win,
        text="How confident are you in your response?",
        pos=(0, -2),
        color='white',
        height=0.8
    )
    while True:
        for btn, txt in zip(buttons, texts):
            if btn.contains(mouse):
                btn.fillColor = 'lightgrey'
            else:
                btn.fillColor = 'darkgrey'
            btn.draw()
            txt.draw()
        prompt.draw()
        win.flip()
        if mouse.getPressed()[0]:
            for idx, btn in enumerate(buttons):
                if btn.contains(mouse):
                    core.wait(0.2)
                    return idx + 1

def draw_and_wait(stim, ms):
    n_frames = int(round(ms/FRAME_MS))
    for _ in range(n_frames):
        stim.draw()
        win.flip()


n_gabors_per_frame = 30
gabor_size = 4.0
mask_size = PRIME_SIZE * 0.75
mask_gabor = visual.GratingStim(win, mask='gauss', size=gabor_size, sf=PRIME_SF, contrast=1.0, opacity=0.275)
def make_gabor_field():
    win.clearBuffer()
    for _ in range(n_gabors_per_frame):
        mask_gabor.ori = np.random.uniform(0, 180)
        mask_gabor.phase = np.random.rand()
        mask_gabor.pos = (np.random.uniform(-mask_size, mask_size),
                     np.random.uniform(-mask_size, mask_size))
        mask_gabor.sf = PRIME_SF * np.random.uniform(0.7, 1.3)
        mask_gabor.contrast = np.random.uniform(0.575, 0.775)
        mask_gabor.draw()
    return visual.BufferImageStim(win)

def draw_blank(ms):
    n_frames = max(1, int(round(ms / FRAME_MS)))
    for _ in range(n_frames):
        win.flip()

def draw_gabor(orientation, soa):
    gabor.ori = orientation
    for _ in range(GABOR_FRAMES):
        gabor.draw()
        win.flip()
    blank_time = max(0, soa - FRAME_MS * GABOR_FRAMES)
    if blank_time:
        draw_blank(blank_time)

rng = np.random.default_rng(42)
N_NOISE_FRAMES = 200
NOISE_FRAMES = np.stack([make_gabor_field() for _ in tqdm(range(N_NOISE_FRAMES))])
frame_counter = 0
def draw_dynamic_mask(ms):
    global frame_counter
    n_frames = max(1, int(round(ms/FRAME_MS)))
    for i in range(n_frames):
        if (i % N_FRAMES_PER_NOISE) == 0:
            frame_counter = (frame_counter + 1) % N_NOISE_FRAMES
        # idx = rng.integers(N_NOISE_FRAMES)
        NOISE_FRAMES[frame_counter].draw()
        win.flip(clearBuffer=False)


# ----------  Practice ----------
win.flip()
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
event.waitKeys()
PRACTICE_TEXT = (
    "Sometimes, there will be a brief stimulus shown at the start of the trial.\n"
    "The stimulus will be very brief, and may be challenging or impossible to see.\n"
    "Press any key to see what the stimulus looks like. "
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = "\n\n\n\n\n\n\n\n\n\n\n\nPress any key to continue."
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
    "After the stimulus (if there is one) there will be brief dynamic noise shown.\n\n"
    "Press any key to see an example."
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = "\n\n\n\n\n\n\n\n\n\n\nPress any key to continue."
i = 0
while True:
    if event.getKeys():  # any key
        break
    if (i % N_FRAMES_PER_NOISE) == 0:
        i = 0
        frame_counter = (frame_counter + 1) % N_NOISE_FRAMES
    NOISE_FRAMES[frame_counter].draw()
    transition_text = visual.TextStim(
        win,
        text=PRACTICE_TEXT,
        color="white",
        height=1.0,
        wrapWidth=75
    )
    transition_text.draw()
    win.flip()
    i += 1
PRACTICE_TEXT = (
    "Each trial will ask whether you saw the original stimulus or not.\n"
    "Report Yes/No if you saw any grating at all, and rate how sure you are."
)
text_and_wait(PRACTICE_TEXT)
PRACTICE_TEXT = (
    "Now we'll do a few practice trials to get you familiar with the timing.\n"
    "Not every trial will have a stimulus present.\n"
    "During practice only you will receive feedback about your responses."
)
text_and_wait(PRACTICE_TEXT)
    
PRACTICE_SOA = MAX_SOA_MS
draw_and_wait(fix, 750)
for p in range(N_PRACTICE):
    show = random.random() < 0.5
    true_ori = random.choice(ORIENTATIONS)
    congruent = random.random() < PRIME_MATCH_PROB
    prime_ori = true_ori if congruent else (true_ori + 90) % 180
    # draw_dynamic_mask(MASK_MS)
    draw_blank(100)
    if show:
        draw_gabor(true_ori, PRACTICE_SOA)
    else:
        draw_blank(PRACTICE_SOA)
    draw_dynamic_mask(MASK_MS)
    # draw_gabor(prime_ori, PRIME_SOA)
    # draw_dynamic_mask(REV_MASK_MS)
    # draw_blank(np.random.randint(250, 550))
    # prime_key, rt_ms = prime_rt()
    draw_blank(100)
    detected = detection_yn()
    conf = confidence_estimate()
    if (detected and show) or (not detected and not show):
        text_and_wait("Correct.", 500)
    else:
        text_and_wait("Incorrect.", 500)

    # prime_correct = (prime_ori == 45 and prime_key == "d") or (prime_ori == 135 and prime_key == "a")
    # print(PRACTICE_SOA, False, detected, conf, rt_ms, congruent, prime_correct, prime_ori)

    ITI = np.random.randint(650, 950)
    draw_and_wait(fix, ITI)


# ----------  Detection Practice ----------
PRACTICE_TEXT = (
    "Now, to help you calibrate your detection, you will be presented some trials which always have a stimulus present.\n"
    "Select whether the stimulus was tilted left (a) or right (d). You will be informed if you are incorrect.\n"
    "The task will get more challenging as it goes on. You will NOT have to report orientation on the real trials."
)
text_and_wait(PRACTICE_TEXT)
soa_ms = MAX_SOA_MS
for s in range(N_DETECT_PRACTICE):
    catch_test = random.random() < 0.0
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, 750)
    # draw_dynamic_mask(MASK_MS)
    draw_blank(100)
    draw_gabor(true_ori, soa_ms)
    draw_dynamic_mask(MASK_MS)
    # detected = detection_yn()
    draw_blank(100)
    key, rt_ms = prime_rt()
    correct = ((true_ori == 45 and key == 'd') or
               (true_ori == 135 and key == 'a'))
    if correct:
        text_and_wait("Correct.", 500)
    else:
        text_and_wait("Incorrect.", 500)
    soa_ms = max(MIN_SOA_MS, soa_ms-21)
    # print("Seen:", detected, "Stimulus:", not catch_test, "soa ms:", soa_ms)


# ---------------  STAIRCASE ---------------
STAIRCASE_TEXT = (
    "We will now calibrate the difficulty.\n"
    "Each trial contains a very brief tilted grating: guess left (a) or right (d).\n"
    "You will not receive feedback. The experiment will begin after this calibration."
)
text_and_wait(STAIRCASE_TEXT)

start_frames = round(STAIR_START_MS / FRAME_MS)
min_frames   = max(0, round(MIN_SOA_MS  / FRAME_MS))
max_frames   = round(MAX_SOA_MS  / FRAME_MS)
stair = StairHandler(
    startVal   = start_frames,   # starting SOA in frames
    stepType   = 'lin',          # linear steps
    stepSizes  = [2,1],              # +/-1 frame each step
    nTrials    = N_STAIR_TRIALS, # total trials
    minVal     = min_frames,     
    maxVal     = max_frames,
    nUp        = 1,              # 1 wrong ⇒ increase SOA
    nDown      = 2               # 1 correct ⇒ decrease SOA
)
soas_collected = []
performance    = []
old_gabor = GABOR_FRAMES
GABOR_FRAMES = 1

for frames in stair:
    frames = max(1, int(round(frames)))
    soa    = frames * FRAME_MS
    soas_collected.append(soa)
    true_ori = random.choice(ORIENTATIONS)
    draw_and_wait(fix, 750)
    draw_blank(50)
    draw_gabor(true_ori, soa)
    draw_dynamic_mask(MASK_MS)
    key, rt_ms = prime_rt()
    correct = ((true_ori == 45  and key == 'd') or
               (true_ori == 135 and key == 'a'))
    # correct = not correct
    # correct = detection_yn()
    print(soa, correct)
    performance.append(correct)
    stair.addResponse(int(correct))

x = np.array(soas_collected, dtype=float)
y = np.array(performance,     dtype=float)
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-(x - x0) / k))
p0 = [1.0, np.median(x), 10.0]                 # L, x0, k
(L, x0, k), _ = curve_fit(logistic, x, y, p0=p0, bounds=([0,0,0],[1,200,100]))
thr_SOA = x0 + k * np.log(3)
SC_FRAMES = int(round(thr_SOA / FRAME_MS))
# SC_FRAMES -= 2
SC_SOA    = SC_FRAMES * FRAME_MS
print(f"Logistic 75 % threshold ≈ {SC_SOA:.1f} ms  ({SC_FRAMES} frames)")
inp = input("Press ⏎ to accept, or type NEW threshold in ms: ").strip()
if inp:
    try:
        SC_SOA    = float(inp)
        SC_FRAMES = int(round(SC_SOA / FRAME_MS))
        print(f"→ Using override: {SC_SOA:.1f} ms  ({SC_FRAMES} frames)")
    except ValueError:
        print("!Invalid number, keeping auto threshold.")
else:
    print("→ Keeping auto threshold.")

# n_use = min(6, len(stair.reversalIntensities))
# mean_rev = np.mean(stair.reversalIntensities[-n_use:]) if n_use>0 else stair.intensities[-1]
# SC_FRAMES = int(round(mean_rev))
# if (SC_FRAMES < 4):
#     print(f"Clipping SC_FRAMES from {SC_FRAMES} to 4")
#     SC_FRAMES = 4
# SC_SOA = SC_FRAMES * FRAME_MS

GABOR_FRAMES = old_gabor

print("Collected SOAs (ms):", soas_collected)
print("Performance:", performance)
print(f"Estimated threshold: {SC_SOA:.1f} ms ({SC_FRAMES} frames)")


# ----------  Build main trial list ----------
offset_frames = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
SOA_frames = np.clip(SC_FRAMES + offset_frames, 0, None)
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
        # trials.append({
        #     "soa": soa,
        #     "catch": False,
        #     "prime_congruent": False,
        #     "prime": False
        # })
random.shuffle(trials)

# ----------  MAIN LOOP ----------
BREAK_TEXT = (
    "You are 1/3 done! Please take a 1-3min break, and continue when you are ready."
)
EXPERIMENT_TEXT = (
    "We will now begin the experiment.\n"
    "You must only answer whether you saw a stimulus or not, along with your confidence.\n"
    "Not every trial will have a stimulus present. Please click NO if you did not see any.\n"
    "You will not receive any feedback about your responses.\n"
    "There will be two breaks partway through."
)
draw_and_wait(fix, 750)
text_and_wait(EXPERIMENT_TEXT)
text_and_wait("Fixate on the cross for the entire experiment duration.")
for tidx, trial in enumerate(trials):
    true_ori = random.choice(ORIENTATIONS)
    prime_ori = true_ori if trial["prime_congruent"] else (true_ori+90)%180
    # draw_dynamic_mask(MASK_MS)
    if not trial["catch"]:
        draw_gabor(true_ori, trial['soa'])
    else:
        draw_blank(trial["soa"])
    draw_dynamic_mask(MASK_MS)
    # if trial["prime"]:
    #     draw_gabor(prime_ori, PRIME_SOA)
    # else:
    #     draw_blank(FRAME_MS * GABOR_FRAMES)
    # draw_dynamic_mask(REV_MASK_MS)
    # draw_blank(np.random.randint(250, 550))
    # prime_key, rt_ms = prime_rt()
    draw_blank(100)
    detected = detection_yn()
    conf = confidence_estimate()
    
    # if trial["prime"]: 
    #     prime_correct = ((prime_ori==45 and prime_key=="d") or (prime_ori==135 and prime_key=="a")) 
    # else: 
    #     prime_correct = None
    # print(trial["soa"], trial["catch"], detected, conf, rt_ms, trial["prime_congruent"], prime_correct)

    csv_writer.writerow([exp_info["Participant"], tidx, trial["soa"], 
                         trial["catch"], detected, conf, SC_SOA])
    csv_file.flush()

    ITI = np.random.randint(650, 950)
    draw_and_wait(fix, ITI)

    if (tidx == (len(trials) // 3)) or (tidx == ((len(trials) * 2) // 3)):
        text_and_wait(BREAK_TEXT)
        draw_and_wait(fix, 750)
        BREAK_TEXT = (
            "You are 2/3 done! Please take a 1-3min break, and continue when you are ready."
        )

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
          f"({pct:0.2f}% of {total}) - check timing!")
else:
    print("\nNo dropped frames detected - timing OK.")
core.quit()
