# `smart_scan_strategy.ipynb` — हर Cell की हर Line की विस्तृत व्याख्या (हिंदी में)

यह फाइल notebook के हर **code cell** को line-by-line (या जहाँ लाइनें आपस में जुड़ी हों, वहाँ छोटे logical group में) समझाती है — **क्या है, क्यों use किया, और कैसे काम करता है**। Markdown (theory वाले) cells को यहाँ दोबारा नहीं समझाया गया, क्योंकि वो पहले से ही explanation हैं।

Numbering यहाँ **"Code Cell 1, 2, 3..."** से है — notebook में markdown cells के बीच-बीच में हैं, इसलिए notebook के "In [ ]:" execution count से match नहीं करेगा, लेकिन ऊपर से नीचे क्रम वही है।

---

## Code Cell 1 — Imports और Config (Section 1 के बाद)

**यह cell क्या करता है:** सारी libraries import करता है, random seeds fix करता है (ताकि result हर बार same आए — reproducibility), और पूरे notebook में इस्तेमाल होने वाले constants (band count, episode length, sensor की properties) define करता है।

```python
import math
from dataclasses import dataclass
```
- `math`: Python का built-in गणित module। इसमें `math.exp` (exponential, यानी `e^x`), `math.ceil` (ऊपर की तरफ round करना) जैसे functions हैं जो हमें logistic curve और periodicity calculation में चाहिए।
- `from dataclasses import dataclass`: `dataclass` एक decorator है जो एक class को सिर्फ "data रखने वाला container" बना देता है — `__init__` खुद-ब-खुद बन जाता है, हमें manually लिखना नहीं पड़ता। हम इसे `PDW` (Pulse Descriptor Word) के लिए इस्तेमाल करेंगे।

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
```
- `numpy as np`: सारे numerical arrays (occupancy grid, belief values वगैरह) NumPy arrays हैं, क्योंकि ये तेज़ और vectorized operations देता है।
- `pandas as pd`: comparison results को एक साफ table (DataFrame) में दिखाने के लिए, जैसे Excel जैसी sheet।
- `matplotlib.pyplot as plt`: graphs/plots बनाने के लिए (waterfall diagram, training curve, bar chart वगैरह)।
- `torch`, `torch.nn`, `torch.nn.functional`: PyTorch — यह हमारा deep learning framework है, जिससे हम neural network (Actor-Critic) बनाएंगे और train करेंगे। `nn` में layers (`Linear`, आदि) हैं, `F` में loss functions (`mse_loss` वगैरह) हैं।

```python
RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)   # a few spots use the legacy global np.random API rather than a
                            # local Generator -- seed it too for full reproducibility.
```
- `RNG_SEED = 7`: एक fixed संख्या जिसे हम "seed" कहते हैं। Computer में randomness असल में pseudo-random होती है — अगर आप same seed दो, तो same "random" numbers हर बार मिलेंगे। इससे हमारा पूरा experiment **reproducible** (दोबारा वही result देने वाला) बनता है।
- `rng = np.random.default_rng(RNG_SEED)`: NumPy का नया, modern random-number generator object बनाया, इस seed के साथ। इसे `rng` नाम दिया — module-level एक general-purpose generator (हालाँकि ज़्यादातर जगह हम अपने local generators बनाते हैं ताकि हर scenario अलग हो)।
- `torch.manual_seed(RNG_SEED)`: PyTorch के अंदर जो random operations होते हैं (जैसे neural network के weights शुरू में random values से set होना), उनको भी इसी seed से control करते हैं।
- `np.random.seed(RNG_SEED)`: NumPy का पुराना ("legacy") global random API — जैसे सीधे `np.random.random()` या `np.random.shuffle()` बिना कोई generator object बनाए — यह भी इसी seed पर fix कर दिया। Comment में साफ लिखा है: कुछ जगह (जैसे PPO training के minibatch shuffle में) हम इसी legacy API का सीधे इस्तेमाल कर रहे हैं, तो उसे भी seed करना ज़रूरी है, नहीं तो पूरा notebook run करने पर हर बार अलग result आएगा।

```python
N_BANDS = 12                                             # receiver spectrum split into 12 bands
BAND_WIDTH_MHZ = 200.0
BAND_EDGES_MHZ = np.arange(N_BANDS + 1) * BAND_WIDTH_MHZ + 2000.0   # 2.0-4.4 GHz, 12 x 200 MHz bands
```
- `N_BANDS = 12`: हमारा receiver पूरे frequency spectrum को 12 बराबर हिस्सों ("bands") में बाँट देता है। असली दुनिया में एक receiver पूरे spectrum को एक साथ नहीं देख सकता (narrowband होता है), इसलिए उसे बारी-बारी बैंड्स scan करने पड़ते हैं — यही इस पूरे problem का मूल है।
- `BAND_WIDTH_MHZ = 200.0`: हर बैंड 200 MHz चौड़ा है।
- `BAND_EDGES_MHZ = np.arange(N_BANDS + 1) * BAND_WIDTH_MHZ + 2000.0`: `np.arange(13)` से `[0,1,2,...,12]` बनता है, हर को 200 से multiply करके 2000 add किया — यानी बैंड्स की सीमाएँ `[2000, 2200, 2400, ..., 4400]` MHz बनती हैं। मतलब कुल spectrum 2.0 GHz से 4.4 GHz तक, 12 बैंड्स में बँटा हुआ। `N_BANDS+1` इसलिए क्योंकि 12 बैंड्स की 13 सीमाएँ (edges) होती हैं (जैसे 12 कमरों की 13 दीवारें)।

```python
EPISODE_SLOTS = 150                                       # receiver dwell slots per episode
SLOT_DURATION_US = 2000.0                                 # one scan decision every 2 ms
```
- `EPISODE_SLOTS = 150`: एक "episode" (यानी एक पूरा simulation-run, जैसे 10 सेकंड का एक अभ्यास) में receiver को 150 बार scan-decision लेने का मौका मिलता है — हर बार सिर्फ एक बैंड चुनकर देख सकता है।
- `SLOT_DURATION_US = 2000.0`: हर एक decision-slot 2000 microseconds (= 2 milliseconds) का है — यानी receiver हर 2ms में एक नया फैसला लेता है कि किस बैंड को देखना है।

```python
PD_SENSOR = 0.9        # scheduler's *assumed* detection probability (belief-update model)
PFA_SENSOR = 0.05       # false-alarm probability, independent of signal amplitude
SENSITIVITY_DB = 8.0    # true sensor operating point: amplitude (dB) at which true Pd = 50%
SLOPE_DB = 3.0          # steepness of the true Pd(SNR) sigmoid around the sensitivity point
```
- `PD_SENSOR = 0.9`: **Pd** मतलब "Probability of Detection" — अगर band में सच में कोई emitter ON है, तो receiver को उसे पकड़ने (detect करने) की कितनी संभावना है। यहाँ `0.9` scheduler का सिर्फ़ **मान लिया हुआ (assumed)** मान है, असली physics नहीं — इसे belief update में इस्तेमाल करेंगे।
- `PFA_SENSOR = 0.05`: **Pfa** मतलब "Probability of False Alarm" — अगर band असल में OFF है, फिर भी receiver ग़लती से "detect हुआ" बोल दे, उसकी संभावना।
- `SENSITIVITY_DB = 8.0`: यह असली (true) sensor की एक property है — signal की amplitude (dB में) कितनी होनी चाहिए ताकि detection की संभावना ठीक 50% हो। यह वो "sensitivity" figure-of-merit है जो problem statement में माँगा गया है।
- `SLOPE_DB = 3.0`: यह बताता है कि detection curve (Pd बनाम signal-strength graph) कितना "तीखा" या "ढलान वाला" है — यह logistic (S-shaped) curve की steepness है।

```python
print(f"{N_BANDS} bands x {EPISODE_SLOTS} slots, spectrum "
      f"{BAND_EDGES_MHZ[0]/1000:.2f}-{BAND_EDGES_MHZ[-1]/1000:.2f} GHz, "
      f"slot = {SLOT_DURATION_US/1000:.1f} ms")
```
- यह सिर्फ़ एक confirmation print है — जो values ऊपर set कीं, वो सही दिख रही हैं या नहीं, इसे human-readable तरीके से दिखाता है (जैसे "12 bands x 150 slots, spectrum 2.00-4.40 GHz, slot = 2.0 ms")। `f"..."` एक **f-string** है — इसके अंदर `{...}` में लिखा expression सीधे calculate होकर string में बैठ जाता है। `:.2f` का मतलब है "2 decimal places तक round करके दिखाओ"।

---

## Code Cell 2 — PDW schema और Emitter Classes (Section 2 के बाद)

**यह cell क्या करता है:** असली दुनिया के radar-pulse का data-format (`PDW`) define करता है, और चार तरह के emitter (signal भेजने वाले source) के behavior-model बनाता है, फिर `make_random_scenario` से एक randomized मिश्रण (mix) तैयार करता है।

```python
@dataclass
class PDW:
    toa_us: float
    freq_mhz: float
    pw_us: float
    aoa_deg: float
    amp_db: float
    emitter_id: int   # ground truth only -- never visible to the scheduler
```
- `@dataclass`: ऊपर बताया — automatic `__init__` बना देता है।
- `class PDW:`: एक pulse को represent करने वाला structure। असली Turing Synthetic Radar Dataset (TSRD) भी बिल्कुल इसी 5 fields के format में data रखता है, इसलिए हमारा synthetic data भी उसी format में है (ताकि बाद में असली data पर switch करना आसान हो — Section 9)।
  - `toa_us`: Time of Arrival — pulse कब आया (microseconds में)।
  - `freq_mhz`: Centre Frequency — pulse किस frequency पर था (MHz में)।
  - `pw_us`: Pulse Width — pulse कितनी देर चला (microseconds)।
  - `aoa_deg`: Angle of Arrival — किस दिशा से आया (degrees में, spatial जानकारी)।
  - `amp_db`: Amplitude — signal कितना तेज़/strong था (decibel में)।
  - `emitter_id: int`: यह बताता है कि यह pulse **असल में किस emitter से आया** — लेकिन comment में साफ है: यह सिर्फ हमारी "ground truth" (सच्चाई) के लिए है, ताकि हम बाद में evaluation कर सकें; scheduler को यह जानकारी कभी नहीं दी जाती — यह "cheat" नहीं करता।

```python
class Emitter:
    kind = "base"

    def __init__(self, emitter_id, band, pw_us=1.0, amp_db=8.0):
        self.emitter_id = emitter_id
        self.band = band          # primary/default band (non-hopping emitters)
        self.pw_us = pw_us
        self.amp_db = amp_db

    def simulate(self, n_slots, seed):
        # Return (on_mask[n_slots] bool, band_seq[n_slots] int)
        raise NotImplementedError
```
- `class Emitter:`: यह एक **base class** (सबका "माता-पिता" class) है — बाकी सारे emitter types इसी से inherit (विरासत में) आगे बनेंगे।
- `kind = "base"`: class-level attribute, सिर्फ debugging/labeling के लिए कि यह किस "प्रकार" का emitter है।
- `__init__`: constructor — जब भी कोई नया emitter बनेगा, यह चलता है। `emitter_id` (पहचान संख्या), `band` (यह किस बैंड में बैठा है), `pw_us` (pulse width), `amp_db` (signal strength) — इन्हें `self.` के ज़रिए object के अंदर save कर लिया, ताकि बाद में methods इन्हें use कर सकें।
- `simulate(self, n_slots, seed)`: यह एक "placeholder" method है। `raise NotImplementedError` का मतलब है — "अगर कोई इस base class को directly इस्तेमाल करने की कोशिश करे, तो error आएगा, क्योंकि हर बच्चा-class (subclass) को अपनी *खुद की* `simulate` लिखनी होगी"। यह एक standard Object-Oriented Programming pattern है जिसे "abstract method" कहते हैं।

```python
class FixedEmitter(Emitter):
    kind = "fixed"

    def simulate(self, n_slots, seed):
        return np.ones(n_slots, dtype=bool), np.full(n_slots, self.band, dtype=int)
```
- `class FixedEmitter(Emitter):`: `Emitter` से inherit किया — यानी `__init__` वही base वाला use होगा।
- `simulate`: सबसे सरल emitter — **हमेशा ON** रहता है (लगातार transmit करता है, जैसे कोई beacon)।
  - `np.ones(n_slots, dtype=bool)`: `n_slots` लंबाई का array बनाया, जिसमें हर value `True` है — यानी हर slot में यह emitter ON है।
  - `np.full(n_slots, self.band, dtype=int)`: उतनी ही लंबाई का array, जिसमें हर जगह इसका अपना fixed `band` number है (क्योंकि यह frequency नहीं बदलता)।
  - `return` में दो चीज़ें एक साथ लौटा रहे हैं — Python में यह एक **tuple** बन जाता है `(on_mask, band_seq)`।

```python
class BurstyEmitter(Emitter):
    # Fixed-frequency emitter switching on/off via a 2-state Markov chain.
    kind = "bursty"

    def __init__(self, *args, p_off_to_on=0.08, p_on_to_on=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_off_to_on = p_off_to_on
        self.p_on_to_on = p_on_to_on
```
- यह emitter बीच-बीच में ON/OFF होता रहता है (जैसे कोई intermittent search-radar)।
- `*args, **kwargs`: Python का तरीका है "बाकी सारे arguments जो भी आएँ, उन्हें आगे pass कर दो" — बिना उनका नाम जाने। यहाँ `p_off_to_on` (OFF से ON होने की probability) और `p_on_to_on` (ON बना रहने की probability) अलग से लिए हैं, बाकी सब (`emitter_id`, `band`, वगैरह) `*args`/`**kwargs` में base class को दे दिए।
- `super().__init__(*args, **kwargs)`: parent class (`Emitter`) का constructor बुलाया, ताकि `emitter_id`, `band` वगैरह properly set हो जाएँ।
- `self.p_off_to_on = p_off_to_on` / `self.p_on_to_on = p_on_to_on`: यह दोनों probability values आगे इस्तेमाल के लिए save कर लीं।

```python
    def simulate(self, n_slots, seed):
        r = np.random.default_rng(seed)
        on = np.zeros(n_slots, dtype=bool)
        state = r.random() < 0.3
        for t in range(n_slots):
            on[t] = state
            p = self.p_on_to_on if state else self.p_off_to_on
            state = r.random() < p
        return on, np.full(n_slots, self.band, dtype=int)
```
- `r = np.random.default_rng(seed)`: इस specific emitter के लिए एक अलग random generator, दिए गए `seed` के साथ — ताकि पूरा simulation reproducible रहे, फिर भी हर emitter का randomness अलग-अलग हो।
- `on = np.zeros(n_slots, dtype=bool)`: शुरू में सारे slots को `False` (OFF) से भर दिया, बाद में भरेंगे।
- `state = r.random() < 0.3`: शुरुआत में 30% संभावना है कि emitter ON state से शुरू हो (`r.random()` 0 से 1 के बीच एक random संख्या देता है; अगर वह 0.3 से कम है तो `True`)।
- `for t in range(n_slots):`: हर time-slot के लिए एक loop।
  - `on[t] = state`: अभी का state जो भी है, उसे array में भर दो।
  - `p = self.p_on_to_on if state else self.p_off_to_on`: यह एक **conditional expression** है (एक-लाइन का if-else) — अगर अभी ON है तो अगली probability `p_on_to_on` इस्तेमाल होगी (ON बना रहने की), अगर OFF है तो `p_off_to_on` (ON में बदलने की) — यही **2-state Markov chain** का मतलब है: अगली state सिर्फ अभी की state पर depend करती है।
  - `state = r.random() < p`: अगली time-slot के लिए नया state निकाला।
- आख़िर में band वही fixed रहता है (जैसे `FixedEmitter` में)।

```python
class PeriodicScanEmitter(Emitter):
    # Spatially-scanning emitter: illuminates the receiver for `dwell_slots` once every
    # `scan_period_slots` (e.g. a rotating-antenna radar beam sweeping past our bearing).
    kind = "periodic"

    def __init__(self, *args, scan_period_slots=20, dwell_slots=3, phase=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_period_slots = scan_period_slots
        self.dwell_slots = dwell_slots
        self.phase = phase
```
- यह वह emitter है जो problem statement में "**spatially scanning emitter**" कहलाता है — जैसे एक घूमने वाला (rotating) antenna radar, जो सिर्फ तब हमें "दिखता" है जब उसका beam घूमते-घूमते हमारी दिशा में आता है।
- `scan_period_slots`: कितने slots में एक बार पूरा चक्कर (rotation) पूरा होता है।
- `dwell_slots`: चक्कर के दौरान कितने slots तक beam हमारी तरफ़ रहता है (यानी कितनी देर तक हमें "दिख" सकता है)।
- `phase`: चक्कर की शुरुआत कहाँ से हो रही है (एक offset, ताकि सारे emitters synced ना हों)।

```python
    def simulate(self, n_slots, seed):
        t = np.arange(n_slots)
        on = ((t - self.phase) % self.scan_period_slots) < self.dwell_slots
        return on, np.full(n_slots, self.band, dtype=int)
```
- `t = np.arange(n_slots)`: `[0, 1, 2, ..., n_slots-1]` — हर slot का index।
- `(t - self.phase) % self.scan_period_slots`: `%` का मतलब **modulo** (remainder / शेषफल) — यह हर slot के लिए बताता है कि "अभी scan-cycle के कौन से बिंदु पर हैं" (0 से `scan_period_slots-1` के बीच, फिर दोबारा 0 से शुरू — यही चक्र/periodicity को represent करता है)।
- `< self.dwell_slots`: अगर वह बिंदु `dwell_slots` से कम है, तो मतलब हम अभी "illumination window" में हैं — यानी ON। यह पूरी line **vectorized** है (पूरे array पर एक साथ काम करती है, कोई for-loop नहीं चाहिए) — NumPy की ताकत यही है।
- नतीजा `on` एक boolean array है जिसमें हर `scan_period_slots` के बाद `dwell_slots` लंबाई की एक "True" खिड़की (window) दोहराती रहती है — यही periodic (आवर्ती) व्यवहार है।

```python
class FrequencyHopEmitter(Emitter):
    # Frequency-agile emitter: hops between `bands` every `hop_slots` slots.
    kind = "hopper"

    def __init__(self, emitter_id, bands, hop_slots=4, on_prob=0.9, **kwargs):
        super().__init__(emitter_id, band=bands[0], **kwargs)
        self.bands = list(bands)
        self.hop_slots = hop_slots
        self.on_prob = on_prob
```
- यह "**frequency-agile**" emitter है — यानी यह एक बैंड पर नहीं टिकता, बल्कि कई बैंड्स के बीच "उछलता" (hop करता) रहता है।
- `bands`: उन बैंड्स की list जिनके बीच यह hop करेगा।
- `hop_slots`: कितने slots के बाद अगला hop होगा।
- `on_prob`: हर hop पर transmit करने की संभावना (हमेशा transmit नहीं करता, कभी-कभी silent भी रह सकता है)।
- `super().__init__(emitter_id, band=bands[0], **kwargs)`: parent को बुलाया, default band के तौर पर पहला बैंड (`bands[0]`) दे दिया (हालाँकि असली में यह हर slot पर बदलता रहेगा)।
- `self.bands = list(bands)`: दी गई bands की एक अपनी copy (list के रूप में) save कर ली।

```python
    def simulate(self, n_slots, seed):
        r = np.random.default_rng(seed)
        on = r.random(n_slots) < self.on_prob
        n_hops = n_slots // self.hop_slots + 1
        hop_choices = r.integers(0, len(self.bands), size=n_hops)
        band_seq = np.array([self.bands[hop_choices[t // self.hop_slots]] for t in range(n_slots)])
        return on, band_seq
```
- `on = r.random(n_slots) < self.on_prob`: एक साथ `n_slots` random numbers निकाले (0-1 के बीच), और हर एक को `on_prob` से compare किया — इससे एक साथ पूरा on/off pattern बन गया (हर slot independent, कोई Markov memory नहीं यहाँ)।
- `n_hops = n_slots // self.hop_slots + 1`: कुल कितने "hop-segments" चाहिए होंगे — `//` का मतलब है integer division (पूरा भाग, दशमलव काटकर)। `+1` सुरक्षा के लिए है ताकि आख़िरी अधूरे segment के लिए भी जगह रहे।
- `hop_choices = r.integers(0, len(self.bands), size=n_hops)`: हर hop-segment के लिए randomly एक बैंड-index चुना (0 से `len(bands)-1` के बीच)।
- `band_seq = np.array([self.bands[hop_choices[t // self.hop_slots]] for t in range(n_slots)])`: यह एक **list comprehension** है (एक-लाइन में loop लिखने का तरीका)। हर slot `t` के लिए, पहले पता लगाया कि यह किस hop-segment में आता है (`t // self.hop_slots`), फिर उस segment के लिए पहले से चुना हुआ बैंड-index निकाला (`hop_choices[...]`), और फिर असली बैंड-नंबर निकाला (`self.bands[...]`)। नतीजा: हर slot के लिए बैंड बताने वाला पूरा array।

```python
def make_random_scenario(n_bands=N_BANDS, n_slots=EPISODE_SLOTS, seed=0):
    # Randomize the emitter mix every call -- this is the 'domain randomization' that
    # lets a scheduler trained here generalize instead of memorizing one fixed layout,
    # matching the 'no prior reliable intelligence' constraint of the problem.
    r = np.random.default_rng(seed)
    emitters, eid = [], 0
```
- यह function एक पूरा "scenario" (यानी emitters का एक random मिश्रण) बनाता है।
- Comment बहुत ज़रूरी है: हम **domain randomization** कर रहे हैं — मतलब हर बार अलग emitters (अलग संख्या, अलग गुण) generate होते हैं, ताकि हमारा scheduler किसी एक ख़ास layout को "रट" (memorize) ना ले, बल्कि एक general strategy सीखे — यह बिलकुल problem statement की माँग से मेल खाता है: "बिना पहले से पता एमिटर्स के"।
- `r = np.random.default_rng(seed)`: इस scenario के लिए एक seeded generator।
- `emitters, eid = [], 0`: खाली list (emitters इकट्ठा करने के लिए) और `eid` (emitter-id counter, हर नए emitter को अगली संख्या मिलेगी) — यह एक साथ दो चीज़ें assign कर रहा है (tuple unpacking)।

```python
    for _ in range(r.integers(1, 3)):
        band = int(r.integers(0, n_bands))
        emitters.append(BurstyEmitter(eid, band, amp_db=float(r.uniform(2, 15)),
                                       p_off_to_on=float(r.uniform(0.04, 0.15)),
                                       p_on_to_on=float(r.uniform(0.7, 0.95))))
        eid += 1
```
- `for _ in range(r.integers(1, 3)):`: `r.integers(1, 3)` से 1 या 2 (randomly) निकलता है — यानी 1 या 2 `BurstyEmitter` बनेंगे। `_` का मतलब है "loop variable का मान हमें चाहिए नहीं, बस इतनी बार दोहराना है"।
- `band = int(r.integers(0, n_bands))`: randomly कोई एक बैंड चुना (0 से 11 के बीच), `int()` से पक्का किया कि यह Python का normal integer है (NumPy का नहीं, ताकि आगे कहीं दिक्कत ना हो)।
- `amp_db=float(r.uniform(2, 15))`: signal strength को 2 से 15 dB के बीच randomly चुना।
- `p_off_to_on=...`, `p_on_to_on=...`: on/off होने की probabilities को भी एक range में randomly चुना, ताकि हर emitter थोड़ा अलग "व्यवहार" करे।
- `emitters.append(...)`: नया `BurstyEmitter` object बनाकर list में जोड़ दिया।
- `eid += 1`: अगले emitter के लिए id एक बढ़ा दी।

```python
    for _ in range(r.integers(1, 3)):
        band = int(r.integers(0, n_bands))
        emitters.append(PeriodicScanEmitter(eid, band, amp_db=float(r.uniform(2, 15)),
                                             scan_period_slots=int(r.integers(12, 30)),
                                             dwell_slots=int(r.integers(2, 4)),
                                             phase=int(r.integers(0, n_slots))))
        eid += 1
```
- यही पैटर्न, लेकिन अब 1-2 `PeriodicScanEmitter` (rotating-antenna टाइप) बनते हैं, हर एक का rotation-period (12-29 slots), dwell-window (2-3 slots), और शुरुआती phase randomly चुना जाता है — ताकि सारे periodic emitters अलग-अलग समय पर "दिखें"।

```python
    if r.random() < 0.8:
        n_hop_bands = int(r.integers(3, 6))
        bands = list(r.choice(n_bands, size=n_hop_bands, replace=False))
        emitters.append(FrequencyHopEmitter(eid, bands, amp_db=float(r.uniform(2, 15)),
                                             hop_slots=int(r.integers(2, 6)),
                                             on_prob=float(r.uniform(0.6, 0.95))))
        eid += 1
```
- `if r.random() < 0.8:`: 80% संभावना है कि इस scenario में एक frequency-hopping emitter भी शामिल हो (हमेशा नहीं — कभी-कभी scenario में यह टाइप गायब भी रहे, ताकि विविधता बनी रहे)।
- `n_hop_bands = int(r.integers(3, 6))`: यह emitter 3, 4, या 5 अलग-अलग बैंड्स के बीच hop करेगा।
- `bands = list(r.choice(n_bands, size=n_hop_bands, replace=False))`: 12 बैंड्स में से बिना दोहराव (`replace=False`) के randomly `n_hop_bands` बैंड चुने।

```python
    return emitters
```
- अंत में सारे बनाए गए emitters की पूरी list लौटा दी — यह पूरा एक "scenario" है।

---

## Code Cell 3 — Ground Truth Generator + Sanity Check (Section 2 का बाकी हिस्सा)

**यह cell क्या करता है:** emitters की list लेकर असली "truth" data बनाता है (कब कौन सा बैंड ON था) और साथ ही असली PDW pulses भी generate करता है — फिर verify करता है कि दोनों आपस में match करते हैं।

```python
def build_truth_and_pdws(emitters, n_bands=N_BANDS, n_slots=EPISODE_SLOTS,
                          slot_us=SLOT_DURATION_US, band_edges=BAND_EDGES_MHZ, seed=0):
    # Ground-truth generator: turns an emitter list into (a) a slot-level occupancy
    # grid, (b) a per-band 'owner' emitter-id grid, (c) a per-band amplitude grid, and
    # (d) a PDW stream in the exact TSRD field layout.
    truth = np.zeros((n_bands, n_slots), dtype=bool)
    owner = np.full((n_bands, n_slots), -1, dtype=int)
    amp_grid = np.full((n_bands, n_slots), -np.inf)
    pdws = []
    r = np.random.default_rng(seed + 10_000)
```
- यह function चार चीज़ें बनाकर देगा — comment में साफ लिखा है:
  1. `truth`: एक 2D grid (बैंड × समय) जिसमें True/False है कि उस वक़्त वह बैंड ON था या नहीं।
  2. `owner`: वही grid, लेकिन बताता है कि उस समय-बैंड में **कौन सा emitter** था (उसका id)।
  3. `amp_grid`: वही grid, लेकिन signal की strength (amplitude) बताता है।
  4. `pdws`: असली pulse-list, TSRD जैसे format में।
- `truth = np.zeros((n_bands, n_slots), dtype=bool)`: शुरू में सब कुछ `False` (OFF) से भरा — आकार है `(12, 150)` यानी 12 बैंड × 150 slots की एक matrix।
- `owner = np.full((n_bands, n_slots), -1, dtype=int)`: शुरू में सब जगह `-1` भर दिया — यह "कोई emitter नहीं" को represent करता है (क्योंकि असली emitter-ids हमेशा 0 या उससे ऊपर होंगी)।
- `amp_grid = np.full((n_bands, n_slots), -np.inf)`: शुरू में सब जगह "minus infinity" (यानी कोई signal नहीं, सबसे कमज़ोर संभव मान)।
- `pdws = []`: खाली list, बाद में pulses इसमें जुड़ेंगी।
- `r = np.random.default_rng(seed + 10_000)`: एक अलग seeded generator, सिर्फ़ pulse-timing/frequency की randomness के लिए (मुख्य `seed` से थोड़ा अलग रखा — `+10_000` — ताकि emitter-behavior की randomness और pulse-detail की randomness आपस में टकराएँ नहीं)।

```python
    for e in emitters:
        on, band_seq = e.simulate(n_slots, seed=seed * 1000 + e.emitter_id)
        for t in range(n_slots):
            if not on[t]:
                continue
            b = int(band_seq[t])
            truth[b, t] = True
            amp_grid[b, t] = max(amp_grid[b, t], e.amp_db)
            if owner[b, t] == -1:
                owner[b, t] = e.emitter_id
```
- `for e in emitters:`: हर emitter के लिए एक-एक करके।
- `on, band_seq = e.simulate(...)`: उस emitter का पूरा on/off pattern और band-sequence निकाला। `seed=seed*1000 + e.emitter_id`: हर emitter को उसका अपना अलग (मगर deterministic — यानी हमेशा एक ही तरह repeat होने वाला) seed मिलता है, ताकि दो अलग emitters का randomness आपस में overlap ना करे।
- `for t in range(n_slots):`: हर time-slot में जाकर देखते हैं।
  - `if not on[t]: continue`: अगर इस slot में emitter OFF है, तो कुछ मत करो, अगली slot पर जाओ (`continue` = loop का बाकी हिस्सा skip)।
  - `b = int(band_seq[t])`: इस slot में emitter किस बैंड में है।
  - `truth[b, t] = True`: उस (बैंड, समय) पर truth grid में True लिख दिया — यानी "यहाँ कुछ transmit हो रहा है"।
  - `amp_grid[b, t] = max(amp_grid[b, t], e.amp_db)`: अगर उस जगह पहले से ही (किसी और emitter की वजह से) कोई amplitude value है, तो उसे तभी बदलो जब यह नया emitter उससे ज़्यादा तेज़ है — `max()` लेने का physical मतलब है कि अगर दो signals एक साथ आ जाएँ, तो जो ज़्यादा तेज़ है वही dominant/visible रहेगा।
  - `if owner[b, t] == -1: owner[b, t] = e.emitter_id`: सिर्फ़ तभी owner भरो जब वहाँ पहले से कोई और claim नहीं कर चुका (यानी पहला-आया-पहला-पाया की तरह) — यह एक जानी-बूझी simplification है: अगर दो emitters कभी टकरा जाएँ (rare case), तो सिर्फ़ पहला वाला "मालिक" गिना जाएगा।

```python
            band_lo, band_hi = band_edges[b], band_edges[b + 1]
            toa = t * slot_us + r.uniform(0, max(slot_us - e.pw_us, 1.0))
            freq = r.uniform(band_lo + 5, band_hi - 5)
            aoa = r.uniform(0, 360)
            amp = e.amp_db + r.normal(0, 1.0)
            pdws.append(PDW(toa, freq, e.pw_us, aoa, amp, e.emitter_id))
```
- `band_lo, band_hi = band_edges[b], band_edges[b + 1]`: इस बैंड की शुरुआत और अंत की frequency।
- `toa = t * slot_us + r.uniform(0, max(slot_us - e.pw_us, 1.0))`: pulse का "Time of Arrival" — इस slot की शुरुआत (`t * slot_us`) के बाद, slot के अंदर कहीं randomly (ताकि pulse बिल्कुल slot के किनारे पर ना बैठे, इसलिए `pw_us` घटाया, और कम-से-कम 1.0 रखा ताकि range कभी negative ना हो)।
- `freq = r.uniform(band_lo + 5, band_hi - 5)`: pulse की असली frequency — बैंड के अंदर, किनारों से 5 MHz दूर रखा (margin), ताकि बाद में जब हम frequency से वापस बैंड-number निकालें, तो कोई confusion ना हो।
- `aoa = r.uniform(0, 360)`: दिशा (angle of arrival) — randomly 0 से 360 डिग्री (चूँकि हमारे मॉडल में इसका कोई खास physical इस्तेमाल नहीं, बस schema-completeness के लिए)।
- `amp = e.amp_db + r.normal(0, 1.0)`: emitter की base amplitude में थोड़ा-सा random noise जोड़ा (Gaussian/normal distribution, mean 0, std-dev 1.0) — असली दुनिया में signal strength कभी बिल्कुल constant नहीं रहती।
- `pdws.append(PDW(...))`: एक नया PDW object बनाकर list में जोड़ दिया — यही असली pulse का record है।

```python
    pdws.sort(key=lambda p: p.toa_us)
    return truth, owner, amp_grid, pdws
```
- `pdws.sort(key=lambda p: p.toa_us)`: सारे pulses को समय (Time of Arrival) के हिसाब से बढ़ते क्रम में sort कर दिया — `lambda p: p.toa_us` एक बिना-नाम का छोटा function है जो बताता है कि किस चीज़ के आधार पर sort करना है। असली receiver को pulses हमेशा time-order में ही मिलते हैं, इसलिए यह ज़रूरी है।
- चारों चीज़ें (truth grid, owner grid, amplitude grid, pdws list) वापस लौटा दीं।

```python
def pulses_to_occupancy(pdws, n_bands=N_BANDS, n_slots=EPISODE_SLOTS,
                         slot_us=SLOT_DURATION_US, band_edges=BAND_EDGES_MHZ):
    # Reconstruct a slot-level occupancy grid from a raw PDW stream. Works identically
    # on synthetic PDWs or real TSRD PDWs -- this is the dataset-agnostic bridge.
    grid = np.zeros((n_bands, n_slots), dtype=bool)
    for p in pdws:
        t = int(p.toa_us // slot_us)
        if t >= n_slots:
            continue
        b = int(np.searchsorted(band_edges, p.freq_mhz, side="right") - 1)
        if 0 <= b < n_bands:
            grid[b, t] = True
    return grid
```
- यह function **उल्टी दिशा** में काम करता है: pulses (PDWs) से वापस occupancy-grid बनाना। यह इसलिए महत्वपूर्ण है क्योंकि यही function असली TSRD data पर भी बिना बदलाव के चल जाएगा (Section 9) — इसे कभी नहीं पता कि pulses synthetic हैं या असली।
- `grid = np.zeros(...)`: खाली grid, सब False।
- `for p in pdws:`: हर pulse के लिए।
  - `t = int(p.toa_us // slot_us)`: pulse किस time-slot में गिरता है — `//` से integer division से slot-number निकाला।
  - `if t >= n_slots: continue`: अगर किसी वजह से यह episode की सीमा से बाहर गिर गया (edge-case), तो skip कर दो।
  - `b = int(np.searchsorted(band_edges, p.freq_mhz, side="right") - 1)`: `np.searchsorted` एक sorted array में value की सही जगह ढूँढता है (binary search जैसा, तेज़)। यहाँ pulse की frequency को बैंड-edges में ढूँढा — कौन सी दो सीमाओं के बीच यह गिरती है, वही उसका बैंड-नंबर है। `side="right"` और `-1` की वजह से boundary-values सही बैंड में गिरें (off-by-one गलती से बचने के लिए)।
  - `if 0 <= b < n_bands: grid[b,t] = True`: अगर बैंड-नंबर valid range में है, तो grid में True mark कर दो।
- यह function `truth` जैसा ही grid वापस देता है, लेकिन इसे pulses से **दोबारा बनाकर** — इसलिए दोनों को compare करके देख सकते हैं कि हमारी generation-pipeline सही है या नहीं (नीचे sanity check में)।

```python
# sanity check: PDW -> occupancy reconstruction must match the ground truth exactly
_demo_emitters = make_random_scenario(seed=1)
_truth, _owner, _amp, _pdws = build_truth_and_pdws(_demo_emitters, seed=1)
_recon = pulses_to_occupancy(_pdws)
assert np.array_equal(_truth, _recon), "occupancy reconstruction from PDWs does not match ground truth"
print(f"Generated {len(_pdws)} PDWs from {len(_demo_emitters)} emitters over "
      f"{EPISODE_SLOTS} slots -- PDW to occupancy-grid reconstruction verified exact.")
```
- यह एक **verification/testing block** है, जो असली data पर काम शुरू करने से पहले ही यह पक्का कर लेता है कि हमारा pipeline सही है।
- `_demo_emitters = make_random_scenario(seed=1)`: एक demo scenario बनाया (seed=1 से, ताकि हमेशा वही scenario आए — नाम के आगे `_` लगाना Python convention है, बताने के लिए कि यह "temporary/internal" variable है)।
- `_truth, _owner, _amp, _pdws = build_truth_and_pdws(_demo_emitters, seed=1)`: उसका ग्राउंड-ट्रुथ और pulses निकाले।
- `_recon = pulses_to_occupancy(_pdws)`: pulses से दोबारा occupancy grid बनाया।
- `assert np.array_equal(_truth, _recon), "..."`: **assert** का मतलब है — "अगर यह शर्त झूठी निकली, तो प्रोग्राम यहीं रुक जाए और यह error-message दिखाए"। `np.array_equal` दो arrays को element-by-element compare करता है। अगर दोनों बिल्कुल एक जैसे हैं (जैसा होना ही चाहिए, हमारे design के हिसाब से), तो कोई error नहीं आएगी — यही हमारा "proof" है कि pipeline सही काम कर रहा है।
- आख़िरी `print`: कितने pulses बने, कितने emitters से, यह confirm करने वाला message।

---

## Code Cell 4 — Demo Plot (Section 2 का visualization)

**यह cell क्या करता है:** ऊपर बनाए गए demo scenario को दो graphs में दिखाता है — एक pulse-scatter-plot, एक occupancy-grid heatmap।

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
```
- `plt.subplots(1, 2, ...)`: एक figure के अंदर 1 row, 2 columns में दो अलग-अलग plots (subplots) बनाए। `figsize=(13,4)` से पूरी image का size (चौड़ाई, ऊँचाई) inches में सेट किया। `fig` पूरे चित्र को represent करता है, `axes` उन दो subplot areas की एक list/array है।

```python
axes[0].scatter([p.toa_us / 1000 for p in _pdws], [p.freq_mhz / 1000 for p in _pdws],
                c=[p.emitter_id for p in _pdws], cmap="tab10", s=8)
axes[0].set_xlabel("Time of arrival (ms)")
axes[0].set_ylabel("Centre frequency (GHz)")
axes[0].set_title("Synthetic PDW stream (colour = true emitter id)")
```
- `axes[0]`: पहला subplot (बाईं तरफ़)।
- `.scatter(x, y, c=..., cmap=..., s=...)`: एक scatter-plot (बिखरे हुए बिंदुओं वाला graph) बनाता है।
  - x-axis: `[p.toa_us / 1000 for p in _pdws]` — हर pulse का समय, microseconds से milliseconds में बदलकर (÷1000)। यह list-comprehension है — हर `p` (pulse) के लिए उसकी value निकाली।
  - y-axis: वैसे ही frequency को MHz से GHz में बदला।
  - `c=[p.emitter_id for p in _pdws]`: हर बिंदु का रंग (`c` = color) उसके सच्चे emitter-id के हिसाब से तय होगा — इससे अलग-अलग emitters अलग रंगों में दिखेंगे, पैटर्न देखना आसान हो जाता है।
  - `cmap="tab10"`: matplotlib का एक built-in रंग-योजना (colormap), जिसमें 10 साफ़-साफ़ अलग रंग होते हैं।
  - `s=8`: हर बिंदु का size (छोटा रखा, क्योंकि हज़ारों points हो सकते हैं)।
- `set_xlabel`, `set_ylabel`, `set_title`: axes के नाम और चित्र का शीर्षक लिख दिया।

```python
axes[1].imshow(_truth, aspect="auto", origin="lower", cmap="Greys",
               extent=[0, EPISODE_SLOTS, 0, N_BANDS])
axes[1].set_xlabel("Time slot")
axes[1].set_ylabel("Band index")
axes[1].set_title("Ground-truth occupancy grid (white = ON)")
plt.tight_layout()
plt.show()
```
- `axes[1]`: दूसरा subplot (दाईं तरफ़)।
- `.imshow(_truth, ...)`: एक 2D array को image/heatmap की तरह दिखाता है। हमारा `_truth` grid `(बैंड, समय)` के आकार का है — इसे सीधे image बना दिया, जहाँ हर pixel True/False (ON/OFF) है।
  - `aspect="auto"`: image को खींचकर subplot के आकार में fit कर दो (pixels चौकोर रहने की ज़िद ना करें)।
  - `origin="lower"`: चित्र का (0,0) point नीचे-बाएँ से शुरू हो (सामान्य graph की तरह), ना कि ऊपर-बाएँ से (जो image-processing में default होता है)।
  - `cmap="Greys"`: काले-सफ़ेद रंगों में — True (ON) सफ़ेद दिखेगा, False (OFF) काला।
  - `extent=[0, EPISODE_SLOTS, 0, N_BANDS]`: axes पर असली scale set की (0 से 150 slots तक x-axis पर, 0 से 12 बैंड तक y-axis पर), नहीं तो सिर्फ़ array-index दिखते।
- `plt.tight_layout()`: matplotlib को कहा कि subplots के बीच की जगह/margins को अपने-आप सही adjust कर ले, ताकि titles/labels overlap ना करें।
- `plt.show()`: चित्र को असल में screen/notebook-output पर दिखा दो।

---

## Code Cell 5 — `ScanEnv` (Section 3)

**यह cell क्या करता है:** यह हमारा पूरा **simulation environment** है — receiver का model जो हर scan-decision के बाद noisy detection देता है, belief update करता है, और आगे के लिए prediction भी करता है। यह पूरे notebook की सबसे केंद्रीय (core) class है।

```python
class ScanEnv:
    def __init__(self, truth, amp_grid=None, sensitivity_db=SENSITIVITY_DB, slope_db=SLOPE_DB,
                 pd_fallback=PD_SENSOR, pfa=PFA_SENSOR, seed=0):
        self.truth = truth
        self.amp_grid = amp_grid
        self.sensitivity_db = sensitivity_db
        self.slope_db = slope_db
        self.pd_fallback = pd_fallback
        self.pfa = pfa
        self.rng = np.random.default_rng(seed)
        self.n_bands, self.n_slots = truth.shape
        self.reset()
```
- Constructor — जब भी एक environment बनता है, इसे `truth` grid (असली सच्चाई) दी जाती है, साथ में amplitude grid (वैकल्पिक), और कुछ sensor-parameters।
- सारी दी गई values को `self.` में save कर लिया, ताकि class के बाकी methods उन्हें इस्तेमाल कर सकें।
- `self.rng = np.random.default_rng(seed)`: इस environment के अंदर होने वाली randomness (detection सफल हुआ या नहीं) के लिए एक अलग seeded generator।
- `self.n_bands, self.n_slots = truth.shape`: `truth` एक 2D NumPy array है; `.shape` से उसका आकार `(बैंड-संख्या, slot-संख्या)` मिलता है — दोनों values एक साथ निकाल लीं।
- `self.reset()`: constructor के अंदर ही एक बार `reset()` बुला दिया, ताकि environment शुरू से ही एक valid initial-state में हो।

```python
    def reset(self):
        self.t = 0
        self.belief = np.full(self.n_bands, 0.5)
        self.last_visit = np.full(self.n_bands, -1)
        self.hit_count = np.zeros(self.n_bands, dtype=int)
        self.visit_count = np.zeros(self.n_bands, dtype=int)
        self.trans_counts = np.ones((self.n_bands, 2, 2))   # Laplace-smoothed online kernel estimate
        self.history = []
        return self._obs()
```
- `reset()`: episode को दोबारा शुरुआत की स्थिति में ले आता है (जैसे नया game शुरू करना)।
- `self.t = 0`: समय-counter शून्य से शुरू।
- `self.belief = np.full(self.n_bands, 0.5)`: हर बैंड के लिए शुरुआती **belief** (मान्यता) 0.5 (यानी 50-50, "कुछ पता नहीं ON है या OFF") — क्योंकि हमें कोई पहले से जानकारी (prior intel) नहीं है।
- `self.last_visit = np.full(self.n_bands, -1)`: हर बैंड को "अभी तक कभी नहीं देखा" के तौर पर `-1` मार्क किया।
- `self.hit_count`, `self.visit_count`: हर बैंड के लिए कितनी बार scan किया और कितनी बार hit (detection) मिला — दोनों शून्य से शुरू।
- `self.trans_counts = np.ones((self.n_bands, 2, 2))`: यह हर बैंड के लिए एक 2×2 counting-table है — "OFF से OFF कितनी बार गया", "OFF से ON कितनी बार गया", वगैरह — यानी 2-state Markov transition को online सीखने के लिए counts। शुरुआत में सब जगह `1` (0 नहीं) रखा — इसे **Laplace smoothing** कहते हैं, ताकि शुरुआत में division-by-zero ना हो और हर संभावना कम-से-कम थोड़ी सी मानी जाए।
- `self.history = []`: पूरे episode का record रखने के लिए खाली list।
- `return self._obs()`: reset के बाद initial "observation" (जो scheduler को दिखेगा) लौटा दी।

```python
    def _obs(self):
        staleness = np.clip(self.t - self.last_visit, 0, None) / max(self.n_slots, 1)
        return {"belief": self.belief.copy(), "staleness": staleness, "t": self.t}
```
- `_obs`: (नाम से पहले `_` का मतलब है कि यह "internal use के लिए" method है) — यह वो जानकारी बनाता है जो scheduler को हर step पर दी जा सकती है।
- `staleness = np.clip(self.t - self.last_visit, 0, None) / max(self.n_slots, 1)`: हर बैंड को आख़िरी बार देखे कितना समय हो गया, उसे 0 से ऊपर clip किया (`np.clip(x, 0, None)` मतलब नीचे की सीमा 0, ऊपर की कोई सीमा नहीं), फिर episode-length से normalize (0 से 1 के बीच scale) कर दिया।
- `self.belief.copy()`: belief का एक **copy** दिया, असली array नहीं — ताकि बाहर कोई गलती से इसे बदल ना दे और अंदर का असली state सुरक्षित रहे।
- एक Python **dictionary** (`{key: value, ...}`) के रूप में लौटाया — तीन चीज़ें एक साथ।

```python
    def transition_probs(self, band):
        c = self.trans_counts[band]
        return c[0, 1] / c[0].sum(), c[1, 1] / c[1].sum()   # (p01, p11)
```
- `transition_probs`: किसी बैंड के लिए, अब तक जो counts इकट्ठा हुए हैं, उनसे probability निकालता है।
- `c = self.trans_counts[band]`: इस बैंड की 2×2 counting-table निकाली।
- `c[0, 1] / c[0].sum()`: `c[0]` मतलब "जब पहले OFF (0) था" वाली row; `c[0,1]` मतलब "OFF से ON (1) में गया कितनी बार"; इसे row के total-sum से divide करके probability निकाली — यह `p01` है (OFF→ON की संभावना)।
- `c[1, 1] / c[1].sum()`: वैसे ही, "ON था और ON ही रहा" की संभावना — यह `p11` है (ON→ON)।
- दोनों values एक tuple में लौटा दीं।

```python
    def _pd_for(self, action):
        if self.amp_grid is None:
            return self.pd_fallback
        amp = self.amp_grid[action, self.t]
        return float(1.0 / (1.0 + math.exp(-(amp - self.sensitivity_db) / self.slope_db)))
```
- `_pd_for`: किसी दिए गए action (यानी scanned बैंड) के लिए, **असली** detection-probability निकालता है (जो scheduler को पता नहीं होती, सिर्फ environment जानता है)।
- अगर amplitude-grid ही नहीं दिया गया (`None`), तो simplistic fallback (constant `pd_fallback`) दे दो।
- वरना, `amp = self.amp_grid[action, self.t]`: उस बैंड और अभी के समय पर signal की amplitude निकाली।
- `1.0 / (1.0 + math.exp(-(amp - sensitivity_db) / slope_db))`: यह **logistic (sigmoid) function** का formula है — एक "S" आकार का curve, जो किसी भी input को 0 से 1 के बीच squeeze कर देता है। जब `amp` ठीक `sensitivity_db` के बराबर हो, तब result 0.5 आता है (यही "sensitivity" की परिभाषा है — जहाँ Pd = 50%)। `slope_db` जितना छोटा, curve उतना ज़्यादा तीखा (sharp) होगा।

```python
    def step(self, action: int):
        assert 0 <= action < self.n_bands
        true_state = bool(self.truth[action, self.t])
        amp_db = None
        if true_state:
            pd_eff = self._pd_for(action)
            detect = self.rng.random() < pd_eff
            if self.amp_grid is not None:
                amp_db = float(self.amp_grid[action, self.t])
        else:
            detect = self.rng.random() < self.pfa
```
- `step`: यह environment का सबसे ज़रूरी method है — scheduler एक action (कौन सा बैंड scan करना है) देता है, और यह method उसका नतीजा निकालता है। यही **"एक time-step आगे बढ़ना"** है।
- `assert 0 <= action < self.n_bands`: पक्का किया कि दिया गया action एक valid बैंड-नंबर है (0 से 11 के बीच), नहीं तो प्रोग्राम रुक जाएगा और गलती पकड़ में आएगी।
- `true_state = bool(self.truth[action, self.t])`: **असली सच्चाई** निकाली — क्या यह बैंड अभी सच में ON है? (यह सिर्फ़ environment को पता है, scheduler को कभी सीधे नहीं दिखाया जाता)।
- `if true_state:`: अगर सच में ON है —
  - `pd_eff = self._pd_for(action)`: असली, amplitude-based detection-probability निकाली।
  - `detect = self.rng.random() < pd_eff`: coin-flip जैसा — random संख्या निकाली, अगर वह `pd_eff` से कम है तो detection **सफल** माना (`True`)।
  - amplitude भी record कर ली (`amp_db`), बाद में sensitivity-curve analysis के लिए काम आएगी।
- `else:` अगर सच में OFF है, फिर भी कभी-कभी receiver ग़लती से "detect हुआ" बोल सकता है — यह **false alarm** है, इसकी संभावना `pfa` (fixed) है।

```python
        reward = 1.0 if detect else 0.0
        self.visit_count[action] += 1
        self.hit_count[action] += int(detect)
```
- `reward = 1.0 if detect else 0.0`: अगर detection हुआ, reward 1, नहीं तो 0 — यही "hits और misses पर training" वाला सीधा-सादा reward है जो problem statement माँगता है।
- `self.visit_count[action] += 1`: इस बैंड को कितनी बार scan किया, उसका counter एक बढ़ाया।
- `self.hit_count[action] += int(detect)`: `int(True)` = 1, `int(False)` = 0 — इसलिए अगर detect हुआ, hit-counter भी बढ़ जाएगा, नहीं तो वैसा ही रहेगा।

```python
        prior = self.belief[action]
        like_on = self.pd_fallback if detect else (1 - self.pd_fallback)
        like_off = self.pfa if detect else (1 - self.pfa)
        post = (like_on * prior) / (like_on * prior + like_off * (1 - prior) + 1e-9)
        self.belief[action] = post
```
- यह **Bayes' theorem** का सीधा इस्तेमाल है — scheduler का belief update करना, लेकिन ध्यान दें: scheduler को असली `pd_eff` नहीं पता (वो environment का secret है), इसलिए वह सिर्फ़ अपने **मान लिए हुए** `pd_fallback` (=0.9) और `pfa` के आधार पर ही अपना belief update करता है — यही "कोई prior reliable intelligence नहीं" की असली गहराई है।
- `prior = self.belief[action]`: scan करने से **पहले** का belief।
- `like_on = self.pd_fallback if detect else (1 - self.pd_fallback)`: "अगर सच में ON होता, तो यह observation (detect हुआ या नहीं) मिलने की कितनी संभावना थी" — यह **likelihood** है।
- `like_off = self.pfa if detect else (1 - self.pfa)`: वैसे ही, "अगर सच में OFF होता तो यह observation मिलने की संभावना"।
- `post = (like_on * prior) / (like_on * prior + like_off * (1 - prior) + 1e-9)`: यही **Bayes का formula** है: नई (posterior) संभावना = (likelihood × prior) / (कुल संभावना)। `1e-9` एक बहुत छोटी संख्या जोड़ी है सिर्फ़ यह पक्का करने के लिए कि denominator कभी बिल्कुल शून्य ना हो (divide-by-zero से बचाव)।
- `self.belief[action] = post`: नया belief save कर लिया — सिर्फ़ उसी बैंड का, जिसे अभी scan किया।

```python
        self.trans_counts[action, int(prior > 0.5), int(post > 0.5)] += 1
        self.last_visit[action] = self.t
```
- `self.trans_counts[action, int(prior > 0.5), int(post > 0.5)] += 1`: यहाँ हम transition-counting table update कर रहे हैं। चूँकि हमें असली state (True/False) नहीं पता (सिर्फ़ belief है), हम belief को ही एक **अनुमानित state** मान लेते हैं — अगर belief 0.5 से ज़्यादा है तो "मानो ON था" (`int(...)` से `True`→1, `False`→0)। इससे pehle-belief और baad-belief, दोनों के आधार पर सही जगह count बढ़ा दिया।
- `self.last_visit[action] = self.t`: इस बैंड को "अभी-अभी देखा" के तौर पर mark कर दिया (staleness का hisab रखने के लिए)।

```python
        for b in range(self.n_bands):          # predict step: propagate every band forward
            p01, p11 = self.transition_probs(b)
            bel = self.belief[b]
            self.belief[b] = bel * p11 + (1 - bel) * p01
```
- यह वो हिस्सा है जो environment को "smart" बनाता है — सिर्फ़ जिस बैंड को scan किया उसी का नहीं, बल्कि **सारे** बैंड्स का belief एक कदम आगे predict किया जाता है, भले ही उन्हें scan ना किया हो।
- `for b in range(self.n_bands):`: हर बैंड के लिए (scanned वाले समेत)।
  - `p01, p11 = self.transition_probs(b)`: उस बैंड के अभी तक सीखे हुए transition-probabilities निकाले।
  - `bel = self.belief[b]`: उस बैंड का अभी का belief।
  - `self.belief[b] = bel * p11 + (1 - bel) * p01`: यह standard **HMM (Hidden Markov Model) predict-step formula** है — "अगली state ON होने की संभावना" = (अभी ON होने की संभावना × ON-रहने की-probability) + (अभी OFF होने की संभावना × OFF-से-ON-जाने की-probability)। यही वो चीज़ है जो एक ना-देखे-गए बैंड का belief समय के साथ prior (औसत) की तरफ़ धीरे-धीरे खिसकाती है, बजाय हमेशा वहीं अटके रहने के।

```python
        self.history.append((self.t, action, true_state, detect))
        self.t += 1
        done = self.t >= self.n_slots
        info = {"true_state": true_state, "detect": detect, "amp_db": amp_db}
        return self._obs(), reward, done, info
```
- `self.history.append(...)`: इस step का पूरा record (समय, action, असली-सच्चाई, detection) history-list में जोड़ दिया — डिबगिंग/analysis के लिए काम आ सकता है।
- `self.t += 1`: समय एक कदम आगे बढ़ा दिया।
- `done = self.t >= self.n_slots`: क्या episode ख़त्म हो गया (सारे slots इस्तेमाल हो गए)?
- `info = {...}`: एक dictionary जिसमें वो सारी details हैं जो सिर्फ़ **evaluation/debugging** के लिए हैं, scheduler के फैसले लेने के लिए नहीं (scheduler सिर्फ़ `reward` और `_obs()` से belief देखता है, `true_state` को सीधे कभी access नहीं करना चाहिए — हालाँकि Python इसे रोकता नहीं, बस design का discipline है)।
- यह चार चीज़ें लौटाता है — यह standard "Gym-जैसा" (OpenAI Gym environment interface जैसा) पैटर्न है: `(observation, reward, done, info)`।

---

## Code Cell 6 — Baseline Schedulers (Section 4)

**यह cell क्या करता है:** एक common `Scheduler` interface (base class) define करता है, फिर दो सबसे सरल ("dumb") schedulers बनाता है — Round-robin और Random — और एक helper function `run_episode` जो किसी भी scheduler को एक पूरे episode में चला सके।

```python
class Scheduler:
    name = "base"
    def reset(self, env): pass
    def select(self, env) -> int: raise NotImplementedError
    def observe(self, env, action, t, reward, info): pass
```
- यह base class है — हर scheduler (चाहे simple हो या PPO वाला complex) इसी interface का पालन करेगा, ताकि हम उन सबको एक ही तरीके से (एक जैसे loop में) चला सकें — यह Object-Oriented Programming का **polymorphism** (एक ही interface, अलग-अलग व्यवहार) दिखाता है।
- `name = "base"`: हर scheduler का एक readable नाम, ताकि tables/graphs में दिखा सकें।
- `def reset(self, env): pass`: episode शुरू होने से पहले कुछ setup करना हो तो यहाँ होगा — base में कुछ नहीं करना (`pass` = कुछ मत करो)।
- `def select(self, env) -> int: raise NotImplementedError`: यह सबसे ज़रूरी method है — "अभी कौन सा बैंड scan करना है" तय करता है। base class में इसे लागू नहीं किया — हर बच्चा-class को खुद अपनी strategy लिखनी होगी।
- `def observe(self, env, action, t, reward, info): pass`: scan के बाद, अगर scheduler को कुछ सीखना/record करना है (जैसे hit-time याद रखना), तो यहाँ होगा — base में कुछ ज़रूरी नहीं।

```python
class RoundRobinScheduler(Scheduler):
    name = "Round-robin (open loop)"
    def reset(self, env): self._next = 0
    def select(self, env):
        a = self._next % env.n_bands
        self._next += 1
        return a
```
- सबसे सरल strategy — बारी-बारी हर बैंड को घुमा-घुमाकर scan करना (जैसे 0,1,2,...,11,0,1,2,...), बिना किसी belief के — यह पूरी तरह **open-loop** है (कोई feedback इस्तेमाल नहीं होता)।
- `self._next = 0`: अगला scan करने वाला बैंड-नंबर शुरू में 0।
- `a = self._next % env.n_bands`: `%` (modulo) से पक्का किया कि नंबर हमेशा 0 से `n_bands-1` के बीच ही रहे, चाहे `_next` कितना भी बढ़ जाए — इसी से "घूमना" (wrap-around) होता है।
- `self._next += 1`: अगली बार के लिए नंबर एक बढ़ा दिया।

```python
class RandomScheduler(Scheduler):
    name = "Random scan (open loop)"
    def reset(self, env): self.rng = np.random.default_rng(123)
    def select(self, env): return int(self.rng.integers(0, env.n_bands))
```
- हर बार बिल्कुल randomly कोई भी बैंड चुन लेना — यह भी open-loop है (कोई सीख नहीं, कोई पैटर्न नहीं)।
- `self.rng = np.random.default_rng(123)`: seed `123` fix रखी, ताकि हर बार जब `reset` हो, randomness reproducible रहे।

```python
def run_episode(env, scheduler, seed=None):
    if seed is not None:
        env.rng = np.random.default_rng(seed)
    env.reset()
    scheduler.reset(env)
    total_reward = 0.0
    for _ in range(env.n_slots):
        t = env.t
        action = scheduler.select(env)
        obs, reward, done, info = env.step(action)
        scheduler.observe(env, action, t, reward, info)
        total_reward += reward
        if done:
            break
    return total_reward
```
- यह एक general-purpose helper है — किसी भी `env` और किसी भी `scheduler` को लेकर, एक पूरा episode चलाता है, और कुल reward बताता है।
- `if seed is not None: env.rng = ...`: अगर एक specific seed माँगी गई है, तो environment के internal randomness को उससे बदल दिया (नहीं तो पहले से जो था वही चलेगा)।
- `env.reset()`, `scheduler.reset(env)`: दोनों को शुरुआती स्थिति में ले आया।
- `for _ in range(env.n_slots):`: episode के हर slot के लिए।
  - `t = env.t`: अभी का समय (action लेने से **पहले**) save कर लिया — यह ज़रूरी है क्योंकि `step()` के बाद `env.t` एक बढ़ जाएगा, तो हमें असली-समय याद रखना ज़रूरी है।
  - `action = scheduler.select(env)`: scheduler से फैसला पूछा — कौन सा बैंड scan करना है।
  - `obs, reward, done, info = env.step(action)`: environment में वो action चलाया, नतीजा लिया।
  - `scheduler.observe(env, action, t, reward, info)`: scheduler को यह नतीजा बताया (ताकि वो, अगर ज़रूरी हो, उससे सीख सके)।
  - `total_reward += reward`: कुल reward में जोड़ा।
  - `if done: break`: अगर episode ख़त्म हो गया, loop से बाहर आ जाओ।
- `return total_reward`: इस episode का कुल score।

---

## Code Cell 7 — Belief-Index + Periodicity Scheduler (Section 5)

**यह cell क्या करता है:** एक ज़्यादा समझदार ("smart") strategy बनाता है — जो belief (सम्भावना) के आधार पर फैसला लेती है, साथ ही periodic (rotating-antenna) emitters की भविष्यवाणी भी करती है।

```python
class PeriodicityTracker:
    def __init__(self, n_bands, min_hits=3):
        self.n_bands = n_bands
        self.min_hits = min_hits
        self.hit_times = [[] for _ in range(n_bands)]
```
- यह class हर बैंड के लिए यह track करती है कि कब-कब detection (hit) हुआ, ताकि एक **periodicity (चक्र)** का पता लगाया जा सके।
- `self.hit_times = [[] for _ in range(n_bands)]`: हर बैंड के लिए एक अलग खाली list — यानी एक "list की list" (`n_bands` खाली lists)। इसे list-comprehension से बनाया, ना कि सिर्फ़ `[[]] * n_bands` से — क्योंकि वह तरीका ग़लत होता (सारी sub-lists असल में एक ही object की तरफ़ point करतीं, जिससे एक बैंड का hit दूसरे बैंड की list में भी दिख जाता)।
- `min_hits=3`: कम-से-कम 3 hits ना मिलें, तब तक हम "period" का अंदाज़ा नहीं लगाएँगे (बहुत कम data पर भरोसा नहीं करेंगे)।

```python
    def record_hit(self, band, t):
        self.hit_times[band].append(t)
```
- जब किसी बैंड में hit हो, उस समय को उस बैंड की list में जोड़ दो।

```python
    def estimate_period(self, band):
        hits = self.hit_times[band]
        if len(hits) < self.min_hits:
            return None
        diffs = np.diff(hits[-6:])
        return float(np.median(diffs)) if len(diffs) else None
```
- `estimate_period`: उस बैंड का "दो hits के बीच कितना गैप रहता है" (यानी period) अंदाज़ा लगाती है।
- अगर पर्याप्त hits नहीं मिले (`min_hits` से कम), `None` (कुछ नहीं) लौटा दो — अभी अंदाज़ा नहीं लगा सकते।
- `diffs = np.diff(hits[-6:])`: सिर्फ़ आख़िरी 6 hits लिए (`hits[-6:]` — पीछे से 6 items), और `np.diff` से consecutive (लगातार) values के बीच का अंतर निकाला — जैसे `[10, 25, 40]` से `[15, 15]` मिलेगा।
- `np.median(diffs)`: इन gaps का **median** (बीच वाला मान) लिया, average नहीं — median एक-दो बेतरतीब outlier values से कम प्रभावित होता है, ज़्यादा भरोसेमंद अंदाज़ा देता है।

```python
    def predicted_next_hit(self, band, t_now):
        hits = self.hit_times[band]
        period = self.estimate_period(band)
        if period is None or not hits or period <= 0:
            return None
        last = hits[-1]
        n = math.ceil((t_now - last) / period) if t_now > last else 1
        return last + max(n, 1) * period
```
- `predicted_next_hit`: अगली बार कब यह emitter फिर से "दिखेगा" (illuminate करेगा), उसका अनुमान।
- अगर period का पता नहीं, या कोई hit ही नहीं मिला, या period 0-या-कम है (ग़लत/बेमानी), तो `None` (predict नहीं कर सकते)।
- `last = hits[-1]`: आख़िरी hit कब हुआ था।
- `n = math.ceil((t_now - last) / period) if t_now > last else 1`: अगर अभी का समय आख़िरी hit से आगे निकल चुका है, तो कितने पूरे "period" गुज़र चुके हैं, यह `math.ceil` (ऊपर की तरफ़ round) से निकाला — ताकि अगला predicted-hit हमेशा भविष्य में हो, अतीत में नहीं।
- `return last + max(n, 1) * period`: आख़िरी hit + n×period = अगला अनुमानित hit-time।

```python
    def bonus(self, band, t_now, window=2.0):
        pred = self.predicted_next_hit(band, t_now)
        if pred is None:
            return 0.0
        return math.exp(-abs(pred - t_now) / window)   # peaks at the predicted illumination slot
```
- `bonus`: एक ऐसा "score" देता है जो 0 से 1 के बीच होता है — जितना ज़्यादा हम **predicted illumination time के करीब** हैं, उतना यह score ज़्यादा (1 के करीब) होगा; दूर होने पर तेज़ी से घटकर 0 के करीब चला जाता है।
- अगर prediction ही नहीं है, bonus 0 (कोई अतिरिक्त फायदा नहीं)।
- `math.exp(-abs(pred - t_now) / window)`: यह एक **exponential decay** curve है — `abs(pred - t_now)` predicted-time और अभी के समय के बीच का फ़ासला, जितना बड़ा, `exp(-बड़ी संख्या)` उतना ही छोटा (0 के करीब)। `window` control करता है कि यह curve कितना "तेज़" घटता है।

```python
def staleness_bonus(env):
    return np.clip(env.t - env.last_visit, 0, None) / max(env.n_slots, 1)
```
- एक standalone helper function (class के बाहर) — हर बैंड को कितने समय से नहीं देखा, उसे 0-1 के बीच normalize करके देता है। यह `ScanEnv._obs()` के अंदर वाले जैसा ही calculation है, लेकिन यहाँ हम इसे सीधे scheduler के अंदर, `env` object से निकालकर फिर से इस्तेमाल कर रहे हैं।

```python
class GreedyBeliefScheduler(Scheduler):
    name = "Belief-index (belief + exploration bonus)"
    def __init__(self, explore_weight=0.15):
        self.explore_weight = explore_weight

    def select(self, env):
        idx = env.belief + self.explore_weight * staleness_bonus(env)
        return int(np.argmax(idx))
```
- यह हमारा "Tier-1 smart" scheduler है — सिर्फ़ belief के आधार पर फैसला (theoretically यह 2-state, positively-correlated channels के लिए optimal साबित है — Section 5 का markdown देखें), मगर साथ में एक "exploration bonus" भी है ताकि बैंड्स कभी हमेशा के लिए "भूले" ना जाएँ।
- `idx = env.belief + self.explore_weight * staleness_bonus(env)`: हर बैंड के लिए एक कुल "index/score" — belief plus (थोड़ा वज़न देकर) staleness-bonus। `explore_weight=0.15` यह तय करता है कि exploration को कितना महत्व देना है (belief के मुक़ाबले)।
- `np.argmax(idx)`: सबसे बड़े index वाले बैंड का **position** (index-number) निकाला — यानी "जो बैंड सबसे ज़्यादा promising लग रहा है, वही चुनो"।
- `int(...)`: NumPy के integer-type से Python के normal `int` में बदला (ताकि आगे बिना दिक्कत इस्तेमाल हो सके)।

```python
class BeliefPeriodicityScheduler(Scheduler):
    name = "Belief + periodicity + exploration index"
    def __init__(self, periodicity_weight=0.6, explore_weight=0.15):
        self.periodicity_weight = periodicity_weight
        self.explore_weight = explore_weight

    def reset(self, env):
        self.tracker = PeriodicityTracker(env.n_bands)
        self._pending_predictions = {}   # band -> predicted next-hit slot
        self.prediction_log = []          # (band, predicted_t, actual_t) for error metric
```
- यह ऊपर वाले का उन्नत (upgraded) version है, जिसमें periodicity-prediction भी जुड़ी है।
- `reset(self, env)`: हर नए episode की शुरुआत में एक नया `PeriodicityTracker` बना लिया (पुराने episode का data आगे carry नहीं होना चाहिए), और दो खाली containers — `_pending_predictions` (एक dictionary: बैंड → अगली predicted hit-time) और `prediction_log` (predictions का पूरा इतिहास, ताकि आगे "prediction कितनी सही थी" यह error निकाल सकें)।

```python
    def select(self, env):
        periodicity = np.array([self.tracker.bonus(b, env.t) for b in range(env.n_bands)])
        idx = env.belief + self.explore_weight * staleness_bonus(env) + self.periodicity_weight * periodicity
        return int(np.argmax(idx))
```
- `periodicity = np.array([self.tracker.bonus(b, env.t) for b in range(env.n_bands)])`: हर बैंड के लिए periodicity-bonus निकाला (list-comprehension से एक-एक करके), फिर उसे NumPy array में बदला।
- `idx = ...`: अब तीन चीज़ों का जोड़ है — belief + (weighted) staleness + (weighted) periodicity। तीनों मिलकर एक ही अंतिम "score" बनाते हैं।
- बाकी वही `argmax` वाला तरीका — सबसे बड़े score वाला बैंड चुना।

```python
    def observe(self, env, action, t, reward, info):
        if action not in self._pending_predictions:
            pred = self.tracker.predicted_next_hit(action, t)
            if pred is not None:
                self._pending_predictions[action] = pred
        if info["detect"]:
            if action in self._pending_predictions:
                self.prediction_log.append((action, self._pending_predictions.pop(action), t))
            self.tracker.record_hit(action, t)
```
- `observe`: scan करने के बाद जो सीखना है, वो यहाँ होता है।
- `if action not in self._pending_predictions:`: अगर इस बैंड के लिए हमने अभी तक कोई prediction "pending" नहीं रखी है, तो एक नई prediction बना लो और save कर लो — ताकि बाद में जब असली hit मिले, तो हम compare कर सकें कि prediction सही थी या कितनी ग़लत।
- `if info["detect"]:`: अगर इस scan में असल में detection हुआ —
  - `if action in self._pending_predictions:`: अगर इस बैंड के लिए कोई pending prediction थी...
    - `self.prediction_log.append((action, self._pending_predictions.pop(action), t))`: उस prediction को log में डाल दो (बैंड, predicted-time, actual-time के साथ), और साथ ही `.pop()` से उसे pending-dictionary से **हटा भी दो** (ताकि दोबारा इस्तेमाल ना हो)।
  - `self.tracker.record_hit(action, t)`: इस hit को tracker में record कर दो, ताकि अगली बार period का अंदाज़ा और बेहतर हो।

---

## Code Cell 8 — Actor-Critic Network + Sanity Check (Section 6)

**यह cell क्या करता है:** Neural-network आधारित policy define करता है — जो शुरुआत में बिल्कुल हमारे belief-heuristic जैसी ही व्यवहार करती है (residual/warm-started design), और फिर verify करता है कि यह वाकई सही से match करती है।

```python
FEATS_PER_BAND = 4

def featurize(env, tracker):
    belief = env.belief
    staleness = np.clip(env.t - env.last_visit, 0, None) / max(env.n_slots, 1)
    hit_rate = env.hit_count / np.maximum(env.visit_count, 1)
    bonus = np.array([tracker.bonus(b, env.t) for b in range(env.n_bands)])
    return np.stack([belief, staleness, hit_rate, bonus], axis=1).astype(np.float32)
```
- `FEATS_PER_BAND = 4`: हर बैंड के बारे में हम neural-network को 4 numbers देंगे।
- `featurize`: environment (और tracker) से नेटवर्क के इनपुट के लिए features निकालने वाला function।
  - `belief`: हर बैंड का current belief।
  - `staleness`: फिर से वही "कितने समय से नहीं देखा" वाला calculation।
  - `hit_rate = env.hit_count / np.maximum(env.visit_count, 1)`: हर बैंड के लिए, अब तक जितनी बार scan किया उसमें से कितनी बार hit मिला — यानी empirical (अनुभव-आधारित) success-rate। `np.maximum(..., 1)` से divide-by-zero से बचाव (अगर किसी बैंड को अभी तक scan ही नहीं किया, तो 0/1 = 0 आएगा, error नहीं)।
  - `bonus`: periodicity-bonus, वैसे ही जैसे पहले।
- `np.stack([...], axis=1)`: चारों arrays (हर एक की length `n_bands`) को जोड़कर एक `(n_bands, 4)` आकार का 2D array बना दिया — `axis=1` का मतलब है कि यह एक नया "column" जोड़ने जैसा है, यानी हर बैंड की row में 4 features।
- `.astype(np.float32)`: PyTorch normally 32-bit floating-point numbers पसंद करता है (64-bit की जगह) — तेज़ और कम memory।

```python
class ActorCritic(nn.Module):
    def __init__(self, n_bands, feats_per_band=FEATS_PER_BAND, hidden=128):
        super().__init__()
        self.n_bands = n_bands
        self.feats_per_band = feats_per_band
        self.trunk = nn.Sequential(
            nn.Linear(n_bands * feats_per_band, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
```
- `class ActorCritic(nn.Module):`: PyTorch में हर neural-network `nn.Module` से inherit करता है — यह हमें automatic gradient-tracking, parameter-management वगैरह मुफ़्त में देता है।
- `super().__init__()`: PyTorch के internal setup को चलाना ज़रूरी है, नहीं तो layers ठीक से register नहीं होंगे।
- `self.trunk = nn.Sequential(...)`: **trunk** मतलब "मुख्य तना" — एक साझा (shared) hidden representation, जिसे आगे actor (policy) और critic (value) दोनों इस्तेमाल करेंगे।
  - `nn.Linear(n_bands * feats_per_band, hidden)`: एक fully-connected layer — इनपुट size `12*4=48`, output size `hidden=128`। यानी हर बैंड के हर feature को flatten करके एक साथ 48-लंबे vector में डाला।
  - `nn.Tanh()`: एक activation function — output को -1 से 1 के बीच "squeeze" करता है, नेटवर्क को non-linear (सिर्फ़ सीधी लाइन नहीं, घुमावदार पैटर्न भी सीख सके) बनाता है।
  - दो layers लगातार — यानी दो "hidden layers" वाला एक छोटा MLP (Multi-Layer Perceptron)।

```python
        self.correction_head = nn.Linear(hidden, n_bands)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
```
- `self.correction_head = nn.Linear(hidden, n_bands)`: एक अलग layer, जो trunk के output से हर बैंड के लिए एक "correction" संख्या निकालेगी।
- `nn.init.zeros_(self.correction_head.weight)` और `.bias`: इस layer के weights और bias को **शुरुआत में बिल्कुल शून्य** कर दिया (normally random होते हैं) — इसका मतलब है कि शुरुआत में यह layer कुछ भी output नहीं करेगी (हमेशा 0), चाहे input कुछ भी हो। यही वो "trick" है जो नेटवर्क को शुरुआत में हमारे heuristic जैसा बनाती है — RL सिर्फ़ धीरे-धीरे यह correction सीखेगा, पूरी policy को शुरू से नहीं सीखना पड़ेगा।

```python
        # feature order from featurize(): [belief, staleness, hit_rate, periodicity_bonus] --
        # these weights exactly match BeliefPeriodicityScheduler's own index formula.
        self.index_weights = nn.Parameter(torch.tensor([1.0, 0.15, 0.0, 0.6]))
        self.value_head = nn.Linear(hidden, 1)
```
- `self.index_weights = nn.Parameter(torch.tensor([1.0, 0.15, 0.0, 0.6]))`: यह चार संख्याएँ (`nn.Parameter` — यानी यह भी एक "सीखने लायक" parameter है, gradient descent से बदल सकता है) बिल्कुल वही weights हैं जो `BeliefPeriodicityScheduler` में hand-tuned थे: belief का weight `1.0`, staleness का `0.15`, hit_rate का `0.0` (इस्तेमाल नहीं होता index-formula में), periodicity का `0.6`। यही असली "warm-start" है — नेटवर्क literally हमारे heuristic के formula से शुरू होता है।
- `self.value_head = nn.Linear(hidden, 1)`: एक और छोटी layer — यह **critic** का हिस्सा है, जो trunk के output से एक ही संख्या निकालती है: "इस state की value (आगे कितना reward मिलने की उम्मीद है) कितनी है"।

```python
    def forward(self, x, max_correction=0.5):
        batch = x.shape[0]
        x_bands = x.view(batch, self.n_bands, self.feats_per_band)
        index_score = (x_bands * self.index_weights).sum(dim=-1)
        h = self.trunk(x)
        correction = max_correction * torch.tanh(self.correction_head(h))
        logits = index_score + correction
        value = self.value_head(h).squeeze(-1)
        return logits, value
```
- `forward`: PyTorch में यह method तब चलता है जब हम network को `ac(x)` की तरह call करते हैं — यही असली "forward pass" (input से output निकालना) है।
- `batch = x.shape[0]`: कितने samples एक साथ process हो रहे हैं (batch-size)।
- `x_bands = x.view(batch, self.n_bands, self.feats_per_band)`: फ्लैट (48-लंबे) vector को वापस `(बैंड, feature)` आकार में "देखा" — `.view()` data को बिना copy किए सिर्फ़ अलग shape में दिखाता है (मेमोरी में वही data है, बस अलग तरह से पढ़ा जा रहा है)।
- `index_score = (x_bands * self.index_weights).sum(dim=-1)`: हर बैंड के 4 features को उनके weights से multiply किया (broadcasting — छोटे array को बड़े array के साथ automatically match कर लिया), फिर `dim=-1` (आख़िरी dimension, यानी feature-dimension) पर sum किया — नतीजा हर बैंड के लिए एक संख्या: बिल्कुल वही जो `BeliefPeriodicityScheduler.select()` में `idx = belief + weight*staleness + weight*periodicity` से निकलता था।
- `h = self.trunk(x)`: trunk से shared hidden-representation निकाला।
- `correction = max_correction * torch.tanh(self.correction_head(h))`: correction-head से raw output निकाला, फिर `torch.tanh(...)` से उसे -1 से 1 के बीच bound किया, फिर `max_correction=0.5` से multiply किया — यानी अंतिम correction हमेशा -0.5 से +0.5 के बीच ही रहेगा, चाहे नेटवर्क अंदर कुछ भी सीखे। **यही वो सुरक्षा-सीमा है जो training को collapse होने से रोकती है।**
- `logits = index_score + correction`: अंतिम "score" हर बैंड के लिए — heuristic-based base-score plus एक सीमित correction।
- `value = self.value_head(h).squeeze(-1)`: value भी निकाल ली। `.squeeze(-1)` से आख़िरी dimension (जो size-1 है) हटा दी, ताकि shape साफ़-सुथरी रहे।
- दोनों (`logits`, `value`) लौटा दिए — logits से action चुनी जाएगी (policy), value से training में advantage निकलेगा (critic)।

```python
class PPOScheduler(Scheduler):
    name = "Residual PPO (warm-started, bounded correction)"
    def __init__(self, ac, n_bands, deterministic=True):
        self.ac = ac
        self.n_bands = n_bands
        self.deterministic = deterministic

    def reset(self, env):
        self.tracker = PeriodicityTracker(env.n_bands)
```
- यह `Scheduler` interface के हिसाब से एक wrapper class है, ताकि trained neural-network को बाकी सारे evaluation-code में उसी तरह इस्तेमाल कर सकें जैसे बाकी schedulers को।
- `self.ac`: trained `ActorCritic` network को save कर लिया।
- `self.deterministic`: अगर `True`, तो हमेशा सबसे अच्छा (highest-logit) action चुनेगा; अगर `False`, तो probability के हिसाब से randomly sample करेगा (training के दौरान exploration के लिए काम आता है)।
- `reset`: हर episode में नया periodicity-tracker।

```python
    def select(self, env):
        feats = featurize(env, self.tracker)
        x = torch.from_numpy(feats.reshape(1, -1))
        with torch.no_grad():
            logits, _ = self.ac(x)
            if self.deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        return int(action.item())
```
- `feats = featurize(env, self.tracker)`: अभी के state से features निकाले (आकार `(12, 4)`)।
- `x = torch.from_numpy(feats.reshape(1, -1))`: `.reshape(1, -1)` से इसे एक-लाइन के 48-लंबे vector में flatten किया, फिर एक extra "batch" dimension जोड़ी (size 1) — क्योंकि PyTorch हमेशा batch की उम्मीद करता है, भले ही एक ही sample हो। `torch.from_numpy(...)` से NumPy array को PyTorch tensor में बदला।
- `with torch.no_grad():`: यह PyTorch को बताता है — "यहाँ gradient calculation की ज़रूरत नहीं" (क्योंकि हम सिर्फ़ prediction ले रहे हैं, training नहीं कर रहे) — इससे memory बचती है और गति तेज़ होती है।
- `logits, _ = self.ac(x)`: network चलाया; value हमें यहाँ नहीं चाहिए इसलिए `_` (dummy variable) में डाल दिया।
- `if self.deterministic: action = torch.argmax(logits, dim=1)`: सबसे बड़े logit वाला action-index निकाला (`dim=1` — बैंड्स वाली dimension पर argmax)।
- `else: ... .sample()`: `Categorical` distribution बनाई (logits से probabilities निकालकर) और उससे एक random sample लिया — training के दौरान explore करने के लिए।
- `int(action.item())`: PyTorch tensor से Python का साधारण integer निकाला (`.item()` एक-value वाले tensor को plain number में बदलता है)।

```python
    def observe(self, env, action, t, reward, info):
        if info["detect"]:
            self.tracker.record_hit(action, t)
```
- वैसे ही जैसे बाकी schedulers में — अगर hit हुआ, tracker में record कर लो, ताकि periodicity-bonus का calculation आगे भी सही चलता रहे।

```python
# sanity check: an untrained residual policy must exactly reproduce the heuristic it warm-started from
_check_ac = ActorCritic(N_BANDS)
_check_sched = PPOScheduler(_check_ac, N_BANDS, deterministic=True)
_check_env = ScanEnv(_truth, amp_grid=_amp, seed=42)
_ref_sched = BeliefPeriodicityScheduler()
_check_sched.reset(_check_env)
_ref_env = ScanEnv(_truth, amp_grid=_amp, seed=42)
_ref_sched.reset(_ref_env)
```
- यह एक और **verification block** है — यह पक्का करना कि हमारा "residual policy बिना training के बिल्कुल heuristic जैसा behave करे" वाला दावा सही है, सिर्फ़ theory में नहीं बल्कि असल कोड में भी।
- `_check_ac = ActorCritic(N_BANDS)`: एक बिल्कुल नया (untrained) network बनाया।
- `_check_sched`, `_ref_sched`: दो schedulers — एक नेटवर्क-आधारित (untrained), एक पुराना heuristic।
- `_check_env`, `_ref_env`: दो **अलग** environment objects, लेकिन बिल्कुल एक जैसा `_truth`/`_amp` data और एक जैसा `seed=42` — ताकि दोनों बिल्कुल एक जैसी परिस्थिति में चलें और उनका आपस में तुलना करना सही रहे।

```python
for _ in range(_check_env.n_slots):
    a1 = _check_sched.select(_check_env)
    a2 = _ref_sched.select(_ref_env)
    assert a1 == a2, "untrained residual policy diverged from its warm-start heuristic"
    _, r1, d1, i1 = _check_env.step(a1)
    _, r2, d2, i2 = _ref_env.step(a2)
    _check_sched.observe(_check_env, a1, _check_env.t - 1, r1, i1)
    _ref_sched.observe(_ref_env, a2, _ref_env.t - 1, r2, i2)
print("Verified: untrained residual PPO policy exactly reproduces the belief+periodicity heuristic.")
```
- हर slot पर, दोनों schedulers से फैसला पूछा।
- `assert a1 == a2, "..."`: पक्का किया कि **दोनों का फैसला बिल्कुल एक जैसा** है — अगर कभी अलग निकला, प्रोग्राम तुरंत रुक जाएगा और यह ग़लती पकड़ में आ जाएगी (यानी हमारा warm-start design सही से काम नहीं कर रहा)।
- फिर दोनों environments में वही (same) action चलाया।
- `_check_env.t - 1`: `step()` के बाद `t` एक बढ़ चुका है, इसलिए `-1` करके वो असली समय वापस निकाला जिस पर action लिया गया था (`observe` को यही चाहिए)।
- लूप पूरा होने पर (यानी assert कभी fail नहीं हुआ), confirmation message print हुआ — यही हमारा **proof** है कि "residual policy = heuristic at step 0" वाला दावा सिर्फ़ बातों में नहीं, कोड में भी सच है।

---

## Code Cell 9 — PPO Training Pipeline (Section 6 का बाकी हिस्सा)

**यह cell क्या करता है:** पूरा Reinforcement Learning training-loop — rollout इकट्ठा करना, advantage निकालना, और नेटवर्क को update करना (critic-warmup और bounded-correction PPO के साथ)।

```python
def collect_rollout(ac, n_episodes, n_bands, n_slots, master_rng):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    ep_rewards = []
```
- `collect_rollout`: मौजूदा policy (नेटवर्क) को असली environment में चलाकर experience (data) इकट्ठा करता है — RL में इसे "rollout" कहते हैं।
- छह खाली lists — observation, action, log-probability, reward, value, और done-flag — हर step का data यहाँ जमा होगा। `ep_rewards`: हर episode का total reward, बाद में training-curve plot करने के लिए।

```python
    for _ in range(n_episodes):
        scenario_seed = int(master_rng.integers(0, 1_000_000))
        emitters = make_random_scenario(n_bands=n_bands, n_slots=n_slots, seed=scenario_seed)
        truth, owner, amp_grid, _ = build_truth_and_pdws(emitters, n_bands=n_bands, n_slots=n_slots,
                                                           seed=scenario_seed)
        env = ScanEnv(truth, amp_grid=amp_grid, seed=scenario_seed + 500_000)
        tracker = PeriodicityTracker(env.n_bands)
        env.reset()
        ep_r = 0.0
```
- हर episode के लिए —
- `scenario_seed = int(master_rng.integers(0, 1_000_000))`: एक बड़े range में से randomly एक seed चुना — यही domain-randomization है: **हर episode का scenario अलग** होगा।
- `emitters = make_random_scenario(...)`, `truth, owner, amp_grid, _ = build_truth_and_pdws(...)`: एक बिल्कुल नया random scenario बनाया।
- `env = ScanEnv(truth, amp_grid=amp_grid, seed=scenario_seed + 500_000)`: उस scenario पर आधारित नया environment — seed में `+500_000` जोड़ा ताकि environment की internal randomness (detection सफल हुआ या नहीं), scenario बनाने वाली randomness से टकराए ना।
- `tracker = PeriodicityTracker(env.n_bands)`: इस episode के लिए एक नया tracker।
- `env.reset()`: environment को शुरुआती स्थिति में लाया।
- `ep_r = 0.0`: इस episode का total reward, शून्य से शुरू।

```python
        for _ in range(n_slots):
            t = env.t
            feats = featurize(env, tracker)
            x = torch.from_numpy(feats.reshape(1, -1))
            with torch.no_grad():
                logits, value = ac(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            a = int(action.item())
            obs, reward, done, info = env.step(a)
            if info["detect"]:
                tracker.record_hit(a, t)
```
- हर slot में —
- `feats = featurize(...)`, `x = torch.from_numpy(...)`: state से नेटवर्क के लिए इनपुट तैयार किया (जैसे पहले)।
- `with torch.no_grad(): logits, value = ac(x)`: नेटवर्क से logits और value दोनों निकाले, बिना gradient track किए (rollout collection के दौरान training नहीं हो रही, सिर्फ़ experience इकट्ठा हो रही है)।
- `dist = torch.distributions.Categorical(logits=logits)`: logits से एक probability-distribution बनाई (softmax अंदर-ही-अंदर लागू होता है)।
- `action = dist.sample()`: उस distribution से **randomly** (probability के अनुसार) एक action चुना — training के दौरान हम हमेशा `argmax` नहीं लेते, ताकि नेटवर्क अलग-अलग actions try करके सीख सके (exploration)।
- `logp = dist.log_prob(action)`: इस चुने हुए action की log-probability निकाली — PPO के गणित में यह ज़रूरी होगी (ratio निकालने के लिए)।
- `a = int(action.item())`: normal Python integer में बदला।
- `obs, reward, done, info = env.step(a)`: environment में action चलाया।
- `if info["detect"]: tracker.record_hit(a, t)`: अगर hit हुआ, tracker को बताया।

```python
            obs_buf.append(feats); act_buf.append(a); logp_buf.append(float(logp.item()))
            rew_buf.append(reward); val_buf.append(float(value.item())); done_buf.append(done)
            ep_r += reward
            if done:
                break
        ep_rewards.append(ep_r)
    return obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, ep_rewards
```
- सारी चीज़ें अपनी-अपनी list में जोड़ दीं — `;` से एक ही लाइन में कई statements लिखे हैं (सिर्फ़ compact रखने के लिए, functionally अलग-अलग लाइनों जैसा ही है)।
- `ep_r += reward`: episode का कुल reward बढ़ाया।
- `if done: break`: episode ख़त्म होने पर loop से बाहर।
- `ep_rewards.append(ep_r)`: इस episode का final total, बड़ी list में जोड़ दिया।
- सारी six buffers + ep_rewards लौटा दिए — यह पूरा "collected experience" है, जो अब training में इस्तेमाल होगा।

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    next_value = next_advantage = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        advantages[t] = delta + gamma * lam * nonterminal * next_advantage
        next_value, next_advantage = values[t], advantages[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
```
- यह **GAE (Generalized Advantage Estimation)** है — PPO/Actor-Critic algorithms में एक standard तरीका यह मापने का कि "यह action कितना अच्छा/बुरा था, average से कितना ज़्यादा/कम"।
- `T = len(rewards)`: कुल कितने steps का data है।
- `advantages = np.zeros(T, ...)`: खाली array, भरेंगे।
- `next_value = next_advantage = 0.0`: दोनों को एक साथ 0 पर set किया (Python में यह allowed है — right से left चेन होती है)।
- `for t in reversed(range(T)):`: **पीछे से आगे** (अंत से शुरुआत की तरफ़) loop चलाया — GAE का calculation recursively पीछे से होता है, क्योंकि "आगे क्या होगा" पता होना ज़रूरी है advantage निकालने के लिए।
- `nonterminal = 0.0 if dones[t] else 1.0`: अगर यह step किसी episode का आख़िरी step है, तो आगे कोई value नहीं गिननी (0), नहीं तो अगली value गिनेंगे (1)।
- `delta = rewards[t] + gamma * next_value * nonterminal - values[t]`: यह **TD-error** (Temporal Difference error) है — "असल में जो मिला (reward + अगली state की अनुमानित value) माइनस जो नेटवर्क ने पहले अनुमान लगाया था" — यही बताता है कि प्रेडिक्शन कितनी सही/ग़लत थी। `gamma=0.99` भविष्य के reward को थोड़ा कम महत्व देता है (discount factor)।
- `advantages[t] = delta + gamma * lam * nonterminal * next_advantage`: यह वर्तमान TD-error में, आगे वाले advantage का एक हिस्सा (lam=0.95 से weighted) जोड़ता है — इससे noise कम होता है, ज़्यादा स्थिर अनुमान मिलता है।
- `next_value, next_advantage = values[t], advantages[t]`: अगली (पीछे की तरफ़, यानी पिछली असल में) iteration के लिए, अभी के values save कर लिए।
- `returns = advantages + np.array(values, ...)`: **return** (यानी value-function का training-target) = advantage + पुराना value-अनुमान — यह standard trick है ताकि value-network को train किया जा सके।

```python
def value_only_update(ac, opt, obs, returns, epochs=4, batch_size=256):
    obs_t = torch.from_numpy(np.stack(obs).reshape(len(obs), -1)).float()
    ret_t = torch.tensor(returns, dtype=torch.float32)
    idx_all = np.arange(obs_t.shape[0])
    for _ in range(epochs):
        np.random.shuffle(idx_all)
        for start in range(0, len(idx_all), batch_size):
            idx = idx_all[start:start + batch_size]
            _, values = ac(obs_t[idx])
            loss = F.mse_loss(values, ret_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())
```
- यह **"critic warm-up"** function है — सिर्फ़ value (critic) को train करता है, policy (actor) को हाथ नहीं लगाता।
- `obs_t = torch.from_numpy(np.stack(obs).reshape(len(obs), -1)).float()`: सारे collected observations को एक साथ एक बड़े tensor में जोड़ा (`np.stack`), फिर flatten किया, फिर float में बदला।
- `ret_t`: targets (returns) को भी tensor में बदला।
- `idx_all = np.arange(obs_t.shape[0])`: सारे data-points के index — `[0, 1, 2, ..., N-1]`।
- `for _ in range(epochs):`: पूरे data को कई बार (epochs) दोहराते हैं।
  - `np.random.shuffle(idx_all)`: हर epoch में indices को फेंट (shuffle) दिया, ताकि हर बार अलग order में mini-batches बनें (यह model को किसी खास order पर निर्भर होने से बचाता है)।
  - `for start in range(0, len(idx_all), batch_size):`: पूरा data छोटे-छोटे **mini-batches** में बाँटा (`batch_size=256` एक बार में)।
    - `idx = idx_all[start:start + batch_size]`: इस batch के indices।
    - `_, values = ac(obs_t[idx])`: सिर्फ़ इन्हीं samples को नेटवर्क से गुज़ारा, logits (policy का हिस्सा) फेंक दिया (`_`), सिर्फ़ value चाहिए।
    - `loss = F.mse_loss(values, ret_t[idx])`: **Mean Squared Error** — नेटवर्क का अनुमान (`values`) असली target (`returns`) से कितना अलग है, उसका औसत वर्ग-अंतर।
    - `opt.zero_grad(); loss.backward(); opt.step()`: यह PyTorch training का standard 3-step पैटर्न है:
      1. `zero_grad()`: पुराने gradients मिटा दो (नहीं तो नए gradients पुराने में जुड़ जाएँगे, ग़लत होगा)।
      2. `loss.backward()`: **backpropagation** — loss से हर parameter तक gradient (कितना और किस दिशा में बदलना चाहिए) calculate किया।
      3. `opt.step()`: उन gradients के आधार पर parameters को असल में update कर दिया (Adam optimizer के rule से)।
- आख़िरी loss value लौटा दी (सिर्फ़ printing/monitoring के लिए)।

```python
def ppo_update(ac, opt, obs, actions, old_logp, advantages, returns,
                epochs=2, batch_size=256, clip=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5):
    obs_t = torch.from_numpy(np.stack(obs).reshape(len(obs), -1)).float()
    actions_t = torch.tensor(actions, dtype=torch.long)
    old_logp_t = torch.tensor(old_logp, dtype=torch.float32)
    adv_t = torch.tensor(advantages, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
```
- यह असली **PPO update** है — policy (correction-head + trunk) और value दोनों को एक साथ train करता है।
- सारे data को tensors में बदला — actions को `dtype=torch.long` (integer type, क्योंकि यह category/index हैं, decimal नहीं)।
- `adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)`: **advantage-normalization** — advantages का mean 0 और standard-deviation 1 पर normalize कर दिया। यह training को स्थिर (stable) बनाने की एक बहुत common technique है — बिना इसके, अलग-अलग batches में advantages का scale बहुत अलग-अलग हो सकता है, जिससे learning-rate असंतुलित असर डालती।

```python
    idx_all = np.arange(obs_t.shape[0])
    last_loss = 0.0
    for _ in range(epochs):
        np.random.shuffle(idx_all)
        for start in range(0, len(idx_all), batch_size):
            idx = idx_all[start:start + batch_size]
            b_obs, b_actions, b_old_logp = obs_t[idx], actions_t[idx], old_logp_t[idx]
            b_adv, b_ret = adv_t[idx], ret_t[idx]
```
- वैसे ही mini-batch loop जैसे पहले।
- `b_obs, b_actions, ...`: इस mini-batch के सारे संबंधित tensors एक साथ निकाल लिए (variable नामों में `b_` prefix "batch" के लिए)।

```python
            logits, values = ac(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
```
- `logits, values = ac(b_obs)`: **अभी के (updated हो रहे) नेटवर्क** से दोबारा logits/values निकाले — ध्यान दें, यह वही नेटवर्क है जो rollout के वक़्त इस्तेमाल हुआ था, मगर training के दौरान (कई updates के बाद) यह थोड़ा बदल चुका होगा — इसीलिए PPO को "ratio" (नीचे) की ज़रूरत पड़ती है।
- `logp = dist.log_prob(b_actions)`: उन्हीं actions की, **अभी के नेटवर्क के हिसाब से** log-probability निकाली (जो rollout के वक़्त वाली `b_old_logp` से अलग हो सकती है)।
- `entropy = dist.entropy().mean()`: distribution की entropy (कितनी "अनिश्चितता"/फैलाव है probabilities में) — इस्तेमाल exploration-bonus के तौर पर हो सकता है, हालाँकि यहाँ `ent_coef=0.0` है तो असर में यह zero-weighted है (कोड में रखा है ताकि भविष्य में आसानी से चालू किया जा सके)।

```python
            ratio = torch.exp(logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
```
- यही PPO का सबसे मशहूर हिस्सा है — **clipped surrogate objective**।
- `ratio = torch.exp(logp - b_old_logp)`: `exp(log(a) - log(b)) = a/b`, यानी यह **नई probability / पुरानी probability** का अनुपात है — कितना policy बदल चुकी है इस action के लिए।
- `surr1 = ratio * b_adv`: सीधा-सादा policy-gradient objective (अगर advantage positive है, तो ratio बढ़ाने से loss घटेगा — यानी उस action की probability बढ़ेगी)।
- `surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv`: वही, लेकिन `ratio` को `[1-clip, 1+clip]` यानी `[0.8, 1.2]` के बीच "clamp" (जबरन सीमित) कर दिया — ताकि एक ही update में policy बहुत ज़्यादा ना बदल जाए (बड़ी छलांग खतरनाक होती है, training अस्थिर कर सकती है)।
- `-torch.min(surr1, surr2).mean()`: दोनों में से जो **छोटा** (ज़्यादा "सावधान") है, वही चुना — यही "pessimistic bound" है, PPO का मूल विचार। Minus इसलिए क्योंकि हम loss को **minimize** करेंगे, जबकि यह असल में एक "objective" है जिसे maximize करना है — minus लगाकर maximize को minimize में बदल दिया (gradient-descent frameworks हमेशा minimize करते हैं)।

```python
            value_loss = F.mse_loss(values, b_ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            opt.step()
            last_loss = float(loss.item())
    return last_loss
```
- `value_loss`: critic की गलती (जैसे पहले `value_only_update` में)।
- `loss = policy_loss + vf_coef * value_loss - ent_coef * entropy`: तीनों को जोड़कर एक ही अंतिम loss बनाया — `vf_coef=0.5` value-loss को कितना वज़न देना है, `ent_coef=0.0` entropy को (यहाँ बंद)। Minus entropy इसलिए कि entropy बढ़ाना चाहते हैं (ज़्यादा entropy = loss को घटाना, इसलिए minus)।
- `opt.zero_grad(); loss.backward(); opt.step()`: वही 3-step training pattern।
- `nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)`: **gradient clipping** — अगर gradients बहुत बड़े हो जाएँ (जो training को अस्थिर कर सकता है), तो उन्हें एक अधिकतम सीमा (`max_grad_norm=0.5`) तक "काट" दिया। यह `backward()` के बाद और `step()` से पहले करना ज़रूरी है।
- `last_loss = float(loss.item())`: आख़िरी batch की loss-value याद रखी, monitoring के लिए।

```python
def train_ppo(n_bands=N_BANDS, n_slots=EPISODE_SLOTS, n_updates=150, episodes_per_update=24,
              gamma=0.99, lam=0.95, lr=1e-4, epochs=2, batch_size=256, clip=0.2,
              vf_coef=0.5, ent_coef=0.0, seed=0, critic_warmup_updates=15):
    ac = ActorCritic(n_bands)
    # index_weights is only 4 shared numbers steering every band's score at once -- a single
    # noisy gradient step there can degrade behaviour everywhere simultaneously. Freeze it as
    # a fixed expert prior; RL only ever learns the local, bounded, per-state correction.
    ac.index_weights.requires_grad_(False)
```
- `train_ppo`: यह पूरे training को coordinate करने वाला "मुख्य" function है — बाक़ी सारे functions (collect_rollout, compute_gae, value_only_update, ppo_update) को सही क्रम में बुलाता है।
- `ac = ActorCritic(n_bands)`: एक नया, untrained network बनाया — warm-started (जैसा ऊपर देखा)।
- `ac.index_weights.requires_grad_(False)`: `requires_grad_(False)` से PyTorch को बताया — "इस parameter का gradient मत निकालो, इसे कभी मत बदलो"। Comment में वजह साफ़ है — यह सिर्फ़ 4 संख्याएँ हैं जो **हर** बैंड के score को प्रभावित करती हैं, इसलिए इन्हें ग़लती से बदलना पूरी policy को एक साथ बिगाड़ सकता है — इसलिए इसे स्थिर (frozen), "भरोसेमंद विशेषज्ञ की राय" जैसा रखा।

```python
    value_opt = torch.optim.Adam(list(ac.trunk.parameters()) + list(ac.value_head.parameters()), lr=lr)
    full_opt = torch.optim.Adam(
        list(ac.trunk.parameters()) + list(ac.correction_head.parameters()) + list(ac.value_head.parameters()),
        lr=lr)
    master_rng = np.random.default_rng(seed)
    reward_history = []
```
- `value_opt`: सिर्फ़ trunk और value_head को update करने वाला optimizer — यह critic-warmup के दौरान इस्तेमाल होगा।
- `full_opt`: trunk, correction_head, और value_head — तीनों को update करने वाला optimizer (लेकिन `index_weights` शामिल नहीं, क्योंकि वो freeze है) — यह असली PPO-phase में इस्तेमाल होगा।
- `torch.optim.Adam(...)`: **Adam** एक बहुत लोकप्रिय optimization-algorithm है, जो हर parameter के लिए अपने-आप learning-rate adjust करता रहता है — साधारण gradient-descent से ज़्यादा तेज़ और स्थिर मानी जाती है।
- `master_rng`: इस पूरी training-run के लिए scenario-seeds चुनने वाला generator।
- `reward_history = []`: हर episode का reward, बाद में plot करने के लिए।

```python
    for update in range(n_updates):
        obs, actions, logp, rewards, values, dones, ep_rewards = collect_rollout(
            ac, episodes_per_update, n_bands, n_slots, master_rng)
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        if update < critic_warmup_updates:
            value_only_update(ac, value_opt, obs, returns, epochs, batch_size)
        else:
            ppo_update(ac, full_opt, obs, actions, logp, advantages, returns,
                       epochs, batch_size, clip, vf_coef, ent_coef)
        reward_history.extend(ep_rewards)

    return PPOScheduler(ac, n_bands, deterministic=True), reward_history
```
- `for update in range(n_updates):`: कुल `n_updates=150` बार यह चक्र दोहराया।
  - `collect_rollout(...)`: नई (मौजूदा policy से) experience इकट्ठा की।
  - `compute_gae(...)`: उससे advantage/returns निकाले।
  - `if update < critic_warmup_updates:`: शुरुआती `15` updates तक — सिर्फ़ critic (value) को train करो, actor को हाथ मत लगाओ।
  - `else:`: उसके बाद — असली PPO update (actor + critic दोनों, मगर सीमित correction के साथ)।
  - `reward_history.extend(ep_rewards)`: इस batch के episode-rewards को बड़ी list में जोड़ दिया (`.extend` — पूरी list को individually जोड़ता है, `.append` की तरह पूरी list को एक ही item की तरह नहीं जोड़ता)।
- आख़िर में, trained network को `PPOScheduler` में लपेटकर (wrap करके) — साथ में पूरा reward-history — दोनों लौटा दिए।

```python
ppo_scheduler, ppo_rewards = train_ppo(n_updates=150, episodes_per_update=24, seed=RNG_SEED)
```
- असली training यहाँ चलती है — 150 updates, हर update में 24 नए random episodes, हमारी fixed `RNG_SEED` के साथ।

```python
window = 10
smoothed = np.convolve(ppo_rewards, np.ones(window) / window, mode="valid")
plt.figure(figsize=(8, 3.5))
plt.plot(ppo_rewards, alpha=0.3, label="per-episode reward")
plt.plot(range(window - 1, len(ppo_rewards)), smoothed, label=f"{window}-episode moving average")
plt.axvline(24 * 15, color="red", linestyle="--", alpha=0.6, label="end of critic warm-up")
plt.xlabel("training episode (fresh randomized scenario each time)")
plt.ylabel("total hits per 150-slot episode")
plt.title("Residual PPO training curve under domain randomization")
plt.legend()
plt.tight_layout()
plt.show()
```
- `window = 10`: moving-average के लिए 10 episodes की खिड़की।
- `np.convolve(ppo_rewards, np.ones(window)/window, mode="valid")`: यह हर 10 consecutive rewards का average निकालता है, आगे खिसकते हुए — इससे noisy raw-data का एक smooth ट्रेंड-लाइन दिखता है। `np.ones(window)/window` एक "averaging filter" है (हर एक 1/10 वज़न का)। `mode="valid"` का मतलब सिर्फ़ वही हिस्से रखो जहाँ पूरी खिड़की data के अंदर fit हो (किनारों पर अधूरी खिड़की नहीं)।
- `plt.plot(ppo_rewards, alpha=0.3, ...)`: असली (noisy) data, हल्के (30% opacity) रंग में — background की तरह।
- `plt.plot(range(window-1, len(ppo_rewards)), smoothed, ...)`: smoothed line, ऊपर से मोटी/स्पष्ट दिखेगी। `range(window-1, ...)` इसलिए क्योंकि पहला smoothed-point सिर्फ़ 10वें episode के बाद ही बन सकता है।
- `plt.axvline(24 * 15, ...)`: एक खड़ी (vertical) लकीर, ठीक वहाँ जहाँ critic-warmup ख़त्म हुआ (15 updates × 24 episodes/update = episode-number 360) — visual reference के लिए।
- बाकी labels/title/legend/save वही matplotlib वाले standard steps।

---

## Code Cell 10 — Held-out Evaluation Scenarios (Section 7)

```python
def make_scenarios(n, base_seed):
    scenarios = []
    for i in range(n):
        seed = base_seed + i
        emitters = make_random_scenario(seed=seed)
        truth, owner, amp_grid, _ = build_truth_and_pdws(emitters, seed=seed)
        scenarios.append((emitters, truth, owner, amp_grid))
    return scenarios

EVAL_SCENARIOS = make_scenarios(25, base_seed=777_000)   # disjoint from training seed range
print(f"Built {len(EVAL_SCENARIOS)} held-out evaluation scenarios.")
```
- `make_scenarios(n, base_seed)`: `n` अलग-अलग scenarios बनाता है, हर एक अलग seed (`base_seed + i`) के साथ।
- `scenarios.append((emitters, truth, owner, amp_grid))`: हर scenario की चार ज़रूरी चीज़ें एक tuple में जोड़कर list में save कर लीं।
- `EVAL_SCENARIOS = make_scenarios(25, base_seed=777_000)`: 25 scenarios, seed-range `777_000` से शुरू — यह range training में इस्तेमाल हुए seeds (`master_rng.integers(0, 1_000_000)` — यानी 0 से 999,999 के बीच कहीं भी) से टकरा **सकता** था, लेकिन practically training randomly चुने गए seeds में इतनी बड़ी specific संख्या दोबारा आने की संभावना बहुत ही कम है — फिर भी, ज़्यादा सफ़ाई के लिए **Section 8d के sweep में** हमने training-seed और eval-seed को पूरी तरह अलग-अलग range में रखा है।
- Comment: "disjoint from training seed range" — यानी training और evaluation के scenarios अलग-अलग हों, नहीं तो नेटवर्क "रटे हुए" जवाब देगा, असली general-strategy नहीं दिखाएगा — यह ML में एक बुनियादी नियम है (train/test split)।

---

## Code Cell 11 — `evaluate_scheduler` + तुलना-तालिका (Section 8)

```python
def evaluate_scheduler(scheduler, scenarios):
    total_scans = total_hits = 0
    true_on_scans = hits_when_on = true_off_scans = false_alarms = 0
    reward_sum = 0.0
    first_true_on, first_intercept = {}, {}
    prediction_errors = []
```
- यह function किसी भी scheduler को कई scenarios पर चलाकर, **सारे figures-of-merit** एक साथ निकालता है।
- कई counters शुरू में शून्य पर सेट किए — एक ही लाइन में कई variables को एक जैसा मान देना (`a = b = 0`) Python में valid है।
- `first_true_on, first_intercept = {}, {}`: दो dictionaries — पहला "कौन सा emitter कब पहली बार सच में ON हुआ" रिकॉर्ड करेगा, दूसरा "कब पहली बार पकड़ा गया"।
- `prediction_errors = []`: periodicity-predictions की सटीकता मापने के लिए खाली list।

```python
    for si, (emitters, truth, owner, amp_grid) in enumerate(scenarios):
        env = ScanEnv(truth, amp_grid=amp_grid, seed=9000 + si)
        scheduler.reset(env)

        for e in emitters:
            slots = np.where(owner == e.emitter_id)[1]
            if len(slots):
                first_true_on[(si, e.emitter_id)] = int(slots.min())
```
- `enumerate(scenarios)`: हर scenario को उसके index (`si`) के साथ लूप में लिया।
- `env = ScanEnv(..., seed=9000+si)`: हर scenario के लिए, evaluation में इस्तेमाल होने वाली randomness (detection सफल/असफल) भी हर scenario के लिए अलग मगर reproducible seed से तय होती है।
- `for e in emitters:`: हर सच्चे emitter के लिए —
  - `slots = np.where(owner == e.emitter_id)[1]`: `owner` grid में जहाँ-जहाँ यह emitter-id मिलता है, वो जगहें ढूँढीं। `np.where(condition)` दो arrays देता है (row-indices, column-indices); `[1]` से हमें सिर्फ़ column-indices (यानी time-slots) चाहिए, क्योंकि हमें सिर्फ़ "कब" चाहिए, "किस बैंड में" नहीं।
  - `if len(slots): first_true_on[(si, e.emitter_id)] = int(slots.min())`: अगर यह emitter कभी ON हुआ ही था, तो सबसे पहला (`min()`) समय record कर लिया — dictionary की "key" यहाँ एक **tuple** `(scenario-index, emitter-id)` है, ताकि अलग-अलग scenarios के emitters आपस में गड्ड-मड्ड ना हों (भले ही उनकी id वही 0, 1, 2... हो)।

```python
        for _ in range(env.n_slots):
            t = env.t
            action = scheduler.select(env)
            obs, reward, done, info = env.step(action)
            scheduler.observe(env, action, t, reward, info)

            total_scans += 1
            reward_sum += reward
            if info["detect"]:
                total_hits += 1
                eid = int(owner[action, t])
                if eid != -1:
                    key = (si, eid)
                    if key not in first_intercept:
                        first_intercept[key] = t
```
- वही standard episode-loop (जैसे `run_episode` में)।
- `total_scans += 1`, `reward_sum += reward`: overall counters बढ़ाए।
- `if info["detect"]:`: अगर hit हुआ —
  - `eid = int(owner[action, t])`: उस बैंड-समय पर सच में कौन सा emitter था, वो निकाला (यह सिर्फ़ evaluation के लिए है, scheduler को यह नहीं दिखता)।
  - `if eid != -1:`: अगर वाकई कोई emitter था (ना कि false-alarm, जहाँ owner `-1` होता)।
    - `key = (si, eid)`: उसी tuple-key style का इस्तेमाल।
    - `if key not in first_intercept: first_intercept[key] = t`: सिर्फ़ **पहली बार** पकड़े जाने का समय रिकॉर्ड करो (अगर पहले से मौजूद है, तो दोबारा मत बदलो — हमें सिर्फ़ पहला intercept चाहिए)।

```python
            if info["true_state"]:
                true_on_scans += 1
                hits_when_on += int(info["detect"])
            else:
                true_off_scans += 1
                false_alarms += int(info["detect"])

            if done:
                break

        if hasattr(scheduler, "prediction_log"):
            prediction_errors += [abs(pt - at) for _, pt, at in scheduler.prediction_log]
```
- `if info["true_state"]:`: अगर सच में यह बैंड ON था —
  - `true_on_scans += 1`: "true-ON scans" का counter बढ़ाया (Pd के denominator के लिए)।
  - `hits_when_on += int(info["detect"])`: अगर detect भी हुआ, तो numerator बढ़ाया (Pd का numerator)।
- `else:`: अगर सच में OFF था —
  - `true_off_scans += 1`, `false_alarms += int(info["detect"])`: वैसे ही Pfa के लिए।
- `if hasattr(scheduler, "prediction_log"):`: `hasattr` जाँचता है कि क्या इस scheduler के पास `prediction_log` नाम की कोई attribute है या नहीं — सिर्फ़ `BeliefPeriodicityScheduler` के पास यह है (बाकी schedulers के पास नहीं, जैसे RoundRobin)। यह एक "duck typing" तरीका है — बिना यह पूछे "तुम कौन सी class हो", सीधे पूछ लिया "क्या तुम्हारे पास यह चीज़ है"।
- `prediction_errors += [abs(pt - at) for _, pt, at in scheduler.prediction_log]`: हर (बैंड, predicted-time, actual-time) triple से predicted और actual के बीच का absolute अंतर निकाला, और पूरी list को मौजूदा `prediction_errors` में जोड़ दिया (`+=` लिस्ट को दूसरी लिस्ट से जोड़ता है)।

```python
    delays = [first_intercept[k] - first_true_on[k] for k in first_true_on if k in first_intercept]

    return {
        "scheduler": scheduler.name,
        "Pd": hits_when_on / max(true_on_scans, 1),
        "Pfa": false_alarms / max(true_off_scans, 1),
        "avg_intercept_rate_per_s": (total_hits / max(total_scans, 1)) / (SLOT_DURATION_US * 1e-6),
        "avg_reward": reward_sum / max(total_scans, 1),
        "pct_correct_predictions": total_hits / max(total_scans, 1),
        "interception_ratio": len(delays) / max(len(first_true_on), 1),
        "avg_intercept_delay_slots": float(np.mean(delays)) if delays else np.nan,
        "avg_intercept_time_error_slots": float(np.mean(prediction_errors)) if prediction_errors else np.nan,
    }
```
- `delays = [first_intercept[k] - first_true_on[k] for k in first_true_on if k in first_intercept]`: सिर्फ़ उन emitters के लिए जो **कभी पकड़े ही गए** (`if k in first_intercept`), "पकड़े जाने का समय माइनस पहली बार ON होने का समय" निकाला — यही **intercept delay** है।
- अब पूरा result एक dictionary में तैयार किया, हर key एक metric है:
  - `"Pd": hits_when_on / max(true_on_scans, 1)`: definition के मुताबिक Pd। `max(..., 1)` से divide-by-zero से बचाव (अगर कभी कोई true-ON scan ही ना हुआ हो)।
  - `"Pfa"`: वैसे ही false-alarm rate।
  - `"avg_intercept_rate_per_s"`: कुल hits ÷ कुल scans (per-scan hit-rate), फिर उसे per-second में बदला — `SLOT_DURATION_US * 1e-6` से microseconds को seconds में बदला (`1e-6` = 0.000001), फिर उससे divide करने से "per-slot rate" "per-second rate" बन जाती है।
  - `"avg_reward"`: औसत reward प्रति scan।
  - `"pct_correct_predictions"`: कुल कितने % scans hit रहे।
  - `"interception_ratio": len(delays) / max(len(first_true_on), 1)`: कितने % emitters कभी-ना-कभी पकड़े गए (सारे true-ON हुए emitters में से)।
  - `"avg_intercept_delay_slots": float(np.mean(delays)) if delays else np.nan`: औसत delay, लेकिन अगर कोई delay data ही नहीं (`delays` खाली list) तो `np.nan` (Not-a-Number, यानी "उपलब्ध नहीं") — ताकि कोई ग़लत/भ्रामक शून्य ना दिखे।
  - `"avg_intercept_time_error_slots"`: वैसे ही, prediction-error का औसत, सिर्फ़ periodicity-tracking वाले schedulers के लिए मौजूद होगा (बाकियों के लिए `NaN`)।

```python
schedulers = [
    RoundRobinScheduler(),
    RandomScheduler(),
    GreedyBeliefScheduler(),
    BeliefPeriodicityScheduler(),
    ppo_scheduler,
]

results = pd.DataFrame([evaluate_scheduler(s, EVAL_SCENARIOS) for s in schedulers]).set_index("scheduler")
results.round(3)
```
- `schedulers = [...]`: सारे पाँच schedulers की एक list — पहले चार के नए (fresh) objects बनाए, आख़िरी (`ppo_scheduler`) पहले से ही trained है इसलिए वही इस्तेमाल कर लिया।
- `[evaluate_scheduler(s, EVAL_SCENARIOS) for s in schedulers]`: हर scheduler के लिए evaluation चलाया, हर एक का result-dictionary एक list में इकट्ठा हुआ।
- `pd.DataFrame(...)`: उस list-of-dictionaries को एक साफ़-सुथरी table (हर dictionary एक row, हर key एक column) में बदल दिया — pandas इसे बहुत आसानी से करता है।
- `.set_index("scheduler")`: "scheduler" कॉलम को row-labels (index) बना दिया, ताकि table पढ़ने में आसान लगे (हर row का नाम सीधे scheduler का नाम है)।
- `results.round(3)`: सारी संख्याओं को 3 decimal-places तक round करके दिखाया (सिर्फ़ display के लिए, असली data नहीं बदला)।

---

## अंत में — एक ज़रूरी याद-दिलाने वाली बात (सारांश)

पूरे notebook का सबसे बड़ा **design-principle** यह है: **हर layer अगले पर निर्भर है, पर पिछले की "internal जानकारी" पर नहीं** —
- `Emitter` classes सिर्फ़ ON/OFF pattern जानती हैं।
- `ScanEnv` सिर्फ़ `(truth, amp_grid)` arrays पर निर्भर है — चाहे वो synthetic हों या असली TSRD डेटा से आई हों, फ़र्क़ नहीं पड़ता।
- हर `Scheduler` सिर्फ़ `env.belief`, `env.t` जैसी public जानकारी देखता है — कभी `env.truth` (असली सच्चाई) को सीधे नहीं छूता — यही "no prior reliable intelligence" को सही मायने में लागू करना है, सिर्फ़ बातों में नहीं।
- `evaluate_scheduler` किसी भी scheduler को (चाहे simple heuristic हो या trained neural-network) बिल्कुल एक जैसे तरीके से test कर सकता है, क्योंकि सबने एक ही `Scheduler` interface (`reset`, `select`, `observe`) अपनाया है।

यही कारण है कि Section 9 में "असली data पर स्विच करना" सिर्फ़ **एक फ़ंक्शन** (`load_tsrd_pulse_train`) बदलने जैसा आसान काम है — बाक़ी सैकड़ों लाइनें बिना छुए वैसे ही चलती रहेंगी।
