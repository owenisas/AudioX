# AudioX Prompt Skills Guide

What AudioX can generate, and how to prompt it effectively.

---

## Base Model Capabilities (AudioX-MAF-MMDiT)

The base model was trained on the IF-Caps dataset (1.3M+ audio, 5.7M music samples) and
supports six generation tasks:

| Task | What it does |
|------|-------------|
| **Text → Audio (T2A)** | Sound effects, ambience, foley from text descriptions |
| **Text → Music (T2M)** | Musical compositions from genre/mood/instrument descriptions |
| **Video → Audio (V2A)** | Synchronized audio for silent video |
| **Video → Music (V2M)** | Background music matched to video content |
| **Text+Video → Audio (TV2A)** | Video-guided sound effects with text steering |
| **Audio continuation** | Extend a previous audio chunk coherently |

---

## Prompt Structure

AudioX uses a T5-base text encoder (256 token max). Prompts should be **descriptive
natural language** — not tags, not keywords. Describe what you *hear*, not what you *see*.

### Anatomy of a good prompt

```
[sound source] + [action/texture] + [spatial detail] + [qualities] + [environment]
```

### Examples by category

**Ambient / Nature**
```
Gentle rain falling on a tin roof, distant thunder rumbling, soft wind
through open windows, cozy indoor atmosphere

Ocean waves crashing rhythmically on rocky shore, seagulls calling
overhead, salt spray, wide stereo field

Dense tropical rainforest at dawn, layered insect chirps, distant bird
calls, occasional leaf rustling, warm humid air, binaural recording
```

**Sound Effects / Foley**
```
Wooden door creaking open slowly, hinges squeaking, followed by
heavy footsteps on a hardwood floor

Glass breaking and shattering on concrete, sharp initial crack followed
by tinkling scattered fragments

Old typewriter keys clacking rhythmically, carriage return bell, paper
feeding mechanism
```

**Music**
```
Calm lo-fi hip hop beat, vinyl crackle, mellow piano chords, soft
brushed drums, warm bass, relaxed tempo

Epic orchestral trailer music, brass fanfare, deep timpani hits,
soaring string section, building to climax

Acoustic fingerpicking guitar, folk style, gentle and melodic,
natural reverb, solo instrument
```

---

## LoRA Fine-Tune: ASMR / Binaural Sound Effects

The LoRA was trained on 537 chapters of ASMR content from `@AsmrMomoNobara`,
covering 50+ distinct trigger categories. It responds best to prompts that match
the training data style.

### Top trigger categories (by training frequency)

| Category | Count | Best prompt keywords |
|----------|-------|---------------------|
| Whisper | 276 | soft whisper, close-mic, breathy, intimate |
| Fabric sounds | 132 | fabric rubbing, cloth texture, silk, cotton |
| Mouth sounds | 102 | wet clicks, pops, sticky smacking, lip sounds |
| Breathing | 72 | gentle exhale, soft breathing, airy, soothing |
| Scratching | 55 | scratching, crunchy texture, fingernails, rhythmic |
| Brushing | 49 | soft brushing strokes, delicate, bristle texture |
| Tapping | 45 | gentle tapping, fingertip, wood, rhythmic |
| Ear blowing | 17 | gentle ear blowing, airy, breathy, intimate |
| Liquid sounds | 14 | water drops, liquid, wet texture |
| Ear cupping | 11 | deep low-frequency cupping, pressure, plunger |
| Heartbeat | 11 | deep rhythmic heartbeat, low frequency, steady |
| Lotion sounds | 8 | lotion squishing, wet creamy texture |
| Stress ball | 11 | squishing, crinkling, rubbery texture |

### ASMR prompt patterns that work

**Describe the physical action and texture, not the category name:**

```
# Good — specific, textural, spatial
Thin wooden stick gently scraping inside ear canal microphone, slow
rhythmic movements with soft crackling, faint ambient hum in background

Soft delicate brushing strokes on microphone capsule, bristle texture,
slow circular motion, gradually diminishing presence, stereo

Dense rapid-fire wet mouth clicks and pops, sticky smacking noise,
layered and tingly, right ear focused

Deep powerful low-frequency ear cupping, plunger-like pressure effect,
panning between left and right channels, binaural

# Bad — vague, category-only
ASMR ear cleaning video
Relaxing ASMR sounds
Tapping and scratching
```

### Spatial / binaural keywords

The training data includes explicit spatial annotations. Use these:

```
left ear          — sound focused in left channel
right ear         — sound focused in right channel
center            — mono center image
stereo            — wide stereo field
panned            — moving between left and right
binaural          — full 3D spatial effect
close-mic         — intimate proximity
```

Example with spatial control:
```
Gentle ear blowing alternating between left and right ears, creating
a dynamic panning effect, each breath soft but distinct, intimate
binaural recording
```

### Intensity and frequency descriptors

```
# Intensity
gentle, soft, delicate       — low intensity
rhythmic, steady, moderate   — medium intensity
aggressive, fast, intense    — high intensity

# Frequency
deep, low-frequency, rumbling, bass    — sub-bass / low
warm, resonant, full-bodied            — mid-range
crisp, sharp, high-frequency, bright   — high / presence
airy, breathy, sibilant               — air / ultra-high
```

### Continuation prompts (chaining segments)

When generating sequences with `audio_prompt` (previous chunk as conditioning):

```
# First chunk — establish the scene
Quiet room ambience, faint air conditioning hum, soft breathing,
binaural microphone setup

# Second chunk — introduce trigger
Gentle fingertip tapping on wooden surface, slow rhythmic pattern,
close to right ear, continuing room ambience

# Third chunk — vary and build
Faster tapping transitions to soft scratching on textured surface,
fingernails creating crunchy patterns, panning left to right
```

---

## Tips

1. **Be specific about materials and surfaces**: "wooden stick on silicone ear"
   beats "tapping sounds". The model learned from detailed descriptions.

2. **Include temporal flow**: "starting slow, building to faster rhythm" or
   "gradually fading" gives the model structure over the 10-second window.

3. **Layer multiple sounds**: "soft whisper layered over gentle brushing with
   faint heartbeat in background" — the model handles composites.

4. **Mention recording quality**: "high quality binaural recording", "close-mic",
   "studio quality" — acts as a positive quality prior.

5. **Use the XML format** for precise event timing (supported by the IF-Caps
   training data):
   ```
   <audio><events><event name="tapping" start="0.0" end="4.0"/>
   <event name="scratching" start="4.0" end="8.0"/>
   <event name="whisper" start="8.0" end="10.0"/></events></audio>
   ```

6. **Avoid negations**: Don't say "no background noise". The model doesn't
   understand negation well. Instead describe what *should* be there.

7. **256 tokens max**: The T5 encoder truncates beyond this. Keep prompts
   detailed but not excessively long. 2-4 sentences is the sweet spot.
