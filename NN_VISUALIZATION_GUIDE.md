# Network Visualization - Complete & Labeled

## What You're Now Seeing

The neural network visualization on the `/analytics` page now displays:

### Title
**"Logistic Regression: Difficult Intubation Prediction"**

### Three Layers

#### INPUT LAYER (Left)
Shows 10 airway assessment features:
- **age** - Patient age in years
- **weight** - Patient weight in kg
- **height** - Patient height in cm
- **bmi** - Body Mass Index
- **sex** - Male (0) or Female (1)
- **dtm** - Thyromental distance (cm)
- **dii** - Interincisor distance (cm)
- **mallampati** - Mallampati classification (1-4)
- **stop_bang** - STOP-BANG score (0-8)
- **alganzouri** - Alganzouri score (0-12)

#### HIDDEN LAYER (Middle)
Single logistic unit labeled **"logit"**
- This is the linear combination of inputs with their weights
- Applies sigmoid activation to produce probability

#### OUTPUT LAYER (Right)
Labeled **"Diff. Risk"** (Difficulty Risk)
- Outputs probability of difficult intubation (0-1)
- Red color indicates this is the "difficult" prediction

### Color Coding

**Blue lines** = positive weight
- Increases the logit value
- Increases difficulty risk
- Example: Lower thyromental distance (DTM) typically increases difficulty risk

**Red lines** = negative weight
- Decreases the logit value
- Decreases difficulty risk
- Example: Lower Mallampati score indicates easier intubation

### Line Thickness
- **Thicker lines** = stronger weight magnitude = more important feature
- **Thinner lines** = weaker weight magnitude = less important feature

### Legend
Clearly shows the meaning of blue vs red lines at the bottom

### Layer Labels
"INPUT", "HIDDEN", and "OUTPUT" clearly marked above each layer

## How to Interpret It

1. **Follow a thick blue line** from an input to the hidden layer → that feature strongly increases difficulty prediction
2. **Follow a thin red line** from an input to the hidden layer → that feature weakly decreases difficulty prediction
3. **The thicker the line from hidden to output** → the stronger influence of the logit on final prediction

## Example Interpretations

- If **DTM has a thick blue line**: Low thyromental distance strongly predicts difficulty
- If **age has a thin red line**: Age has minimal negative effect on difficulty prediction
- If **Mallampati has thick red lines**: High Mallampati score strongly indicates easier intubation (negative weight)

The visualization is now a complete, self-documenting representation of what features matter most for predicting difficult intubation!
