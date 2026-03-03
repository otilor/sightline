# Sightline: Building an AI-Powered Esports Analytics Platform

*How computer vision, sequence models, and unsupervised learning can turn raw broadcast footage into actionable competitive strategy.*

---

## 1. The Problem

Competitive Call of Duty is a game of milliseconds and inches. In the Call of Duty League (CDL), teams of four players compete across three game modes — Search and Destroy (SND), Hardpoint, and Control — in best-of-five map series. The margins between winning and losing a championship are razor-thin, and increasingly, those margins are being decided not in-game but in the preparation that happens before a team ever picks up a controller.

Every CDL match is broadcast live, and the recordings — known as VODs (Video on Demand) — contain a staggering amount of strategic information. The broadcast overlays show player positions on a minimap, real-time statistics for every player, a kill feed documenting every engagement, and a scoreboard tracking the flow of the match. A coaching staff reviewing these VODs manually might spend 4–6 hours preparing for a single opponent, scrubbing through footage, taking notes, and building strategy from memory and intuition.

**Sightline** is a system designed to automate this process. It ingests raw broadcast VODs, extracts structured game-state data through computer vision, discovers patterns through machine learning, and generates strategic recommendations — all delivered through a web platform where team managers can scout opponents, review losses, and build playbooks.

This chapter walks through every layer of the system: from the pixel-level challenge of reading a minimap, to the mathematical elegance of clustering movement trajectories, to the practical engineering of making it all work at scale.

---

## 2. What the Broadcast Gives Us

Before diving into algorithms, we need to understand what data is actually visible in a CDL broadcast frame. The broadcast HUD (heads-up display) is surprisingly information-dense:

```
┌──────────────────────────────────────────────────────────┐
│ FAZE ROSTER         SCOREBOARD / CLOCK         OPP ROSTER│
│ names, K/D,       score, time, mode,          names, K/D,│
│ strk, time        hill timer, BO5             strk, time │
├──────────────┬───────────────────────┬───────────────────┤
│  KILL FEED   │                       │                   │
│  (mid-left)  │      GAMEPLAY         │                   │
│              │   (first-person)      │                   │
├──────────────┤                       ├───────────────────┤
│  MINIMAP     │                       │ SPECTATED PLAYER  │
│  (bot-left)  │                       │ (bot-right)       │
└──────────────┴───────────────────────┴───────────────────┘
```

Six distinct data regions, each requiring its own extraction strategy:

**Roster Tables** (top-left and top-right) list all eight players — four per team — with their current kills, deaths, streak count, and time on objective. This is a continuously-updating stat sheet that eliminates the need to track individual performances manually.

**The Scoreboard** (top-center) shows team scores, the game clock, the current mode label (e.g., "HARDPOINT"), and mode-specific information like hill timers or round counts. This is the primary signal for understanding the macro state of the game.

**The Kill Feed** (mid-left) is the most time-sensitive element. Each kill appears as a transient entry — visible for roughly three seconds before scrolling away — showing the killer's name, a weapon icon, and the victim's name. Miss a frame, and you've lost an event.

**The Minimap** (bottom-left) is the strategic goldmine. It shows a top-down view of the map with colored dots representing every player. The positions update in real time, and over the course of a round, they paint a picture of how a team moves, controls territory, and executes plays.

**The Spectated Player Panel** (bottom-right) shows details about whichever player the broadcast camera is currently following — their name, weapon, ability status, and a timer.

The core insight of Sightline is that all of this information can be extracted programmatically, transformed into structured data, and fed into models that find patterns no human could spot across 20 hours of footage.

---

## 3. The First Problem: Finding the Game

CDL broadcasts don't start with gameplay. There's an analyst desk, player introductions, sponsor segments, replays from previous matches, and halftime breaks. A typical one-hour VOD might contain only 35–40 minutes of actual gameplay. Processing non-gameplay frames wastes compute and introduces garbage data.

The solution is elegant in its simplicity: **the minimap is only visible during live gameplay**. It disappears during every non-gameplay segment — desk segments, replays, transitions, ads. So the first pass through any VOD is a cheap minimap-presence scan.

At 0.5 frames per second, we extract a frame every two seconds and check the bottom-left region of interest (ROI) for the presence of minimap-like content via template matching. This produces a list of `(start_sec, end_sec)` time windows where gameplay is happening. The cost is trivial — an hour-long VOD produces only 1,800 frames at this rate, and the template match is a few milliseconds per frame.

Secondary signals can be stacked for confidence: the kill feed region having text content, the scoreboard region having a specific format, or the absence of large centered text (which appears during production segments). But minimap presence alone is sufficient in practice.

---

## 4. Adaptive Frame Sampling

Not all moments in a match are created equal. A stretch of 15 seconds where players are running from spawn to their positions contains far less strategic information than the two seconds around a crucial first-blood engagement. Sampling at a flat rate wastes either compute (too high) or information (too low).

Sightline uses a three-tier adaptive sampling strategy:

**Tier 1 — Discovery (0.5 fps).** The cheapest pass, used only to locate gameplay windows within the VOD. No heavy models run here.

**Tier 2 — Tactical (4–5 fps).** The primary extraction rate, active during confirmed gameplay. At this rate, a 40-minute gameplay window produces approximately 12,000 frames — enough to track player positions with sub-second granularity. All extraction pipelines (minimap, OCR, kill feed) operate at this tier.

**Tier 3 — Burst (10+ fps).** Triggered automatically when an engagement is detected — typically when a player dot suddenly disappears from the minimap, indicating a kill. The system ramps up to 10+ fps for ±5 seconds around the event, capturing the micro-positioning and sequencing of trades that determine round outcomes.

All frames are downscaled to 720p on ingest. This resolution is sufficient for every extraction task (the minimap crop is only ~200×150 pixels, the roster text is large and high-contrast) while cutting pixel count by 56% compared to 1080p and 85% compared to 4K.

---

## 5. Seeing Through the Minimap

The minimap is a small, flat, top-down representation of the game map. On it, colored dots represent players. The problem is deceptively simple to state — find the dots and track them — and deceptively complex to solve well.

### 5.1 Detecting Player Dots

We train a **YOLOv8-nano** model on annotated minimap crops. The training set is bootstrapped from approximately 200–300 manually-labeled screenshots: for each frame, the user draws bounding boxes around player dots, the bomb indicator (in SND), and objective markers. YOLOv8-nano is chosen for its speed — inference on a single minimap crop takes under 1 millisecond via ONNX Runtime, allowing us to process thousands of frames without bottlenecking.

The detection classes are deliberately simple: `player_dot`, `bomb`, and `objective`. We don't try to classify team allegiance at the detection stage — that's handled by a separate color analysis pipeline.

### 5.2 The Color Problem

Each team's player dots are rendered in a distinct color — typically some shade of blue versus some shade of red. The naive approach is to define HSV color ranges for each team and threshold. This fails spectacularly in practice:

- Broadcast color grading varies between events and even between maps
- Video compression artifacts shift pixel colors unpredictably
- The specific hues assigned to teams are not standardized across seasons
- Dots near colored map features (water, fire, foliage) get contaminated

The solution is to treat team identification as an **unsupervised clustering problem**. After detecting all player dots in a frame, we extract the dominant pixel color of each dot and convert it to the **CIELAB color space**. Unlike RGB or HSV, CIELAB is perceptually uniform — the Euclidean distance between two colors in Lab space corresponds to how different they actually look to a human eye. This means clustering in Lab space naturally separates "similar-looking" colors from "different-looking" ones.

**K-Means with k=2** on the Lab color vectors produces two clusters. These clusters correspond to the two teams. We don't need to know *what* the colors are in advance — only that there are two of them, and they're different.

To anchor which cluster corresponds to Faze versus the opponent, we use a **calibration frame** — the map loading screen or pre-round scoreboard, which clearly shows team colors alongside team names. A one-time calibration per map locks in the mapping.

### 5.3 Temporal Voting

Color clustering on a single frame is noisy. Compression artifacts, partial occlusion, and edge effects can cause a dot to be classified as the wrong team for one or two frames. The fix is **temporal voting**: for each tracked dot, we collect its team classification over a sliding window of 10 frames and take the majority vote. A dot that's classified as "Team A" in 8 out of 10 frames is unambiguously Team A. This smoothing step eliminates virtually all color misclassifications.

### 5.4 Tracking Across Frames

Individual frame detections are stateless — YOLO doesn't know that the dot at (0.3, 0.5) in frame N is the same player as the dot at (0.31, 0.51) in frame N+1. **ByteTrack** solves this by assigning persistent IDs to detections and maintaining them across frames using motion prediction (Kalman filter) and appearance matching. The output is a set of trajectories: for each player, a time series of `(x, y, t)` positions throughout the round.

---

## 6. Reading the HUD

The HUD contains four distinct data sources, each requiring its own extraction approach.

### 6.1 Roster Tables

The top-left (Faze) and top-right (opponent) panels list all eight players with their names, K/D ratios, streak counts, and time-on-objective values. We crop these regions and run **EasyOCR** to extract the text. The structured, high-contrast format of these tables (white text on dark backgrounds, consistent font, tabular layout) makes OCR highly reliable here.

Because the roster stats update slowly — a kill might happen once every few seconds — we sample these regions at 2–3 second intervals. Processing every frame would be redundant.

The parsed output is a `PlayerStatSnapshot` per player per sample: name, kills, deaths, streak, time, and the timestamp of extraction. Over the course of a match, these snapshots form per-player stat curves — how their K/D evolved, when they went on streaks, when they fell behind.

### 6.2 Scoreboard

The top-center scoreboard is extracted similarly: team scores, game clock, mode label, and mode-specific data (hill timer for Hardpoint, round counter for SND). The mode label is particularly important early in the pipeline because it determines which downstream logic applies — SND has round segmentation and bomb events, Hardpoint has rotating objectives, Control has limited lives.

### 6.3 Kill Feed

The kill feed is the most challenging extraction target because it's **transient and dense**. Kill entries appear for approximately three seconds, stack vertically, and scroll off the screen. Missing a single entry means losing a kill event.

Each kill feed entry contains: a killer name (text), a weapon icon (graphical), and a victim name (text). OCR handles the names. The weapon icons, however, are small graphical assets — not text — and require **template matching**. We build a library of weapon icon templates (approximately 20–30 unique weapons in competitive play) by cropping one clean example of each. At inference time, we compare detected icons against the library using normalized cross-correlation. Since CDL broadcasts use standardized weapon icon assets, this matching is highly reliable.

The kill feed must be processed at every frame (or at least at the tactical sampling rate of 4–5 fps) to ensure no events are missed.

### 6.4 Game Mode Classification

While the scoreboard OCR often reads the mode label directly ("HARDPOINT," "SEARCH & DESTROY"), a fallback classifier ensures robustness. A lightweight **ResNet-18** fine-tuned on ~100 labeled screenshots per mode serves as the primary classifier, with the OCR reading as a high-confidence override. The HUD itself has structural differences between modes — SND shows a round counter, Hardpoint shows a hill timer, Control shows a lives indicator — which the classifier learns to distinguish.

---

## 7. Understanding Rounds

In Search and Destroy — the mode most amenable to strategic analysis — teams play a series of rounds, alternating between attack and defense. Each round is a discrete strategic unit: a fresh start where players choose positions, execute plays, and either win or lose independently of other rounds.

Detecting round boundaries requires fusing multiple signals:

- **Score changes**: when the scoreboard increments, a round has ended
- **Screen transitions**: black frames appear between rounds
- **Round-end overlays**: "Round Won" or "Round Lost" text appears prominently
- **Kill feed gaps**: unusually long periods without kills suggest inter-round downtime

By combining these signals, we segment the continuous gameplay stream into discrete `Round` objects, each annotated with its start time, end time, outcome (win/loss for Faze), the side played (attack/defense), and the win condition (elimination, bomb detonation, bomb defuse, or time expiry). These annotated rounds are the fundamental unit of strategic analysis.

---

## 8. The Map Knowledge Problem

Raw minimap coordinates — (0.42, 0.68) — are meaningless to a coach. Strategy isn't communicated in coordinates; it's communicated in callouts: "push mid," "hold B-site headglitch," "rotate through top-green." The system needs to translate positions into the language that teams actually use.

Building a complete callout map for every competitive map is labor-intensive — each map has 30–50 named positions, and the competitive map pool rotates by season. Instead, Sightline uses a **progressive grid system**.

Every minimap is divided into a **5×5 grid**, creating 25 generic zones labeled A1 through E5:

```
     1     2     3     4     5
A  [A1]  [A2]  [A3]  [A4]  [A5]
B  [B1]  [B2]  [B3]  [B4]  [B5]
C  [C1]  [C2]  [C3]  [C4]  [C5]
D  [D1]  [D2]  [D3]  [D4]  [D5]
E  [E1]  [E2]  [E3]  [E4]  [E5]
```

The pipeline works immediately with grid labels. Strategies reference zones: "the opponent tends to stack C2 and C3 on defense." Over time, high-traffic cells are aliased to callout names via a simple per-map JSON file: `{"C2": "mid", "A4": "A-site", "E4": "B-site"}`. This means the system starts producing useful output from day one, and the output gets more readable as map knowledge is incrementally added.

The grid also provides the spatial vocabulary for the feature engineering and ML layers described in the next sections.

---

## 9. Feature Engineering

Raw position data and kill events are not directly useful for machine learning. They need to be transformed into features that capture the *behavior* behind the data. Sightline computes features at three levels.

### 9.1 Per-Player Movement Features

For each player within a round, we compute:

**Speed** — the Euclidean distance between consecutive positions divided by the time interval. A player sprinting across the map has high speed; a player holding an angle has near-zero speed.

**Grid-cell transition sequences** — the ordered list of grid cells a player passes through during a round. `[A1 → B2 → C3 → C4]` is a route through the map, encoded as a sequence of discrete symbols. This representation is powerful because it captures the *path* a player took without needing continuous coordinates, and it's directly comparable across rounds using sequence distance metrics.

**Direction-to-objective** — rather than computing raw angles (which suffer from the circular variable problem, where 359° and 1° are numerically distant but directionally adjacent), we compute the cosine similarity between the player's movement vector and the vector pointing toward each objective. A value close to 1.0 means the player is moving toward that objective; -1.0 means moving away.

**Time-in-zone** — what percentage of the round the player spent in each grid cell. This produces a 25-element histogram that serves as a "positional fingerprint" for that player in that round.

**Idle time** — consecutive frames where speed is approximately zero. Long idle times indicate a passive playstyle (holding angles, camping positions); short or zero idle times indicate constant aggression.

**First-move cell** — which grid cell the player enters first after round start. This captures opening tendencies, which are highly predictive in SND where the first five seconds of a round often determine the entire strategic framework.

### 9.2 Team Formation Features

Individual features tell us what each player does. Formation features tell us how the four players move as a unit:

**Team centroid** — the geometric mean of all four players' positions. The centroid's trajectory over time shows where the team's "center of gravity" is and how it shifts during a round.

**Spatial spread** — the standard deviation of player positions around the centroid. Low spread means the team is stacked together; high spread means they're spread across the map.

**Convex hull area** — the area of the smallest polygon enclosing all four players. This quantifies "map control" — a larger hull means the team controls more territory.

**Pairwise distances** — the six distances between all pairs of players. These reveal buddy-pair structures (two players close together for trading) and isolated players (potential flankers).

**Buddy pair ratio** — the minimum pairwise distance divided by the maximum. A ratio near 1.0 means all players are evenly spaced; a ratio near 0.0 means some players are very close while others are far away, indicating a split setup.

**Centroid velocity** — how fast the team's center of gravity is moving. High centroid velocity = aggressive push. Low centroid velocity = methodical default.

These features are computed at each timestamp, producing a time series of formation states per round. Importantly, they are **order-invariant** — they don't depend on which player is labeled as Player 1, 2, 3, or 4. This is critical because the tracker's ID assignment is arbitrary and can vary across rounds.

### 9.3 Engagement Features

From the kill feed, we derive features that capture how the team fights:

**First blood timing** — how many seconds into the round before the first kill occurs. Early first bloods indicate aggressive play; late first bloods indicate methodical, information-gathering setups.

**Trade timing** — the gap in seconds between a teammate's death and the revenge kill. Fast trades (under 2 seconds) indicate disciplined team play; slow or failed trades indicate isolation.

**Trade success rate** — what percentage of teammate deaths are traded. This is one of the most diagnostic features in competitive CoD — teams with high trade rates consistently outperform teams with similar gunskill but poor trading discipline.

**Kill position heatmap** — which grid cell the killer was in when each kill occurred. This reveals power positions — spots on the map where a team consistently wins gunfights.

**Weapon distribution** — the percentage of kills by weapon category (SMG, AR, sniper, shotgun). Heavy SMG usage correlates with aggressive, close-range play; heavy AR/sniper usage indicates passive, lane-holding setups.

---

## 10. Discovering Strategies with Unsupervised Learning

The central insight of Sightline's ML layer is that we don't define strategies manually — we discover them from data. Given enough rounds of an opponent's gameplay, patterns emerge: routes they prefer, formations they default to, adjustments they make. Three unsupervised techniques operate at different granularities.

### 10.1 Individual Route Clustering (DTW + DBSCAN)

Each player's round can be represented as a grid-cell transition sequence. Two rounds where a player takes the same route will produce similar sequences, but not identical ones — maybe they lingered in one cell slightly longer, or took a minor detour. We need a distance metric that handles this variability.

**Dynamic Time Warping (DTW)** is that metric. Unlike Euclidean distance, which requires sequences to be the same length and aligned in time, DTW finds the optimal alignment between two sequences of different lengths by warping the time axis. Two routes that follow the same path but at slightly different speeds will have a small DTW distance. Two routes that go to completely different parts of the map will have a large DTW distance.

With DTW as the distance metric, we cluster routes using **DBSCAN**. Unlike K-Means, DBSCAN doesn't require specifying the number of clusters in advance — it discovers them from the density of the data. Points in dense regions form clusters; isolated points are labeled as outliers (novel or unusual routes). The output for each player on each map might look like:

- **Cluster A (38% of rounds):** Push mid tunnel, hold passive from window
- **Cluster B (27% of rounds):** Fast B-site rush, play for first blood
- **Cluster C (22% of rounds):** Anchor A-site, rarely reposition
- **Noise (13% of rounds):** Unusual routes, one-off adaptations

This gives us a concrete vocabulary for describing what a player *tends to do* — not just in aggregate statistics, but in terms of specific, named routes that we can reference in strategy discussions.

### 10.2 Team Formation Clustering (GMM)

Individual routes don't capture team coordination. A "2-2 split" setup, where two players pressure one bombsite while two pressure another, is a *team-level* pattern that only emerges when you look at all four players simultaneously.

We represent each round as a fixed-length feature vector of team formation features (centroid, spread, hull area, pairwise distances, etc.) aggregated or sampled at key timestamps. These vectors are then clustered using a **Gaussian Mixture Model (GMM)**.

GMMs offer a crucial advantage over hard-clustering methods: **soft assignments**. A round can be 70% "2-2 split" and 30% "slow default" — reflecting a team that started with a split setup but converged to a different play mid-round. This captures the fluid, adaptive nature of competitive strategy better than forcing each round into a single category.

The output is a set of discovered formations per team per map:

- **"Fast B Execute" (30%):** All four collapse to B within 15 seconds
- **"2-2 Split" (45%):** Two players pressure A, two pressure B, converge on info
- **"Slow Default" (25%):** Spread across mid, gather info, execute at 0:30+

### 10.3 Playstyle Fingerprints (UMAP)

The previous two techniques operate per-player and per-team on individual maps. The playstyle fingerprint aggregates across all maps and matches against a given opponent to produce a single, holistic description of how that team plays.

The fingerprint is a vector of aggregate cluster distributions — what percentage of rounds were classified as aggressive vs. passive, fast vs. slow, split vs. stacked — combined with engagement statistics (trade rate, first blood rate, adaptation rate after losses). This vector lives in a high-dimensional space that humans can't directly interpret.

**UMAP** (Uniform Manifold Approximation and Projection) projects these vectors down to two dimensions while preserving the topological structure of the data. The result is a scatter plot where each point represents a team, and teams that play similarly are placed near each other:

```
         Aggressive
            ↑
   OpTic •  |  • LAT
            |
  ← Passive ----+---- Active →
            |
    G2 •    |     • Faze
            ↓
        Methodical
```

This visualization serves two practical purposes. First, it gives managers an intuitive feel for how the competitive landscape is structured — which teams are threats because they play a style that historically gives Faze trouble. Second, teams that cluster together in playstyle space can often be countered with similar strategies, allowing preparation on one opponent to transfer to another.

---

## 11. Sequence Models for Behavioral Prediction

Clustering reveals what teams tend to do. Sequence models reveal *when* and *how* — the temporal structure of strategy that static features miss.

### 11.1 LSTM for Movement Prediction

A two-layer LSTM (Long Short-Term Memory network) takes a player's `(x, y, t)` position sequence as input and learns to predict what comes next. After training on an opponent's VODs, the model can take the first five seconds of a player's movement in a new round and predict with meaningful accuracy where they're heading.

This directly enables pre-round strategy: "Based on MAMBA's first move toward B2, there's a 78% probability he's running his fast B-rush route. Position accordingly."

The LSTM's predictions aren't deterministic — they produce probability distributions over grid cells — but they capture the conditional structure of movement. If a player goes to C2 first, they almost always go to C3 next (moving through a lane). If a player stops at B3 for more than three seconds, they're likely settling into a hold position rather than pushing through.

### 11.2 Transformer for Round Event Sequences

Individual movements are low-level. At a higher level of abstraction, each round is a sequence of *events*: round starts, players move to positions, first blood occurs, a trade happens, a bomb is planted, a clutch is attempted. The **Transformer** architecture — the same attention-based mechanism that powers modern language models — excels at learning patterns in event sequences.

Each event is encoded as an embedding combining the event type, the player involved, their grid position, and the timestamp. The Transformer processes the sequence with self-attention, learning which events are predictive of the round outcome.

The trained model can then identify the **tipping point** in a losing round — the specific moment where the sequence diverged from patterns that typically lead to wins. This is the foundation of the loss analysis engine: "The sequence diverged from winning patterns at 0:45 when SIMP pushed C3 alone without a trade partner in B3. In 83% of winning rounds with this formation, the B3 player pushes simultaneously."

---

## 12. The Strategy Engine

The ML models produce patterns. The strategy engine turns patterns into actionable recommendations.

### 12.1 Playstyle Profiler

The profiler aggregates all ML outputs for a given opponent into a structured report:

- **Pace**: Aggressive (60% fast executes) vs. methodical (70% slow defaults)
- **Map control priority**: Which grid cells they contest first, per map
- **Trading discipline**: Average trade timing and success rate
- **Adaptation patterns**: How their cluster distribution shifts after consecutive losses
- **Post-plant setups**: Positions they gravitate to after planting the bomb
- **Retake tendencies**: Routes used when retaking a bombsite

### 12.2 SND Strategy Suggester

For each upcoming SND map, the suggester combines the opponent's playstyle profile with Faze's own historical performance to generate pre-round strategy:

- Which formation has the highest win probability against this opponent's tendencies
- Where to position based on the opponent's likely opening plays
- Which bombsite to prioritize attacking based on the opponent's defensive weaknesses

### 12.3 Loss Analyzer

For rounds that Faze lost, the analyzer provides:

- The **tipping point moment** — identified by the Transformer as the event where the sequence diverged from winning patterns
- A **comparison** between Faze's actual positions and the positions typical of winning rounds
- A **concrete suggestion** — a specific adjustment (different position, different timing, different trading structure) that historically correlates with better outcomes

### 12.4 LLM Narration

Raw ML output — cluster IDs, probability distributions, embedding distances — is not something a coach wants to read. The narration layer takes structured data and feeds it to a large language model (GPT-4o or Gemini 2.0 Flash) to produce human-readable strategy text.

The prompt is carefully constructed to include all relevant context:

```
Given: Faze (attack) on Karachi SND Round 5.
Opponent (G2 Minnesota) defensive formation: 82% match to "2-1-2 passive"
cluster. MAMBA typically holds B3 (mid-window). KREMP anchors E4 (B-site).
Faze lost this round via elimination at 1:23.
Faze positions at tipping point (0:45): SIMP alone in C3, nearest teammate
in A2 (distance: 0.42).
Winning pattern: B3 player pushes with C3 player within 2 seconds.

Generate: A tactical breakdown of what went wrong and a specific
alternative approach for this round.
```

The LLM synthesizes this into readable, actionable text — but critically, the intelligence comes from the ML models, not from the LLM. The LLM is a translation layer, not an analysis layer. This distinction is important because it means the strategic insights are grounded in actual data patterns from the team's VODs, not in the LLM's general-purpose training data (which has no direct knowledge of CDL meta-strategy or specific team tendencies).

---

## 13. Data Architecture

All extracted and derived data flows into a relational database. The schema mirrors the hierarchical structure of competitive CoD:

```
VOD → Match → Map Game → Round → {Positions, Kill Events, Stats}
                                → {Strategy Suggestions}
Match → Team Playstyle Profile
```

A **VOD** contains one or more **Matches** (a VOD file might cover an entire day of competition). Each **Match** between Faze and an opponent consists of multiple **Map Games** (best-of-five series). Each Map Game in SND contains multiple **Rounds**. Each Round contains time-series data: player positions (sampled at 4–5 fps), kill events, and stat snapshots from the roster tables.

Strategy suggestions and playstyle profiles are computed artifacts stored alongside the raw data, linked to their source rounds and matches. This allows coaches to trace any recommendation back to the specific game footage it was derived from.

---

## 14. The Platform

Sightline's web platform is built with **Next.js** (React-based, server-rendered) backed by a **FastAPI** Python API that serves data from the database. The UI uses **shadcn/ui** for interface components and **Tremor** for analytics visualizations, styled in a dark-first esports aesthetic.

The platform supports five primary workflows:

**VOD Processing**: Drag-and-drop VOD upload with real-time processing status. The pipeline runs asynchronously, and managers can see progress per stage (gameplay detection → extraction → analysis → strategy generation).

**Opponent Scouting**: Select an opponent and receive a comprehensive scouting report — playstyle fingerprint, per-map formation breakdown, individual player route tendencies, and the UMAP position showing where this opponent sits relative to the rest of the league.

**Match Review**: Browse processed matches with map-by-map breakdowns, round timelines, and stat progressions. Each round links to a detailed view.

**Round Replay**: An interactive minimap canvas replays player positions as colored dots in real time, synchronized with a kill feed sidebar and strategy overlay. Managers can scrub to any moment in a round and see exactly where every player was.

**Strategy Builder**: AI-generated suggestions are presented alongside an interactive map where managers can draw custom positions, annotate plays, and save them to a team playbook. The playbook tracks which strategies are used in matches and their win rates over time, creating a feedback loop between preparation and performance.

---

## 15. Putting It All Together

The complete pipeline, from raw YouTube playlist to actionable strategy, follows this flow:

1. **Acquire**: `sightline download --playlist "URL" --filter "faze" --limit 20` fetches VODs from YouTube, filtered by team name, downscaled to 720p.

2. **Detect**: The gameplay detector scans each VOD at 0.5 fps, finding the windows where actual competitive play is happening.

3. **Sample**: The adaptive sampler extracts frames within gameplay windows at 4–5 fps, with burst sampling around detected engagements.

4. **Extract**: Four parallel extractors process each frame — minimap positions via YOLO + color clustering, roster stats via OCR, scoreboard data via OCR, and kill events via OCR + template matching.

5. **Segment**: Round boundaries are detected from score changes and screen transitions, producing annotated Round objects.

6. **Engineer**: Raw positions and events are transformed into movement, formation, and engagement features.

7. **Discover**: DTW + DBSCAN discovers individual routes. GMM discovers team formations. UMAP produces playstyle fingerprints. LSTM and Transformer models learn temporal patterns.

8. **Strategize**: The strategy engine synthesizes ML outputs into scouting reports, pre-round suggestions, and loss analyses. The LLM narration layer translates these into readable text.

9. **Deliver**: Everything is accessible through the Sightline web platform, where managers scout opponents, replay rounds, and build playbooks.

For a corpus of 20 VODs, the full pipeline — from raw video to strategy-ready database — processes in approximately 30–45 minutes on commodity hardware with a modest GPU. Subsequent VODs process incrementally, and the ML models improve with each batch as the training dataset grows.

---

## 16. What This Enables

With Sightline running, a coaching staff's match preparation workflow transforms from hours of manual VOD review to:

1. Open the opponent's scouting page
2. Review their playstyle fingerprint and formation tendencies per map
3. Read the AI-generated counter-strategy for each SND map in the series
4. Replay key rounds from previous matchups against this opponent
5. Adjust the suggested strategies in the strategy builder and save to the playbook

The total preparation time drops from 4–6 hours to 30–45 minutes. More importantly, the quality of preparation improves — the system identifies patterns across 20 hours of footage that no human could track, surfaces specific positional recommendations backed by statistical evidence, and provides a shared strategic vocabulary (grid zones, cluster names, formation labels) that the entire coaching staff and roster can reference.

Competitive Call of Duty, like all esports, is converging toward a future where data-driven preparation is not a competitive advantage but a competitive requirement. Sightline is a blueprint for what that future looks like.
