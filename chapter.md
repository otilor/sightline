# Chapter X: Sightline — Building an AI-Powered Esports Analytics Platform

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

## 3. The Extraction Pipeline

### 3.1 Unsupervised Minimap Tracking
The minimap is a small, flat, top-down representation of the game map. On it, colored dots represent players. The problem is deceptively simple to state — find the dots and track them.

We train a **YOLOv8-nano** model on annotated minimap crops. The training set is bootstrapped from approximately 200–300 manually-labeled screenshots. YOLOv8-nano is chosen for its speed — inference on a single minimap crop takes under 1 millisecond via ONNX Runtime, allowing us to process thousands of frames without bottlenecking. 

To determine team alliance, we treat team identification as an **unsupervised clustering problem**. After detecting all player dots in a frame, we extract the dominant pixel color of each dot and convert it to the **CIELAB color space**. Unlike RGB or HSV, CIELAB is perceptually uniform. **K-Means with k=2** on the Lab color vectors naturally splits the room into the two teams without manually hardcoding specific hex values.

### 3.2 Transient Event Capture: The Killfeed
The kill feed is the most challenging extraction target because it's **transient and dense**. Missing a single entry means losing a kill event.

Each kill feed entry contains: a killer name (text), a weapon icon (graphical), and a victim name (text). We use **EasyOCR** for the names. For the weapon icons, which are graphical assets, we utilize **template matching** with normalized cross-correlation. Since CDL broadcasts use standardized weapon icon assets, this matching is highly reliable in perfect conditions. *However, this area presented the greatest challenges in the Sightline platform, heavily affected by broadcast compression and scaling artifacts.* 

---

## 4. Modeling Strategy and Behavior

Raw position data and kill events are not directly useful for machine learning. They need to be transformed into features that capture the *behavior* behind the data.

### 4.1 Trajectory Sequence Modeling (LSTM)
A two-layer LSTM (`num_layers=2` with 0.2 dropout) takes a player's `(x, y, t)` position sequence as a sliding window input (20 timesteps) and learns to predict what comes next. By isolating the hidden state at the final timestep and mapping it through a linear layer back into 2D coordinate space, the LSTM acts as a regression mechanism predicting the next move.

This directly enables pre-round strategy: "Based on MAMBA's first move toward B2, there's a 78% probability he's running his fast B-rush route. Position accordingly." 

### 4.2 Route Clustering (DTW + DBSCAN)
Each player's movement patterns over a round are grouped using **Dynamic Time Warping (DTW)**. Unlike Euclidean distance, DTW finds the optimal alignment between two sequences of different lengths by warping the time axis. Two routes that follow the same path but at slightly different speeds will have a small DTW distance. We then cluster these similarities using **DBSCAN**.

The output gives us a concrete vocabulary:
- **Cluster A (38% of rounds):** Push mid tunnel, hold passive from window
- **Cluster B (27% of rounds):** Fast B-site rush, play for first blood

### 4.3 Team Formation Clustering (GMM)
Individual routes don't capture team coordination. A "2-2 split" setup is a *team-level* pattern that only emerges when you look at all four players simultaneously. We represent each round as a fixed-length feature vector of team formation features (centroid, spread, hull area, pairwise distances, etc.) aggregated at key timestamps. 

These vectors are clustered using a **Gaussian Mixture Model (GMM)**, creating a soft-assignment playstyle fingerprint for a given team on a specific map. The pipeline aggregates all of this to understand global playstyles using **UMAP** down-projection.

---

## 5. Deployment Results

Following iterative development, the Sightline system was deployed on a recent VOD sample spanning multiple CDL maps. The pipeline successfully ran from end-to-end, converting raw mp4 footage into a fully normalized SQLite database structure.

**Raw Extraction Stats:**
- **543** unique rounds detected and segmented
- **3,923** individual kill events parsed via OCR and template matching
- **3,322** player stat snapshots generated from real-time roster extraction
- **20+** unique players accurately tracked and identified across multiple series 

**Derived Output:**
- **Automated Data Feed:** The dashboard actively consumes these stats, generating completely dynamic K/D leaderboards and mode breakdowns. For instance, players like "SIMP" and "ABEZY" are automatically ranked based purely on the CV-extracted kill events.
- **AI Strategy Generation:** Using the statistical tendencies derived from the data, the strategist engine generated 3 distinct AI counter-strategies (e.g., executing a fast 4-man push on Karachi, or predicting a 2-1-2 split defense on Terminal) with 85%+ confidence ratings. 
- **Minimap Reconstruction:** The web replay tool successfully replayed the X/Y coordinates extracted by the YOLO tracker, plotting player movement paths perfectly atop a 5x5 tactical grid.

---

## 6. The Web Platform

Sightline's web platform is built with **Next.js** (React-based, server-rendered) backed by a **FastAPI** Python API that serves data from the SQLite database. The UI uses modern glassmorphism styling in a dark-first esports aesthetic.

The platform supports five primary workflows:
1. **Dashboard Overview**: Macro-level VOD processing states and recent activity feeds.
2. **Opponent Scouting**: Playstyle fingerprints, per-map formation breakdown, and individual player route tendencies.
3. **Strategy Engine & Playbook**: AI-generated counter-strategies with high-confidence action items based on clustered behavior matching.
4. **Interactive Minimap Replays**: A canvas that replays player positions as colored dots in real time, synchronized with a kill feed sidebar and timeline scrubber.
5. **Team Stats**: Dynamically calculated global K/D averages using aggregated kill events decoupled from broadcast inaccuracies.

---

## 7. The Data vs. CV Trade-off: Looking Forward

The most significant learning from the Sightline project was the brittleness of pure Computer Vision for dense UI applications. While minimap tracking of simple colored dots is highly robust after CIELAB clustering and temporal smoothing, extracting tiny, fast-moving killfeed weapon icons is intensely prone to compression artifacts and false positives. 

The system's modularity allows for a hybrid pivot: moving forward, platforms like Sightline can swap the brittle OCR components out for deterministic web scraping (pulling play-by-play logs from official APIs or community stat sites) while retaining the Computer Vision exclusively for the spatial, X/Y geometry data of the minimap. This hybrid approach guarantees 100% statistical accuracy while preserving the spatial data that makes positional playbook strategy possible.
