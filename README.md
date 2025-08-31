# 🚗 Driver Drowsiness Detection System 💡💻🧠

Monitor driver fatigue in real-time using **Python, OpenCV, Dlib, and Playsound**.  
This system detects **sleeping, drowsy, or yawning** states and triggers an alarm if needed.

---

## 🧑‍💻 Project Overview
- Tracks facial landmarks using Dlib’s **68-point predictor**.  
- Calculates **Eye Aspect Ratio (EAR)** to determine eye closure.  
- Calculates **Mouth Opening Ratio (MOR)** to detect yawning.  
- Triggers an alarm using `playsound` when drowsiness or yawning is detected.

---

## 🔍 How It Works
1. Capture video from webcam via OpenCV.  
2. Detect faces and facial landmarks using Dlib.  
3. Compute **EAR** and **MOR** for each frame.  
4. Count consecutive frames to determine state:  
   - **Active 🙂** → Eyes open  
   - **Drowsy 😐** → Eyes partially closed  
   - **Sleeping 😴** → Eyes fully closed → Alarm  
   - **Yawning 😮** → Mouth open beyond threshold → Alarm

---

## 📊 Working Principle

**Eye Aspect Ratio (EAR)**:  

$$
EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}
$$

- \(p_1\) to \(p_6\) are eye landmark points.  
- EAR decreases when eyes close.  
- Thresholds determine Drowsy/Sleeping state.

**Mouth Opening Ratio (MOR)**:

$$
MOR = \frac{||Z_2 - Z_1||}{3 \times ||W_2 - W_1||}
$$

- \(Z_1, Z_2\) → Top and bottom inner lips  
- \(W_1, W_2\) → Left and right mouth corners  
- High MOR indicates yawning.

---

## 🖼️ Output

| State      | Color  | Icon | Alarm |
|-----------|--------|------|-------|
| Active    | Green  | 🙂    | No    |
| Drowsy    | Orange | 😐    | Optional |
| Sleeping  | Red    | 😴    | Yes   |
| Yawning   | Blue   | 😮    | Yes   |

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install opencv-python dlib imutils numpy playsound
