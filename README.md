# 🚗 Driver Drowsiness Detection System 💡💻🧠

Excited to share my latest project – **Driver Drowsiness Detection** – built using **Python, OpenCV, and Dlib**!  
This system monitors a driver’s facial landmarks to detect signs of drowsiness and fatigue, aiming to enhance road safety.  
If the system detects **Sleeping**, it triggers an **alarm sound** to alert the driver.

---

## 🧑‍💻 Project Overview
The system uses Dlib’s 68 facial landmark detector to track key facial points—especially around the eyes.  
By calculating the **Eye Aspect Ratio (EAR)**, it determines the driver’s state:  
- ✅ **Active** (eyes open)  
- ⚠ **Drowsy** (partially closed)  
- ❌ **Sleeping** (fully closed)

---

## 🔍 How It Works
1. Capture real-time video using OpenCV.  
2. Detect facial landmarks with Dlib’s pre-trained model.  
3. Calculate the **EAR** (eye openness) to monitor eye movement.  
4. Identify and alert if the driver is active, drowsy, or sleeping.

---

## 📊 Working Principle
The **Eye Aspect Ratio (EAR)** is calculated as:

$$
EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}
$$

Where:  
- \(p_1\) to \(p_6\) are the eye landmark points  
- EAR decreases when eyes are closed  
- If eyes remain closed for a threshold duration → Driver is **Drowsy/Sleeping**

---

## 🖼️ Output
- **Active 🙂** → Green text  
- **Drowsy 😐** → Red text  
- **Sleeping 😴** → Blue text + Alarm sound  

---

## ⚙️ Requirements
Install dependencies:

```bash
pip install opencv-python dlib imutils numpy playsound
