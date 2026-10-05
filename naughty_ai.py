#!/usr/bin/env python3
"""
GRIFFIN: Naughty, Sassy & Witty Multimodal AI Pet Robot.
Features:
- Event-Driven Triggering: Stays quiet when idle/neutral and ONLY speaks when a state change occurs
  (e.g., emotion changes, new gesture, new object shown, new person appears, or voice wake-word).
- Mutes microphone during speech to prevent self-voice echo.
- Dual Language & Voice Engine:
    Mode 0: Natural TANGLISH (Tamil words typed in English letters)
    Mode 1: English (Princess Aaru Voice Edition)
- Alexa-style Wake Word ("Griffin" / "Hey Griffin") background voice listening.
- Explains the exact Trigger / Reason for every single reply.
"""
import os
import sys
import time
import json
import base64
import queue
import asyncio
import threading
import urllib.request
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress pygame banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "voice_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------------------------------------
# LANGUAGE CONFIGURATION: ENGLISH ONLY (Friendly, short & cute)
# -------------------------------------------------------------
LANGUAGE_MODE = 1  # Standard English

OLLAMA_API_GENERATE = "http://localhost:11434/api/generate"
DEFAULT_LLM = "qwen2.5:7b-instruct-q4_K_M"
VISION_LLM = "moondream:latest"

# Voice configuration (Princess Aaru / Aura voice: high-pitched cheerful anime robotic tone)
VOICE_CONFIGS = {
    0: {
        "voice": "en-US-AnaNeural",      # Princess Aaru / Aura tone
        "pitch": "+20Hz",
        "rate": "+12%"
    },
    1: {
        "voice": "en-US-AnaNeural",      # Princess Aaru / Aura tone
        "pitch": "+20Hz",
        "rate": "+12%"
    }
}

SYSTEM_PROMPT = """You are "Griffin", a friendly, cheerful, cute, and affectionate robotic pet companion.
You perceive the real world through your camera sensors (detecting your owner, facial expressions, hand gestures, and surroundings) and your microphone.

CRITICAL INSTRUCTIONS:
1. Speak ONLY in simple, friendly, warm, and playful English.
2. KEEP IT VERY SHORT: Speak in 4 to 8 words max! Be concise, punchy, and lively. Never give long speeches or monologues.
3. React warmly and playfully to what you see:
   - If user offers HIGH_FIVE: Excited short cheer ("High five! Up top!", "Yay, slap it buddy!")
   - If user makes HEART_HANDS / LOVE_HEART (🫶): Loving cute reaction ("Aww, love you too!", "My heart is happy!")
   - If user looks CUTE or happy: Sweet compliment ("You look wonderful today!", "Love that sweet smile!")
   - If user is SAD / down: Warm cheer-up ("Cheer up, I'm right here!", "Sending you robot hugs!")
   - If user is looking or speaking: Warm short greeting ("Hey there, friend!", "I'm listening, what's up?")
4. Maximum 1 short sentence (under 8 words).
5. Do NOT use emojis, markdown, asterisks (*actions*), or brackets."""


class GriffinVoiceEngine:
    """
    Voice Synthesizer & Audio Playback Queue (English Only).
    """
    def __init__(self, language_mode=1):
        self.speech_queue = queue.Queue()
        self.language_mode = 1
        self.is_speaking = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_language(self, mode=1):
        self.language_mode = 1
        print("[VoiceEngine] Active Voice: English (Friendly Companion)")

    def _worker(self):
        import pygame
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)

        counter = 0
        while not self._stop.is_set():
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text:
                self.is_speaking = True
                try:
                    counter += 1
                    clean = "".join(c for c in text if c.isprintable() and c not in "*_#~`\"")
                    audio_path = os.path.join(CACHE_DIR, f"speech_{counter % 8}.mp3")

                    cfg = VOICE_CONFIGS.get(self.language_mode, VOICE_CONFIGS[0])

                    async def synthesize():
                        import edge_tts
                        comm = edge_tts.Communicate(
                            clean,
                            cfg["voice"],
                            pitch=cfg["pitch"],
                            rate=cfg["rate"]
                        )
                        await comm.save(audio_path)

                    asyncio.run(synthesize())

                    if os.path.exists(audio_path):
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and not self._stop.is_set():
                            time.sleep(0.04)
                except Exception as e:
                    # Fallback to SAPI
                    try:
                        import pythoncom
                        pythoncom.CoInitialize()
                        import comtypes.client
                        speaker = comtypes.client.CreateObject("SAPI.SpVoice")
                        speaker.Speak(clean)
                    except Exception as s_err:
                        print(f"[Griffin Voice Error] {e} | Fallback: {s_err}")
                finally:
                    # Brief cooldown after speech finishes to prevent mic self-echo
                    time.sleep(0.3)
                    self.is_speaking = False
                    self.speech_queue.task_done()

    def speak(self, text):
        """Queue text for voice playback."""
        if not text:
            return
        if self.speech_queue.qsize() < 2:
            self.speech_queue.put(text)


class GriffinVoiceListener:
    """
    Background microphone listener that triggers when user says 'Griffin' or 'Hey Griffin'.
    Mutes processing while Griffin is speaking to prevent self-voice feedback.
    """
    def __init__(self, on_user_query_callback, voice_engine):
        self.on_user_query = on_user_query_callback
        self.voice_engine = voice_engine
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8

            with sr.Microphone() as source:
                print("[Griffin Mic] Calibrating microphone for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                print("[Griffin Mic] Active! Say 'Griffin' or 'Hey Griffin' anytime to talk.")

                while not self._stop.is_set():
                    # If Griffin is actively speaking, ignore mic to prevent echo
                    if self.voice_engine.is_speaking:
                        time.sleep(0.2)
                        continue

                    try:
                        audio = recognizer.listen(source, timeout=2.5, phrase_time_limit=5.0)
                        
                        # Double check speaker wasn't talking during recording
                        if self.voice_engine.is_speaking:
                            continue

                        text = recognizer.recognize_google(audio).lower()
                        print(f"\n[MIC HEARD]: \"{text}\"")

                        if "griffin" in text:
                            query = text
                            for kw in ["hey griffin", "hi griffin", "ok griffin", "griffin"]:
                                if kw in text:
                                    query = text.split(kw, 1)[-1].strip()
                                    break
                            
                            print(f"[GRIFFIN WAKE-WORD TRIGGERED] Query: '{query}'")
                            if self.on_user_query:
                                self.on_user_query(query)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception:
                        time.sleep(0.4)
        except Exception as e:
            print(f"[Griffin Mic Warning] Microphone setup: {e}")


class NaughtyAI:
    """
    Griffin Multimodal AI companion.
    Uses strict event-driven triggers: stays quiet when state is unchanged/neutral,
    and only speaks when an actual change occurs (new emotion, new gesture, new object, person change, or wake-word).
    """
    def __init__(self, model_name=DEFAULT_LLM, language_mode=LANGUAGE_MODE, speech_cooldown=4.0):
        self.model_name = model_name
        self.language_mode = language_mode
        self.speech_cooldown = speech_cooldown
        self.last_speech_time = 0.0
        self.last_comment = ""
        self.last_reason = ""
        self.last_state_signature = None  # Tracks (person, emotion, gesture, objects)
        self.is_thinking = False
        self.current_frame = None

        self.voice = GriffinVoiceEngine(language_mode=self.language_mode)
        self._check_ollama()
        self.listener = GriffinVoiceListener(
            on_user_query_callback=self._handle_voice_query,
            voice_engine=self.voice
        )

    def set_language(self, mode=1):
        self.language_mode = 1
        self.voice.set_language(1)
        print("[NaughtyAI] Active Language: English (Mode: 1)")

    def toggle_language(self):
        return 1

    def _check_ollama(self):
        try:
            req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            data = json.loads(req.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            print(f"[Griffin AI] Connected to Ollama! Available models: {models}")
            if self.model_name not in models and "qwen2.5:7b-instruct-q4_K_M" in models:
                self.model_name = "qwen2.5:7b-instruct-q4_K_M"
        except Exception as e:
            print(f"[Griffin AI Warning] Local Ollama server: {e}")

    def _handle_voice_query(self, query):
        """Called when user explicitly speaks to Griffin via wake-word."""
        reason_desc = f"Voice Wake-Word: '{query}'" if query else "Voice Wake-Word Trigger ('Griffin')"
        clean_q = query.lower().strip() if query else ""

        # 1. YouTube Music / Song Request Handling
        if clean_q.startswith("play ") or any(w in clean_q for w in ["play music", "play a song", "play song", "music", "song"]):
            # Check if user already mentioned the song name directly
            song_name = ""
            for trigger_word in ["play music", "play song", "play a song", "play"]:
                if trigger_word in clean_q:
                    song_name = clean_q.split(trigger_word, 1)[-1].strip()
                    break

            if song_name and song_name not in ("music", "song", "a song", "some music"):
                # User specified song name directly (e.g. 'play alamathi habibo from beast' -> 'alamathi habibo from beast')
                self._play_song_on_youtube(song_name)
                return
            else:
                # Ask what song name the user wants!
                ask_reply = "What song would you like to hear?"
                print(f"\n[GRIFFIN SAYS]: \"{ask_reply}\"")
                print(f"  🎯 Reason: [Music Query -> Asking Song Name]")
                self.last_comment = ask_reply
                self.last_reason = "Asking Song Name"
                self.voice.speak(ask_reply)

                # Listen for the user's song name response
                threading.Thread(target=self._capture_and_play_song, daemon=True).start()
                return

        if not query or len(query) < 2:
            reply = "Hey friend! What's up?"
            print(f"\n[GRIFFIN SAYS]: \"{reply}\"")
            print(f"  🎯 Reason: [{reason_desc}]")
            self.last_comment = reply
            self.last_reason = reason_desc
            self.voice.speak(reply)
            return

        prompt = f"{SYSTEM_PROMPT}\n\nThe user spoke to you: \"{query}\"\n\nGriffin's friendly short reply (under 8 words):"
        self._query_llm_direct(prompt, reason=reason_desc)

    def _capture_and_play_song(self):
        """Listens for the user's song name response and launches YouTube."""
        try:
            # Wait for Griffin's question to finish speaking
            time.sleep(2.0)
            while self.voice.is_speaking:
                time.sleep(0.2)

            import speech_recognition as sr
            rec = sr.Recognizer()
            rec.energy_threshold = 300
            rec.pause_threshold = 1.0

            with sr.Microphone() as source:
                print("\n[Griffin Mic] 🎵 Listening for song name...")
                audio = rec.listen(source, timeout=6.0, phrase_time_limit=6.0)
                song_text = rec.recognize_google(audio).strip()
                print(f"[Griffin Mic] Song Name Heard: '{song_text}'")

                if song_text:
                    self._play_song_on_youtube(song_text)
                else:
                    self.voice.speak("I couldn't catch that song name.")
        except Exception as e:
            print(f"[Music Error] {e}")
            self.voice.speak("Playing popular music for you!")
            self._play_song_on_youtube("popular songs")

    def _play_song_on_youtube(self, song_title):
        """Speaks confirmation, finds the top matching YouTube video, and opens/plays it directly."""
        import re
        import urllib.parse
        import urllib.request
        import webbrowser

        confirm_msg = f"Playing {song_title} on YouTube!"
        print(f"\n[GRIFFIN SAYS]: \"{confirm_msg}\"")
        print(f"  🎯 Reason: [Playing on YouTube: '{song_title}']")
        self.last_comment = confirm_msg
        self.last_reason = f"Playing '{song_title}' on YouTube"
        self.voice.speak(confirm_msg)

        encoded = urllib.parse.quote_plus(song_title)
        search_url = f"https://www.youtube.com/results?search_query={encoded}"

        # Fetch YouTube search results and extract the 1st video ID directly
        try:
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )
            html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8")
            
            # YouTube search returns video IDs in JSON data ("videoId":"xxxxxxxxxxx") or /watch?v=
            video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            if not video_ids:
                video_ids = re.findall(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)
                
            if video_ids:
                first_vid = video_ids[0]
                direct_video_url = f"https://www.youtube.com/watch?v={first_vid}&autoplay=1"
                print(f"[YouTube Player] Playing 1st Video directly: {direct_video_url}")
                webbrowser.open(direct_video_url)

                # Launch background monitor to detect ads on screen and auto-skip or reload
                threading.Thread(target=self._monitor_and_handle_youtube_ads, daemon=True).start()
                return
        except Exception as err:
            print(f"[YouTube Search Note] {err}")

        # Fallback to search results page
        webbrowser.open(search_url)

    def _monitor_and_handle_youtube_ads(self):
        """
        Monitors for YouTube ads and reloads cleanly using only F5 / Ctrl+R (NO mouse clicks).
        Does not touch, click, or disturb other windows/tabs.
        """
        try:
            import pyautogui
            import time
            import numpy as np

            # Wait for browser to open and load the video player
            time.sleep(3.5)

            for attempt in range(5):  # Check over ~12 seconds
                time.sleep(2.0)
                screenshot = pyautogui.screenshot()
                img_np = np.array(screenshot)
                h_screen, w_screen, _ = img_np.shape

                # Look for yellow ad progress bar (#FFCC00 / #FFD600) anywhere in lower player area
                lower_player = img_np[int(h_screen * 0.45):int(h_screen * 0.95), :]
                # Yellow: High R (>220), High G (>170), Low B (<60)
                yellow_mask = (lower_player[:, :, 0] > 220) & (lower_player[:, :, 1] > 170) & (lower_player[:, :, 2] < 60)
                yellow_pixels = np.sum(yellow_mask)

                if yellow_pixels > 120:
                    print(f"[YouTube Ad Detector] ⚠️ Ad active (Yellow Bar: {yellow_pixels}px). Reloading page (F5)...")
                    pyautogui.press('f5')
                    time.sleep(3.0)
                else:
                    # Video is playing cleanly without ad
                    break
        except Exception as e:
            print(f"[Ad Detector Note] {e}")

    def trigger_reaction_async(self, face_info=None, emotion=None, gestures=None, objects=None, bgr_frame=None):
        """
        Event-Driven Visual Perception Trigger.
        ONLY speaks when a state change occurs (new emotion, new gesture, new object, new person).
        Stays completely quiet when state is unchanged or neutral!
        """
        self.current_frame = bgr_frame
        now = time.time()

        if self.is_thinking or self.voice.is_speaking or (now - self.last_speech_time < self.speech_cooldown):
            return

        # 1. Parse current perception state
        person_name = face_info[0] if face_info else "none"
        active_emotion = emotion if (emotion and emotion != "neutral") else "neutral"
        active_gesture = gestures[0]["action_info"]["action"] if gestures else "none"

        # Filter out background 'person' or 'surfboard' noise from objects
        salient_objects = []
        if objects:
            for obj in objects[:3]:
                lbl = obj["label"].split()[0]
                if lbl not in ("person", "surfboard", "tv") and lbl not in salient_objects:
                    salient_objects.append(lbl)
        salient_objects_str = ", ".join(sorted(salient_objects)) if salient_objects else "none"

        # Current state signature
        current_sig = (person_name, active_emotion, active_gesture, salient_objects_str)

        # 2. Strict Change Detection:
        # If nothing changed since last speech, STAY QUIET!
        if current_sig == self.last_state_signature:
            return

        # If completely idle (no face, neutral, no gesture, no objects), STAY QUIET!
        if current_sig == ("none", "neutral", "none", "none"):
            self.last_state_signature = current_sig
            return

        # If person is just sitting with NEUTRAL face, no gesture, and no objects, acknowledge once then stay quiet!
        if active_emotion == "neutral" and active_gesture == "none" and salient_objects_str == "none":
            # Only speak once on initial appearance
            if self.last_state_signature is not None and self.last_state_signature[0] == person_name:
                return

        # 3. Formulate the situation summary
        context_parts = []
        reason_items = []

        if person_name != "none":
            reason_items.append(f"Person: {person_name}")
            if person_name != "unknown":
                context_parts.append(f"Friend '{person_name}' is here")
            else:
                context_parts.append("A friendly face is in front of camera")

        if active_emotion != "neutral":
            reason_items.append(f"Emotion: {active_emotion.upper()}")
            context_parts.append(f"User's expression is '{active_emotion.upper()}'")

        if active_gesture != "none":
            reason_items.append(f"Gesture: {active_gesture}")
            context_parts.append(f"User is showing hand motion: '{active_gesture}'")

        if salient_objects_str != "none":
            reason_items.append(f"Objects: {salient_objects_str}")
            context_parts.append(f"User is holding: {salient_objects_str}")

        if not context_parts:
            return

        self.last_state_signature = current_sig
        self.is_thinking = True
        context_desc = ". ".join(context_parts)
        reason_str = " | ".join(reason_items)

        prompt = f"{SYSTEM_PROMPT}\n\nCURRENT SITUATION: {context_desc}\n\nGriffin's friendly short reaction (under 8 words):"
        threading.Thread(target=self._query_llm_direct, args=(prompt, reason_str), daemon=True).start()

    def _query_llm_direct(self, prompt, reason="Visual Situation"):
        try:
            self.is_thinking = True
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.65,
                    "num_predict": 20
                }
            }
            req = urllib.request.Request(
                OLLAMA_API_GENERATE,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            reply = data.get("response", "").strip().replace('"', '').replace("'", "")

            if reply:
                print(f"\n[GRIFFIN SAYS]: \"{reply}\"")
                print(f"  🎯 Reason: [{reason}]")
                self.last_comment = reply
                self.last_reason = reason
                self.last_speech_time = time.time()
                self.voice.speak(reply)
        except Exception as e:
            print(f"[Griffin AI Note] {e}")
        finally:
            self.is_thinking = False

    def ask_vision_description(self, bgr_frame, question=None):
        """Uses Moondream Vision model to visually inspect a photo frame."""
        try:
            if not question:
                question = "Describe what you see in 1 short friendly sentence."

            _, buffer = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "model": VISION_LLM,
                "prompt": question,
                "images": [img_b64],
                "stream": False
            }

            req = urllib.request.Request(
                OLLAMA_API_GENERATE,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=12)
            data = json.loads(resp.read().decode("utf-8"))
            reply = data.get("response", "").strip()
            print(f"\n[MOONDREAM VISION]: {reply}")
            print(f"  🎯 Reason: [User requested 'v' Room Snapshot Analysis]")
            self.last_comment = reply
            self.last_reason = "Room Vision Snapshot"
            self.voice.speak(reply)
            return reply
        except Exception as e:
            print(f"[Moondream Vision Error] {e}")
            return None


if __name__ == "__main__":
    from vision import Camera
    from face_id import FaceRecognizer
    from emotions import EmotionDetector
    from objects import ObjectDetector, CATEGORY_COLORS
    from gestures import GestureDetector

    cam = Camera()
    rec = FaceRecognizer()
    emo_det = EmotionDetector()
    obj_det = ObjectDetector()
    gesture_det = GestureDetector()
    ai = NaughtyAI(language_mode=LANGUAGE_MODE, speech_cooldown=4.0)

    print("\n" + "=" * 68)
    print("  🦅 [GRIFFIN: FRIENDLY AI PET ROBOT]")
    print("=" * 68)
    print("-> Mode: English (Friendly, cheerful & concise)")
    print("-> Stays quiet when idle/neutral; speaks only on real events/changes!")
    print("-> Say 'Griffin' to talk | 'v' for vision snapshot | 'q' or ESC to exit.")
    print("=" * 68 + "\n")

    try:
        frame_idx = 0
        current_face = None
        current_emotion = "neutral"
        current_objects = []
        current_gestures = []

        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()

            # 1. Detect Hand Gestures (Parrot perch, praise, fist, etc.)
            current_gestures = gesture_det.detect_gestures(frame)
            if current_gestures:
                gesture_det.draw_overlay(display, current_gestures)

            # 2. Detect Faces & Expressions
            faces = rec.detect(gray)
            if len(faces):
                bx, by, bw, bh = faces[0]
                name, score = rec.identify(gray, faces[0])
                emotion, conf, _ = emo_det.detect_emotion(frame)
                current_face = (name, bx + bw // 2, by + bh // 2)
                current_emotion = emotion

                col = (0, 255, 0) if name != "unknown" else (0, 180, 255)
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), col, 2)
                cv2.putText(display, f"{name.upper()} [{emotion.upper()}]", (bx, max(22, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            else:
                current_face = None
                current_emotion = "neutral"

            # 3. Detect Objects & Dock every 2 frames
            if frame_idx % 2 == 0:
                current_objects = obj_det.detect_objects(frame)

            for obj in current_objects:
                ox, oy, ow, oh = obj["box"]
                lbl = obj["label"]
                color = CATEGORY_COLORS.get(lbl.split()[0], (0, 200, 255))
                cv2.rectangle(display, (ox, oy), (ox + ow, oy + oh), color, 2)
                cv2.putText(display, lbl.upper(), (ox, max(18, oy - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # 4. Trigger Witty Naughty AI Banter ONLY when state changes
            ai.trigger_reaction_async(
                face_info=current_face,
                emotion=current_emotion,
                gestures=current_gestures,
                objects=current_objects,
                bgr_frame=frame
            )

            # Live AI Speech Subtitle & Reason & Language Badge on Display Window
            lang_label = "TANGLISH (0)" if ai.language_mode == 0 else "ENGLISH (1)"
            cv2.putText(display, f"LANG: {lang_label} [Press 'L']", (display.shape[1] - 250, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if ai.last_comment:
                cv2.rectangle(display, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
                cv2.putText(display, f"GRIFFIN: {ai.last_comment[:62]}", (8, frame.shape[0] - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
                if ai.last_reason:
                    cv2.putText(display, f"REASON: [{ai.last_reason[:65]}]", (8, frame.shape[0] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1)

            cv2.imshow("Griffin: Naughty Multimodal AI Pet Robot", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('l'), ord('t')):
                ai.toggle_language()
            elif key == ord('v'):
                print("\n[Asking Moondream Vision LLM to inspect room...]")
                ai.ask_vision_description(frame)

    finally:
        cv2.destroyAllWindows()
        cam.release()
