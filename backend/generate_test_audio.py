"""
Generates two-speaker airline customer service / check-in / boarding
conversations in English (offline TTS) and Arabic (Google TTS).

All output goes into the test_audio/ folder organized by language.

Usage:
    python generate_test_audio.py                # generate everything
    python generate_test_audio.py english        # only English
    python generate_test_audio.py arabic         # only Arabic
    python generate_test_audio.py english:poor   # one specific case
"""

import os
import sys
import io
import pyttsx3
from pydub import AudioSegment

# Force UTF-8 stdout so Arabic descriptions print without crashing on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_DIR = "test_audio"

# ── English cases (pyttsx3 — uses Microsoft David/Zira) ────────────────────

ENGLISH_CASES = {
    "excellent_callcenter": {
        "filename": "excellent_callcenter.wav",
        "description": "Call center: empathetic agent + happy customer",
        "script": [
            ("agent",    "Good morning, thank you for calling SkyAir, my name is Sarah, how may I assist you today?"),
            ("customer", "Hi Sarah, I need to upgrade my flight to business class for next Tuesday."),
            ("agent",    "Absolutely, I'd be delighted to help. Could you please share your booking reference?"),
            ("customer", "Yes it's BK nine nine one one."),
            ("agent",    "Thank you so much. I have one business seat available, and as a loyalty member I can apply a fifteen percent discount."),
            ("customer", "Oh wow, that's wonderful, thank you!"),
            ("agent",    "It's my pleasure. The new boarding pass has been sent to your email. You'll also enjoy lounge access."),
            ("customer", "Sarah you've been amazing, thank you so much."),
            ("agent",    "You're very welcome. Is there anything else I can help you with?"),
            ("customer", "No that's all, have a great day."),
            ("agent",    "Thank you for choosing SkyAir, wishing you a pleasant flight."),
        ],
    },

    "good_checkin_luggage": {
        "filename": "good_checkin_luggage.wav",
        "description": "Check-in counter: passport, ticket and luggage weighing",
        "script": [
            ("agent",    "Good afternoon, welcome to SkyAir check-in. May I have your passport and ticket please?"),
            ("customer", "Yes, here you go."),
            ("agent",    "Thank you Mr. Ahmed. You are flying to Istanbul today, is that correct?"),
            ("customer", "That's correct."),
            ("agent",    "Please place your suitcase on the scale for me."),
            ("customer", "Sure, here it is."),
            ("agent",    "Your luggage weighs twenty four kilograms. Unfortunately the limit is twenty three, so there is a small excess fee of twenty dollars, or you can rearrange items into your carry-on."),
            ("customer", "Oh okay, can I move some clothes to my hand bag?"),
            ("agent",    "Of course, please take your time."),
            ("customer", "Alright, can you weigh it again?"),
            ("agent",    "Twenty two point eight kilograms, perfect. Would you prefer a window or aisle seat?"),
            ("customer", "Window please."),
            ("agent",    "Done. Here is your boarding pass, your gate is B twelve and boarding starts at three thirty PM. Have a pleasant flight."),
            ("customer", "Thank you very much."),
        ],
    },

    "good_boarding_gate": {
        "filename": "good_boarding_gate.wav",
        "description": "Boarding gate: ID check + boarding pass scan",
        "script": [
            ("agent",    "Good evening, boarding pass and passport please."),
            ("customer", "Here you are."),
            ("agent",    "Thank you. Mr. Khan, seat fourteen A. Could you please confirm your final destination?"),
            ("customer", "Dubai."),
            ("agent",    "Perfect. Please proceed down the jet bridge, mind the step. Have a pleasant flight."),
            ("customer", "Thank you."),
            ("agent",    "Next passenger please. Boarding pass and ID."),
            ("customer", "Hi, here you go."),
            ("agent",    "Thank you Ms. Lewis, seat eighteen C. Welcome aboard."),
            ("customer", "Thanks!"),
        ],
    },

    "needs_improvement_luggage": {
        "filename": "needs_improvement_luggage.wav",
        "description": "Check-in: agent rude about overweight bag",
        "script": [
            ("agent",    "Passport and ticket."),
            ("customer", "Hello, here they are."),
            ("agent",    "Put your bag on the scale."),
            ("customer", "Okay."),
            ("agent",    "Your bag is six kilos overweight. You have to pay one hundred and twenty dollars."),
            ("customer", "What? That's a lot, can I move some things to my carry-on?"),
            ("agent",    "The line is long, just pay the fee."),
            ("customer", "But I really don't want to pay that much."),
            ("agent",    "Look, either pay or step aside, other people are waiting."),
            ("customer", "Can I please have a moment?"),
            ("agent",    "Fine but be quick."),
        ],
    },

    "poor_callcenter": {
        "filename": "poor_callcenter.wav",
        "description": "Call center: rude agent, cancelled flight",
        "script": [
            ("agent",    "Yeah, SkyAir."),
            ("customer", "Hello, my flight was cancelled and I'm stuck at the airport!"),
            ("agent",    "Okay, and what do you want me to do about it?"),
            ("customer", "Excuse me? I want a refund or a new flight!"),
            ("agent",    "Cancellations happen, you'll need to wait."),
            ("customer", "Wait for what? I've been here for six hours!"),
            ("agent",    "That's not my problem, I just answer phones."),
            ("customer", "I want to speak to your manager right now!"),
            ("agent",    "There's no manager available."),
            ("customer", "This is the worst service I have ever experienced!"),
            ("agent",    "Well that's your opinion. Anything else?"),
            ("customer", "I will be filing a complaint!"),
            ("agent",    "Sure, whatever, goodbye."),
        ],
    },
}

# ── Arabic cases (gTTS — needs internet) ───────────────────────────────────

ARABIC_CASES = {
    "excellent_checkin": {
        "filename": "excellent_checkin.mp3",
        "description": "تسجيل الدخول: موظف ودود ومسافر سعيد",
        "script": [
            ("agent",    "مرحباً بكم في الخطوط الجوية، أنا سارة، كيف يمكنني مساعدتكم اليوم؟"),
            ("customer", "أهلاً، أريد تسجيل الدخول لرحلتي إلى دبي."),
            ("agent",    "بكل سرور، تفضل بإعطائي جواز السفر والتذكرة من فضلك."),
            ("customer", "تفضلي."),
            ("agent",    "شكراً لك أستاذ أحمد، الرجاء وضع الحقيبة على الميزان."),
            ("customer", "حسناً، تفضلي."),
            ("agent",    "وزن حقيبتك اثنان وعشرون كيلوغرام، ضمن الحد المسموح. هل تفضل مقعد بجانب النافذة أم الممر؟"),
            ("customer", "النافذة من فضلك."),
            ("agent",    "تم الحجز، هذه بطاقة الصعود، البوابة رقم سبعة، والصعود الساعة الثالثة. رحلة سعيدة."),
            ("customer", "شكراً جزيلاً لك، يومٌ سعيد."),
        ],
    },

    "good_boarding_gate": {
        "filename": "good_boarding_gate.mp3",
        "description": "بوابة الصعود: التحقق من الجواز وبطاقة الصعود",
        "script": [
            ("agent",    "مساء الخير، بطاقة الصعود وجواز السفر من فضلك."),
            ("customer", "تفضل."),
            ("agent",    "شكراً، أستاذ خالد، مقعدك رقم اثني عشر، صف ب. تفضل بالصعود."),
            ("customer", "شكراً لك."),
            ("agent",    "المسافر التالي من فضلك."),
            ("customer", "أهلاً، هذه بطاقتي."),
            ("agent",    "أهلاً وسهلاً، رحلة سعيدة."),
        ],
    },

    "poor_luggage": {
        "filename": "poor_luggage.mp3",
        "description": "تسجيل الدخول: موظف غير لطيف بسبب وزن الحقيبة",
        "script": [
            ("agent",    "الجواز والتذكرة."),
            ("customer", "تفضل."),
            ("agent",    "ضع الحقيبة على الميزان بسرعة."),
            ("customer", "حسناً."),
            ("agent",    "حقيبتك زيادة خمسة كيلو، ادفع مئة دولار رسوم زيادة."),
            ("customer", "هل يمكنني نقل بعض الأشياء إلى حقيبة اليد؟"),
            ("agent",    "الطابور طويل، إما تدفع أو تتنحى جانباً."),
            ("customer", "لكن المبلغ كبير، أعطني فرصة من فضلك."),
            ("agent",    "أنا مشغول، الناس تنتظر."),
            ("customer", "هذه طريقة سيئة في التعامل!"),
            ("agent",    "كما تريد، التالي من فضلك."),
        ],
    },
}


# ── TTS helpers ────────────────────────────────────────────────────────────

def get_english_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.stop()
    return voices


def english_speak_to_wav(text: str, voice_id: str, output_path: str, rate: int = 175):
    engine = pyttsx3.init()
    engine.setProperty("voice", voice_id)
    engine.setProperty("rate", rate)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()


def arabic_speak_to_mp3(text: str, output_path: str, voice: str):
    """
    Generate Arabic speech via Microsoft Edge TTS (free, requires internet).
    Multiple distinct male/female Arabic voices — diarizer can tell them apart.
    """
    import asyncio
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    asyncio.run(_run())


def concat_audio(paths: list, out_path: str, output_format: str):
    """Concatenate audio files into a single output file."""
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=300)  # 300ms gap between turns
    for p in paths:
        seg = AudioSegment.from_file(p)
        combined += seg + silence
    combined.export(out_path, format=output_format)


# ── Generators ─────────────────────────────────────────────────────────────

def generate_english_case(case_key: str, voices: list, out_dir: str):
    case = ENGLISH_CASES[case_key]
    print(f"  [EN] {case_key}: {case['description']}")

    # Pick female voice for agent, male for customer (or swap if more variety needed)
    agent_voice    = voices[1].id if len(voices) > 1 else voices[0].id
    customer_voice = voices[0].id

    tmp_dir = os.path.join(out_dir, f"_tmp_{case_key}")
    os.makedirs(tmp_dir, exist_ok=True)

    chunk_paths = []
    for i, (role, text) in enumerate(case["script"]):
        voice = agent_voice if role == "agent" else customer_voice
        # Faster + slightly clipped rate for the rude poor case
        rate = 195 if "poor" in case_key and role == "agent" else 175
        path = os.path.join(tmp_dir, f"{i:02d}.wav")
        english_speak_to_wav(text, voice, path, rate=rate)
        chunk_paths.append(path)

    out_path = os.path.join(out_dir, case["filename"])
    concat_audio(chunk_paths, out_path, output_format="wav")

    for p in chunk_paths:
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(tmp_dir)
    except OSError: pass

    size_kb = os.path.getsize(out_path) / 1024
    print(f"       -> {out_path} ({size_kb:.1f} KB)")


def generate_arabic_case(case_key: str, out_dir: str):
    case = ARABIC_CASES[case_key]
    print(f"  [AR] {case_key}: {case['description']}")

    # Distinct male / female Arabic voices so diarization can separate the speakers
    AGENT_VOICE    = "ar-EG-SalmaNeural"   # female, Egyptian Arabic
    CUSTOMER_VOICE = "ar-SA-HamedNeural"   # male, Saudi Arabic

    tmp_dir = os.path.join(out_dir, f"_tmp_{case_key}")
    os.makedirs(tmp_dir, exist_ok=True)

    chunk_paths = []
    for i, (role, text) in enumerate(case["script"]):
        voice = AGENT_VOICE if role == "agent" else CUSTOMER_VOICE
        path = os.path.join(tmp_dir, f"{i:02d}.mp3")
        arabic_speak_to_mp3(text, path, voice=voice)
        chunk_paths.append(path)

    out_path = os.path.join(out_dir, case["filename"])
    concat_audio(chunk_paths, out_path, output_format="mp3")

    for p in chunk_paths:
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(tmp_dir)
    except OSError: pass

    size_kb = os.path.getsize(out_path) / 1024
    print(f"       -> {out_path} ({size_kb:.1f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = [a.lower() for a in sys.argv[1:]]

    # Parse: "english", "arabic", "english:poor", or empty for all
    do_english = (not args) or any(a == "english" or a.startswith("english:") for a in args)
    do_arabic  = (not args) or any(a == "arabic"  or a.startswith("arabic:")  for a in args)

    en_filter = next((a.split(":", 1)[1] for a in args if a.startswith("english:")), None)
    ar_filter = next((a.split(":", 1)[1] for a in args if a.startswith("arabic:")),  None)

    en_dir = os.path.join(OUTPUT_DIR, "english")
    ar_dir = os.path.join(OUTPUT_DIR, "arabic")

    if do_english:
        os.makedirs(en_dir, exist_ok=True)
        voices = get_english_voices()
        if len(voices) < 2:
            print("WARNING: only one English system voice — speakers may sound similar.")

        print(f"\nEnglish cases (voices: {[v.name for v in voices]})")
        keys = [en_filter] if en_filter else list(ENGLISH_CASES.keys())
        for k in keys:
            if k not in ENGLISH_CASES:
                print(f"  unknown case '{k}'. available: {', '.join(ENGLISH_CASES.keys())}")
                continue
            generate_english_case(k, voices, en_dir)

    if do_arabic:
        os.makedirs(ar_dir, exist_ok=True)
        print(f"\nArabic cases (Edge TTS — needs internet)")
        keys = [ar_filter] if ar_filter else list(ARABIC_CASES.keys())
        for k in keys:
            if k not in ARABIC_CASES:
                print(f"  unknown case '{k}'. available: {', '.join(ARABIC_CASES.keys())}")
                continue
            try:
                generate_arabic_case(k, ar_dir)
            except Exception as e:
                print(f"       FAILED ({e}). Check internet connection.")

    print(f"\nAll files saved under: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
