import difflib
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def normalize_ar(text):
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

api_tokens = ['أما', 'وأنه', 'صار', 'الاحتفال', 'عند', 'نتنياهو', 'أن', 'الـ', '64', 'صاروا', '54', 'مش', '50', 'فهذا', 'اعتراف', 'بالهزيمة']
whisper_tokens = [
    {'word': 'أما', 'start': 0.00, 'end': 0.54},
    {'word': 'وأنه', 'start': 0.54, 'end': 1.08},
    {'word': 'سال', 'start': 1.08, 'end': 1.44},
    {'word': 'احتفال', 'start': 1.44, 'end': 2.04},
    {'word': 'أن', 'start': 2.04, 'end': 2.30},
    {'word': 'تنطنياهه', 'start': 2.30, 'end': 3.46},
    {'word': 'أنه', 'start': 3.46, 'end': 3.76},
    {'word': 'الأربع', 'start': 3.76, 'end': 4.08},
    {'word': 'وستين', 'start': 4.08, 'end': 4.60},
    {'word': 'صارو', 'start': 4.60, 'end': 4.92},
    {'word': 'أربع', 'start': 4.92, 'end': 5.20},
    {'word': 'وخمسين', 'start': 5.20, 'end': 5.78},
    {'word': 'مش', 'start': 5.78, 'end': 6.06},
    {'word': 'ومشخمسين', 'start': 6.06, 'end': 7.32},
    {'word': 'فهذا', 'start': 7.32, 'end': 8.18},
    {'word': 'اعتراف', 'start': 8.18, 'end': 9.40},
    {'word': 'بالهزينة', 'start': 9.40, 'end': 10.38}
]

def align_ground_truth(api_words, whisper_words):
    """
    Map ground-truth API words to Whisper audio time intervals.
    """
    norm_api = [normalize_ar(w) for w in api_words]
    norm_wsp = [normalize_ar(w['word']) for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, norm_api, norm_wsp)
    aligned_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        api_slice = api_words[i1:i2]
        wsp_slice = whisper_words[j1:j2]
        
        if not api_slice:
            continue
            
        if wsp_slice:
            block_start = wsp_slice[0]['start']
            block_end = wsp_slice[-1]['end']
        else:
            # Fallback if no matching whisper block
            prev_end = aligned_words[-1]['end'] if aligned_words else 0.0
            block_start = prev_end
            block_end = prev_end + 0.4 * len(api_slice)

        block_dur = max(0.2, block_end - block_start)
        weights = [max(1, len(w)) for w in api_slice]
        tot_wt = sum(weights)
        curr = block_start

        for w, wt in zip(api_slice, weights):
            dur = (wt / tot_wt) * block_dur
            aligned_words.append({
                'word': w,
                'start': round(curr, 2),
                'end': round(curr + dur, 2)
            })
            curr += dur

    return aligned_words

aligned = align_ground_truth(api_tokens, whisper_tokens)
for idx, a in enumerate(aligned):
    print(f"{idx:2d}: {a['word']:12} | [{a['start']:.2f}s -> {a['end']:.2f}s]")
