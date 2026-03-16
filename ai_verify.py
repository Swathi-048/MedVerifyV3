"""
MedVerify AI v3 — 19-Parameter Medicine Authenticity Analyzer
Analyzes medicine packaging image directly. No barcode needed.
"""
 
import base64, re, logging
from datetime import date
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
 
try:
    import pytesseract
    TESS = True
except ImportError:
    TESS = False
 
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR = True
except ImportError:
    PYZBAR = False
 
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
 
def decode_image(b64):
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None
 
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
def resize(img, size=(400, 400)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
 
def result(name, status, score, reason):
    """status: PASS / WARNING / FAIL"""
    return {"name": name, "status": status, "score": score, "reason": reason}
 
def ocr_text(img):
    if not TESS or img is None:
        return ""
    try:
        results = []
        # Try multiple preprocessing approaches
        for scale in [1.0, 2.0]:
            h, w = img.shape[:2]
            scaled = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
 
            # Approach 1: OTSU threshold
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Approach 2: Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 11)
            # Approach 3: Sharpened
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            sharp = cv2.filter2D(gray, -1, kernel)
 
            for proc in [gray, thresh1, thresh2, sharp]:
                pil = Image.fromarray(proc)
                for psm in ["6", "11", "3"]:
                    try:
                        t = pytesseract.image_to_string(pil, config=f"--psm {psm} --oem 3")
                        results.append(t)
                    except:
                        pass
 
        combined = " ".join(results).upper().strip()
        # Clean noise characters but keep alphanumeric + common punctuation
        combined = re.sub(r'[^A-Z0-9\s\.\-/:]', ' ', combined)
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""
 
def levenshtein(a, b):
    a, b = a.lower(), b.lower()
    if len(a) < len(b): return levenshtein(b, a)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a):
        curr = [i+1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca!=cb)))
        prev = curr
    return prev[-1]
 
# ─────────────────────────────────────────────
# GROUP 1 — Spelling & Print Quality
# ─────────────────────────────────────────────
 
def check_drug_name_spelling(text, ref_name=None):
    """Check 1 — Drug name spelling + letter substitution detection"""
    if not text:
        return result("Drug Name Spelling", "WARNING", 0.5,
                      "Could not extract text from image")
 
    # Common letter substitutions used in fakes
    subs = {"0":"O","1":"I","5":"S","8":"B","RN":"M","VV":"W"}
 
    if ref_name:
        ref = ref_name.upper()
        # Strip dosage numbers from ref for matching (e.g. "Paracetamol 500" → "PARACETAMOL")
        ref_drug = re.sub(r'\s*\d+\s*(MG|ML|MCG|G)?\s*$', '', ref).strip()
 
        # Direct exact match
        if ref_drug in text:
            return result("Drug Name Spelling", "PASS", 1.0,
                          f"'{ref_drug}' found correctly spelled on packaging")
 
        # Partial match — at least 70% of drug name present (handles OCR cutting first letter)
        for length in range(len(ref_drug), max(3, len(ref_drug)//2), -1):
            for start in range(len(ref_drug) - length + 1):
                substr = ref_drug[start:start+length]
                if len(substr) >= 5 and substr in text:
                    coverage = length / len(ref_drug)
                    if coverage >= 0.75:
                        return result("Drug Name Spelling", "PASS", 1.0,
                                      f"'{ref_drug}' clearly detected on packaging (partial: '{substr}')")
                    elif coverage >= 0.5:
                        return result("Drug Name Spelling", "WARNING", 0.6,
                                      f"Partial drug name detected: '{substr}' — OCR may have missed some characters")
 
        # Check substitutions
        fake_ver = ref_drug
        for fake, real in subs.items():
            fake_ver = fake_ver.replace(real, fake)
        if fake_ver in text and fake_ver != ref_drug:
            return result("Drug Name Spelling", "FAIL", 0.0,
                          f"Letter substitution detected — '{fake_ver}' found instead of '{ref_drug}'")
 
        # Levenshtein against all words in OCR text
        words = re.findall(r"[A-Z]{4,}", text)
        ref_words = re.findall(r"[A-Z]{4,}", ref_drug)
        if words and ref_words:
            min_dist = min((levenshtein(w, rw) for w in words for rw in ref_words), default=99)
            max_allowed = max(2, len(ref_drug) // 5)  # Allow ~20% error
            if min_dist == 0:
                return result("Drug Name Spelling", "PASS", 1.0,
                              f"Drug name '{ref_drug}' found correctly on packaging")
            elif min_dist <= max_allowed:
                return result("Drug Name Spelling", "WARNING", 0.6,
                              f"Drug name found with minor OCR variation (edit distance: {min_dist})")
            elif min_dist <= max_allowed * 2:
                return result("Drug Name Spelling", "WARNING", 0.4,
                              f"Possible spelling mistake detected (edit distance: {min_dist})")
 
        return result("Drug Name Spelling", "FAIL", 0.0,
                      f"Drug name '{ref_drug}' not clearly found on packaging")
    else:
        words = re.findall(r"[A-Z]{4,}", text)
        if len(words) >= 2:
            return result("Drug Name Spelling", "PASS", 1.0,
                          f"Medicine name text detected: {' '.join(words[:3])}")
        return result("Drug Name Spelling", "WARNING", 0.5,
                      "Could not clearly read medicine name from image")
 
 
def check_brand_name(text, ref_brand=None):
    """Check 2 — Brand/company name spelling"""
    if not text:
        return result("Brand Name", "WARNING", 0.5, "Could not extract text")
 
    pharma_keywords = ["LIMITED","LTD","PHARMA","LABS","INDUSTRIES",
                       "PHARMACEUTICALS","PVT","INC","CORP","CIPLA","MICROLABS",
                       "SUN","MICRO","MANKIND","ABBOTT","ALKEM","TORRENT",
                       "LUPIN","ZYDUS","CADILA","INTAS","GLENMARK","IPCA",
                       "WOCKHARDT","PFIZER","NOVARTIS","GSK","HETERO","STRIDES"]
 
    if ref_brand:
        brand = ref_brand.upper()
 
        # 1. Exact full match
        if brand in text:
            return result("Brand Name", "PASS", 1.0,
                          f"Brand '{ref_brand}' correctly found on packaging")
 
        # 2. Multi-word brand: check each word individually in OCR text
        # e.g. "Micro Labs" -> check if "MICRO" in text OR "LABS" in text
        brand_parts = re.findall(r"[A-Z]{3,}", brand)
        ocr_words = set(re.findall(r"[A-Z]{3,}", text))
        if brand_parts:
            matched = [p for p in brand_parts if p in ocr_words]
            if len(matched) >= 1:
                return result("Brand Name", "PASS", 1.0,
                              f"Brand word '{matched[0]}' from '{ref_brand}' found on packaging")
 
        # 3. Partial prefix match for single-word brands
        for length in range(len(brand), max(4, len(brand)//2), -1):
            substr = brand[:length]
            if len(substr) >= 4 and substr in text:
                return result("Brand Name", "PASS", 1.0,
                              f"Brand '{ref_brand}' detected on packaging")
 
        # 4. Levenshtein — each brand word vs each OCR word
        ocr_word_list = re.findall(r"[A-Z]{3,}", text)
        if ocr_word_list and brand_parts:
            for bp in brand_parts:
                dists = [levenshtein(w, bp) for w in ocr_word_list]
                if min(dists) <= max(1, len(bp)//4):
                    return result("Brand Name", "PASS", 1.0,
                                  f"Brand '{ref_brand}' detected with minor OCR variation")
 
        # 5. Pharma keyword fallback
        found = [k for k in pharma_keywords if k in text]
        if found:
            return result("Brand Name", "PASS", 1.0,
                          f"Pharmaceutical company detected: {found[0]}")
 
        return result("Brand Name", "FAIL", 0.0,
                      f"Brand '{ref_brand}' not found on packaging")
    else:
        found = [k for k in pharma_keywords if k in text]
        if found:
            return result("Brand Name", "PASS", 1.0,
                          f"Company name detected: {found[0]}")
        return result("Brand Name", "WARNING", 0.5,
                      "No company/brand name clearly detected")
 
 
def check_print_quality(img):
    """Check 3 — Print sharpness / blur detection"""
    gray = to_gray(resize(img))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var >= 200:
        return result("Print Quality", "PASS", 1.0,
                      f"Print is sharp and clear (sharpness: {lap_var:.0f})")
    elif lap_var >= 100:
        return result("Print Quality", "WARNING", 0.5,
                      f"Print quality is moderate (sharpness: {lap_var:.0f})")
    else:
        return result("Print Quality", "FAIL", 0.0,
                      f"Print is blurry or smudged — possible photocopy/fake (sharpness: {lap_var:.0f})")
 
 
# ─────────────────────────────────────────────
# GROUP 2 — Packaging Visual Checks
# ─────────────────────────────────────────────
 
def is_blank_image(img):
    """Detect if reference image is a blank placeholder (grey/white/black)"""
    if img is None:
        return True
    gray = to_gray(img)
    std = np.std(gray)
    return std < 15  # Very low variance = blank/solid color image
 
def check_packaging_color(img, ref_img=None):
    """Check 4 — Packaging color analysis"""
    def dominant_color(i):
        hsv = cv2.cvtColor(resize(i), cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv],[0],None,[18],[0,180])
        return int(np.argmax(h_hist)) * 10
 
    if ref_img is not None and not is_blank_image(ref_img):
        dom = dominant_color(img)
        ref_dom = dominant_color(ref_img)
        diff = abs(dom - ref_dom)
        diff = min(diff, 180 - diff)
        if diff <= 20:
            return result("Packaging Color", "PASS", 1.0,
                          f"Packaging color matches reference (hue diff: {diff}°)")
        elif diff <= 40:
            return result("Packaging Color", "WARNING", 0.5,
                          f"Packaging color slightly different from reference (hue diff: {diff}°)")
        else:
            return result("Packaging Color", "FAIL", 0.0,
                          f"Packaging color significantly different from genuine (hue diff: {diff}°)")
    else:
        # Standalone — check color consistency across packaging
        hsv = cv2.cvtColor(resize(img), cv2.COLOR_BGR2HSV)
        s_std = np.std(hsv[:,:,1])
        if s_std < 80:
            return result("Packaging Color", "PASS", 1.0,
                          "Packaging color appears consistent and uniform (no reference in DB)")
        else:
            return result("Packaging Color", "WARNING", 0.5,
                          "Packaging color variation detected — add reference image to DB for comparison")
 
 
def check_layout(img, ref_img=None):
    """Check 5 — Logo/design layout using ORB keypoints"""
    if ref_img is None or is_blank_image(ref_img):
        # Standalone: check that image itself has enough visual structure
        gray = to_gray(img)
        orb = cv2.ORB_create(nfeatures=500)
        kp, _ = orb.detectAndCompute(gray, None)
        if len(kp) >= 50:
            return result("Layout & Design", "PASS", 1.0,
                          f"Packaging has clear printed design ({len(kp)} visual features detected)")
        elif len(kp) >= 20:
            return result("Layout & Design", "WARNING", 0.5,
                          f"Packaging design is faint ({len(kp)} features) — add reference image for comparison")
        else:
            return result("Layout & Design", "FAIL", 0.0,
                          "Very little printed design detected on packaging")
    try:
        g1 = resize(to_gray(img))
        g2 = resize(to_gray(ref_img))
        orb = cv2.ORB_create(nfeatures=500)
        _, d1 = orb.detectAndCompute(g1, None)
        _, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(d1)<10 or len(d2)<10:
            return result("Layout & Design", "WARNING", 0.5,
                          "Could not extract enough features for layout comparison")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = [m for m in bf.match(d1, d2) if m.distance < 50]
        score = min(len(matches)/50.0, 1.0)
        if score >= 0.4:
            return result("Layout & Design", "PASS", score,
                          f"Layout matches reference ({len(matches)} keypoints matched)")
        elif score >= 0.15:
            return result("Layout & Design", "WARNING", score,
                          f"Layout partially matches reference ({len(matches)} keypoints)")
        else:
            return result("Layout & Design", "FAIL", score,
                          f"Layout does not match genuine packaging ({len(matches)} keypoints)")
    except:
        return result("Layout & Design", "WARNING", 0.5, "Layout check could not run")
 
 
def check_texture(img, ref_img=None):
    """Check 6 — Packaging material texture using LBP"""
    def lbp_hist(i):
        gray = to_gray(resize(i, (200,200)))
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        h, _ = np.histogram(lbp.ravel(), bins=np.arange(0,11), range=(0,10))
        h = h.astype(float); h /= (h.sum()+1e-7)
        return h
 
    if ref_img is not None and not is_blank_image(ref_img):
        corr = float(np.corrcoef(lbp_hist(img), lbp_hist(ref_img))[0,1])
        corr = max(0.0, corr)
        if corr >= 0.7:
            return result("Packaging Texture", "PASS", corr,
                          f"Packaging texture matches genuine (correlation: {corr:.2f})")
        elif corr >= 0.4:
            return result("Packaging Texture", "WARNING", corr,
                          f"Packaging texture partially matches (correlation: {corr:.2f})")
        else:
            return result("Packaging Texture", "FAIL", corr,
                          f"Packaging texture very different from genuine (correlation: {corr:.2f})")
    else:
        # Standalone — check texture is consistent (not a photocopy/low quality print)
        gray = to_gray(resize(img, (200,200)))
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,11), range=(0,10))
        hist = hist.astype(float); hist /= hist.sum()+1e-7
        # Good print has varied texture distribution; photocopies are uniform
        uniformity = float(np.max(hist))
        if uniformity < 0.5:
            return result("Packaging Texture", "PASS", 1.0,
                          "Packaging texture is consistent with genuine printed material")
        else:
            return result("Packaging Texture", "WARNING", 0.5,
                          "Add reference image to DB for texture comparison")
 
 
def check_visual_similarity(img, ref_img=None):
    """Check 7 — SSIM overall visual similarity"""
    if ref_img is None or is_blank_image(ref_img):
        # Standalone: check image quality/clarity instead
        gray = to_gray(resize(img))
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > 150:
            return result("Visual Similarity", "PASS", 1.0,
                          "Image quality is good — add reference image to DB for visual comparison")
        else:
            return result("Visual Similarity", "WARNING", 0.5,
                          "No reference image in database — upload a genuine medicine photo as reference")
    g1 = resize(to_gray(img))
    g2 = resize(to_gray(ref_img))
    score, _ = ssim(g1, g2, full=True)
    score = float(score)
    if score >= 0.6:
        return result("Visual Similarity", "PASS", score,
                      f"Packaging looks similar to genuine medicine (SSIM: {score:.2f})")
    elif score >= 0.35:
        return result("Visual Similarity", "WARNING", score,
                      f"Packaging partially resembles genuine (SSIM: {score:.2f})")
    else:
        return result("Visual Similarity", "FAIL", score,
                      f"Packaging looks very different from genuine (SSIM: {score:.2f})")
 
 
# ─────────────────────────────────────────────
# GROUP 3 — Missing Information
# ─────────────────────────────────────────────
 
def check_batch_number(text):
    """Check 8 — Batch number presence"""
    # Look for explicit batch keywords first
    if re.search(r"BATCH[\s\.\-:]*[A-Z0-9]{3,}", text):
        m = re.search(r"BATCH[\s\.\-:]*([A-Z0-9]{3,})", text).group()
        return result("Batch Number", "PASS", 1.0, f"Batch number found: {m}")
    if re.search(r"\bB\.?\s*NO\.?[\s:]*[A-Z0-9]{3,}", text):
        m = re.search(r"B\.?\s*NO\.?[\s:]*([A-Z0-9]{3,})", text).group()
        return result("Batch Number", "PASS", 1.0, f"Batch number found: {m}")
    # Common batch patterns: letters followed by digits
    if re.search(r"\b[A-Z]{1,4}\d{4,10}\b", text):
        match = re.search(r"\b[A-Z]{1,4}\d{4,10}\b", text).group()
        return result("Batch Number", "PASS", 1.0, f"Batch number found: {match}")
    # Check for partial OCR matches - "NO." followed by alphanumeric
    if re.search(r"NO\.?\s*[A-Z0-9]{4,}", text):
        m = re.search(r"NO\.?\s*([A-Z0-9]{4,})", text).group()
        return result("Batch Number", "PASS", 1.0, f"Lot/Batch number found: {m}")
    # If OCR found text but couldn't read batch — warn not fail (OCR limitation)
    if len(text) > 100:
        return result("Batch Number", "WARNING", 0.5,
                      "Batch number not clearly readable — check physical packaging")
    return result("Batch Number", "FAIL", 0.0,
                  "No batch number found — genuine medicines always have batch numbers")
 
 
def check_mfg_date(text):
    """Check 9 — Manufacturing date presence"""
    import re as _re
    patterns = [
        r"(?:MFD|MFG|MANUFACTURED|MANUF|MF\.D)[:\s\.]*\d{2}[/\-\.]\d{2,4}",
        r"(?:MFD|MFG)[:\s\.]*[A-Z]{3,}[/\-\.\s]\d{4}",
        r"(?:DATE\s*OF\s*MFG|DOM)[:\s\.]*\d{2}[/\-\.]\d{2,4}",
        r"MFD[:\s\.]*\d{4}",
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s\.]*\d{4}\b",
        r"\b\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}\b",
        r"\b\d{2}/\d{4}\b",
    ]
    for pat in patterns:
        if _re.search(pat, text):
            m = _re.search(pat, text).group()
            return result("Manufacturing Date", "PASS", 1.0, f"Manufacturing date found: {m}")
 
    # Look for date-like patterns near MFD/MFG keywords with some tolerance
    # OCR often garbles the keyword itself
    mfg_variants = ["MFD", "MFG", "MANUFACTURED", "MANUF", "MFR", "MFRD",
                    "MF D", "M.F.D", "MFO", "NFD", "MED"]
    if any(k in text for k in mfg_variants):
        return result("Manufacturing Date", "WARNING", 0.5,
                      "Manufacturing date keyword found but date not clearly readable — check physical packaging")
 
    # If substantial text extracted — likely OCR missed MFD label
    if len(text) > 150:
        return result("Manufacturing Date", "WARNING", 0.5,
                      "Manufacturing date not clearly visible in image — check physical packaging")
    return result("Manufacturing Date", "FAIL", 0.0,
                  "No manufacturing date found — required on genuine medicines")
 
 
def check_expiry_date(text):
    """Check 10 — Expiry date presence, validity, and tampering"""
    patterns = [
        r"(?:EXP|EXPIRY|EXPDATE|USE\s*BEFORE|USE\s*BY)[:\s.]*(\d{2})[/\-.](\d{4})",
        r"(?:EXP|EXPIRY)[:\s.]*(\d{2})[/\-.](\d{2})",
        r"(?:EXP|EXPIRY)[:\s.]*([A-Z]{3,})[/\-.\s](\d{4})",
        r"EXP[:\s.]*(\d{4})",
        r"(?:EXP|EXPIRY|USE\s*BEF)[:\s.]*\d{1,2}[/\-.\s]\d{2,4}",
        r"(?:EXP|EXPIRY)[:\s.]*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s\.]\d{4}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                raw = m.group()
                nums = re.findall(r"\d+", raw)
                if len(nums) >= 2:
                    month, year = int(nums[0]), int(nums[1])
                    if year < 100: year += 2000
                    if 1 <= month <= 12 and 2000 <= year <= 2050:
                        exp = date(year, month, 1)
                        if exp < date.today():
                            return result("Expiry Date", "FAIL", 0.0,
                                          f"MEDICINE IS EXPIRED! Expiry: {month:02d}/{year}")
                        else:
                            return result("Expiry Date", "PASS", 1.0,
                                          f"Valid expiry date found: {month:02d}/{year}")
            except:
                pass
            return result("Expiry Date", "PASS", 1.0, f"Expiry date field found: {m.group()}")
 
    exp_variants = ["EXP", "EXPIRY", "USE BEFORE", "USE BY", "EXP.", "EXPDT", "EXPI"]
    if any(k in text for k in exp_variants):
        return result("Expiry Date", "WARNING", 0.5,
                      "Expiry keyword found but date not clearly readable — check physical packaging")
    if len(text) > 150:
        return result("Expiry Date", "WARNING", 0.5,
                      "Expiry date not clearly visible in image — check physical packaging")
    return result("Expiry Date", "FAIL", 0.0,
                  "No expiry date found — required on all genuine medicines")
 
 
def check_license_number(text):
    """Check 11 — Manufacturing license number"""
    patterns = [
        r"\bML[\s/\-]?\w{4,15}\b",
        r"\bDL[\s/\-]?\w{4,15}\b",
        r"(?:MFG|MFR)[\s.]*LIC[\s.]*(?:NO)?[\s.:]*[\w/\-]{4,20}",
        r"LIC[\s.]*NO[\s.:]*[\w/\-]{4,20}",
        r"LICENCE[\s.]*NO[\s.:]*[\w/\-]{4,20}",
        r"LIC\.?\s*NO\.?\s*[A-Z0-9/\-]{4,}",
        r"W/S[A-Z0-9/\-]{3,}",   # Common Indian drug license format
        r"G/\d{4,}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return result("License Number", "PASS", 1.0,
                          f"Manufacturing license found: {m.group()[:35]}")
    # If significant text was extracted, downgrade to WARNING not FAIL
    if len(text) > 100:
        return result("License Number", "WARNING", 0.5,
                      "License number not clearly readable — check physical packaging for Mfg. Lic. No.")
    return result("License Number", "FAIL", 0.0,
                  "No manufacturing license number found — required on genuine Indian medicines")
 
 
# ─────────────────────────────────────────────
# GROUP 4 — Date Tampering Detection
# ─────────────────────────────────────────────
 
def check_date_tampering(img, text):
    """Check 12 — Detect overwritten/tampered dates"""
    issues = []
 
    # Check OCR confidence on date regions
    if TESS:
        try:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if str(c).isdigit() and int(c) > 0]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                date_words = [i for i, w in enumerate(data['text'])
                              if re.search(r'\d{2}[/\-]\d{2,4}', w)]
                if date_words:
                    date_confs = [int(data['conf'][i]) for i in date_words
                                  if str(data['conf'][i]).isdigit()]
                    if date_confs:
                        date_avg = sum(date_confs) / len(date_confs)
                        if date_avg < avg_conf * 0.6:
                            issues.append("Date text has lower print confidence than surrounding text")
        except:
            pass
 
    # Check for noise/artifacts in image regions with dates
    gray = to_gray(img)
    # Look for high-frequency noise which indicates overprinting
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_score = float(np.std(lap))
    if noise_score > 80:
        issues.append("High image noise detected — possible overprinting or alteration")
 
    if issues:
        return result("Date Tampering", "FAIL", 0.0,
                      "Tampering signs: " + "; ".join(issues))
 
    if "EXP" in text or "MFD" in text:
        return result("Date Tampering", "PASS", 1.0,
                      "No tampering signs detected on date fields")
    return result("Date Tampering", "WARNING", 0.5,
                  "Could not fully verify date integrity")
 
 
# ─────────────────────────────────────────────
# GROUP 5 — Physical Tablet Appearance
# ─────────────────────────────────────────────
 
def detect_tablets(img):
    """Detect individual tablet/capsule contours in image"""
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (9,9), 2)
    # Try HoughCircles for round tablets
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=20, param1=50, param2=30,
                                minRadius=8, maxRadius=60)
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return None
 
 
def check_tablet_color_uniformity(img):
    """Check 13 — All tablets same color?"""
    circles = detect_tablets(img)
    if circles is None or len(circles) < 2:
        return result("Tablet Color Uniformity", "WARNING", 0.5,
                      "Could not detect individual tablets — ensure tablets are clearly visible")
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hues = []
    for (x, y, r) in circles[:12]:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x,y), max(r-3,3), 255, -1)
        region_hue = hsv[:,:,0][mask > 0]
        if len(region_hue) > 0:
            hues.append(float(np.median(region_hue)))
 
    if len(hues) < 2:
        return result("Tablet Color Uniformity", "WARNING", 0.5,
                      "Could not sample tablet colors")
 
    hue_std = np.std(hues)
    if hue_std < 8:
        return result("Tablet Color Uniformity", "PASS", 1.0,
                      f"All {len(hues)} tablets have uniform color (std: {hue_std:.1f})")
    elif hue_std < 20:
        return result("Tablet Color Uniformity", "WARNING", 0.5,
                      f"Slight color variation across tablets (std: {hue_std:.1f})")
    else:
        return result("Tablet Color Uniformity", "FAIL", 0.0,
                      f"Significant color difference across tablets detected (std: {hue_std:.1f}) — genuine tablets are uniform")
 
 
def check_tablet_size_consistency(img):
    """Check 14 — All tablets same size?"""
    circles = detect_tablets(img)
    if circles is None or len(circles) < 2:
        return result("Tablet Size Consistency", "WARNING", 0.5,
                      "Could not detect enough tablets for size analysis")
 
    radii = [r for (_,_,r) in circles[:12]]
    mean_r = np.mean(radii)
    cv_r = np.std(radii) / mean_r * 100  # coefficient of variation %
 
    if cv_r < 10:
        return result("Tablet Size Consistency", "PASS", 1.0,
                      f"All {len(radii)} tablets are consistent in size (variation: {cv_r:.1f}%)")
    elif cv_r < 20:
        return result("Tablet Size Consistency", "WARNING", 0.5,
                      f"Minor size variation across tablets (variation: {cv_r:.1f}%)")
    else:
        return result("Tablet Size Consistency", "FAIL", 0.0,
                      f"Tablets have inconsistent sizes (variation: {cv_r:.1f}%) — fake medicines often have uneven tablets")
 
 
def check_tablet_shape(img):
    """Check 15 — Tablet shape regularity"""
    circles = detect_tablets(img)
    if circles is not None and len(circles) >= 2:
        return result("Tablet Shape", "PASS", 1.0,
                      f"Detected {len(circles)} uniformly circular tablets")
 
    # Fallback: contour circularity check
    gray = to_gray(img)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tablet_contours = [c for c in contours if 200 < cv2.contourArea(c) < 8000]
 
    if not tablet_contours:
        return result("Tablet Shape", "WARNING", 0.5,
                      "Could not detect tablet shapes clearly")
 
    circularities = []
    for c in tablet_contours[:10]:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circ = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(circ)
 
    if not circularities:
        return result("Tablet Shape", "WARNING", 0.5, "Shape analysis inconclusive")
 
    avg_circ = np.mean(circularities)
    if avg_circ > 0.7:
        return result("Tablet Shape", "PASS", 1.0,
                      f"Tablets have regular shape (circularity: {avg_circ:.2f})")
    elif avg_circ > 0.4:
        return result("Tablet Shape", "WARNING", 0.5,
                      f"Some irregular tablet shapes detected (circularity: {avg_circ:.2f})")
    else:
        return result("Tablet Shape", "FAIL", 0.0,
                      f"Irregular tablet shapes detected (circularity: {avg_circ:.2f}) — genuine tablets are uniform")
 
 
def check_blister_pattern(img):
    """Check 16 — Blister pack cell spacing regularity"""
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blister_cells = [c for c in contours if 300 < cv2.contourArea(c) < 15000]
 
    if len(blister_cells) < 3:
        return result("Blister Pattern", "WARNING", 0.5,
                      "Could not detect enough blister cells for pattern analysis")
 
    centers = []
    for c in blister_cells[:16]:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
 
    if len(centers) < 3:
        return result("Blister Pattern", "WARNING", 0.5,
                      "Could not compute blister cell positions")
 
    # Check spacing consistency
    centers.sort(key=lambda p: (p[1]//50, p[0]))
    spacings = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        spacings.append(np.sqrt(dx*dx + dy*dy))
 
    spacing_cv = np.std(spacings) / (np.mean(spacings) + 1e-7) * 100
 
    if spacing_cv < 20:
        return result("Blister Pattern", "PASS", 1.0,
                      f"Blister cells are evenly spaced ({len(centers)} cells detected)")
    elif spacing_cv < 40:
        return result("Blister Pattern", "WARNING", 0.5,
                      f"Slight irregularity in blister spacing (variation: {spacing_cv:.1f}%)")
    else:
        return result("Blister Pattern", "FAIL", 0.0,
                      f"Irregular blister pack pattern — genuine packs have uniform cell spacing (variation: {spacing_cv:.1f}%)")
 
 
# ─────────────────────────────────────────────
# GROUP 6 — Security Features
# ─────────────────────────────────────────────
 
def check_qr_barcode(img_b64, img):
    """Check 17 — QR code / barcode presence"""
    if PYZBAR:
        codes = pyzbar_decode(img)
        if codes:
            data = codes[0].data.decode("utf-8")
            return result("QR/Barcode", "PASS", 1.0,
                          f"Security code found: {data[:40]}")
 
    # OpenCV QR fallback
    gray = to_gray(img)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(gray)
    if data:
        return result("QR/Barcode", "PASS", 1.0,
                      f"QR code found: {data[:40]}")
 
    return result("QR/Barcode", "WARNING", 0.5,
                  "No QR code or barcode detected — many genuine medicines include security codes")
 
 
def check_hologram(img):
    """Check 18 — Hologram / security sticker detection"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Holograms have high saturation and varying hue
    high_sat = (hsv[:,:,1] > 180).astype(np.uint8)
    high_sat_pct = np.sum(high_sat) / high_sat.size * 100
 
    # Also check for metallic/shiny regions (high value, high saturation)
    metallic = ((hsv[:,:,1] > 120) & (hsv[:,:,2] > 180)).astype(np.uint8)
    metallic_pct = np.sum(metallic) / metallic.size * 100
 
    if high_sat_pct > 2.0 or metallic_pct > 1.5:
        return result("Hologram/Security Sticker", "PASS", 1.0,
                      f"Possible hologram or security feature detected (shiny region: {max(high_sat_pct, metallic_pct):.1f}%)")
    elif high_sat_pct > 0.5:
        return result("Hologram/Security Sticker", "WARNING", 0.5,
                      "Faint security feature detected — verify hologram physically")
    else:
        return result("Hologram/Security Sticker", "WARNING", 0.5,
                      "No hologram detected — some genuine medicines may not have holograms")
 
 
def check_seal_integrity(img):
    """Check 19 — Pack seal / border integrity"""
    gray = to_gray(img)
    edges = cv2.Canny(gray, 50, 150)
 
    h, w = edges.shape
    # Check border edges (outer 5% of image)
    border_mask = np.zeros_like(edges)
    border_w = max(int(w * 0.05), 5)
    border_h = max(int(h * 0.05), 5)
    border_mask[:border_h, :] = 1
    border_mask[-border_h:, :] = 1
    border_mask[:, :border_w] = 1
    border_mask[:, -border_w:] = 1
 
    border_edge_pct = np.sum((edges > 0) & (border_mask > 0)) / (np.sum(border_mask) + 1e-7) * 100
 
    # Check for broken/inconsistent edges in pack interior
    interior = edges[border_h:-border_h, border_w:-border_w]
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
 
    if len(large_contours) >= 1 and border_edge_pct > 5:
        return result("Seal Integrity", "PASS", 1.0,
                      "Pack borders appear intact and unbroken")
    elif border_edge_pct > 2:
        return result("Seal Integrity", "WARNING", 0.5,
                      "Pack seal is partially visible — check physical seal carefully")
    else:
        return result("Seal Integrity", "WARNING", 0.5,
                      "Could not fully assess seal integrity from image")
 
 
# ─────────────────────────────────────────────
# OCR-based medicine identification
# ─────────────────────────────────────────────
 
def identify_medicine(text, all_records):
    """Try to identify medicine from OCR text"""
    if not text or not all_records:
        return None, 0.0
 
    best, best_score = None, 0.0
    for rec in all_records:
        name = rec.get("name","").upper()
        brand = rec.get("brand","").upper()
        ref_text = rec.get("ocr_text","").upper()
 
        score = 0.0
        if name and name in text: score += 0.5
        if brand and brand in text: score += 0.3
 
        # Word overlap
        user_words = set(re.findall(r"[A-Z]{3,}", text))
        ref_words = set(re.findall(r"[A-Z]{3,}", ref_text))
        if ref_words:
            overlap = len(user_words & ref_words) / len(ref_words)
            score += overlap * 0.2
 
        if score > best_score:
            best_score, best = score, rec
 
    return best, best_score
 
 
# ─────────────────────────────────────────────
# MASTER ANALYZE FUNCTION
# ─────────────────────────────────────────────
 
WEIGHTS = {
    "group1": 0.20,  # spelling & print
    "group2": 0.20,  # packaging visual
    "group3": 0.25,  # missing info
    "group4": 0.15,  # tampering
    "group5": 0.10,  # tablet appearance
    "group6": 0.10,  # security
}
 
def analyze(user_b64, all_records):
    """
    Main analysis function.
    Returns full result with all 19 checks + final verdict.
    """
    img = decode_image(user_b64)
    if img is None:
        return {"error": "Could not decode image"}
 
    text = ocr_text(img)
 
    # Try to identify medicine from DB
    matched_record, match_confidence = identify_medicine(text, all_records)
    ref_img = decode_image(matched_record["image"]) if (
        matched_record and matched_record.get("image") and
        len(matched_record.get("image","")) > 1000
    ) else None
 
    ref_name  = matched_record.get("name")  if matched_record else None
    ref_brand = matched_record.get("brand") if matched_record else None
 
    # ── Run all 19 checks ──
    checks = []
 
    # Group 1 — Spelling & Print
    checks.append(check_drug_name_spelling(text, ref_name))
    checks.append(check_brand_name(text, ref_brand))
    checks.append(check_print_quality(img))
 
    # Group 2 — Packaging Visual
    checks.append(check_packaging_color(img, ref_img))
    checks.append(check_layout(img, ref_img))
    checks.append(check_texture(img, ref_img))
    checks.append(check_visual_similarity(img, ref_img))
 
    # Group 3 — Missing Information
    checks.append(check_batch_number(text))
    checks.append(check_mfg_date(text))
    checks.append(check_expiry_date(text))
    checks.append(check_license_number(text))
 
    # Group 4 — Tampering
    checks.append(check_date_tampering(img, text))
 
    # Group 5 — Tablet Appearance
    checks.append(check_tablet_color_uniformity(img))
    checks.append(check_tablet_size_consistency(img))
    checks.append(check_tablet_shape(img))
    checks.append(check_blister_pattern(img))
 
    # Group 6 — Security
    checks.append(check_qr_barcode(user_b64, img))
    checks.append(check_hologram(img))
    checks.append(check_seal_integrity(img))
 
    # ── Score by group ──
    g1 = [checks[0],  checks[1],  checks[2]]
    g2 = [checks[3],  checks[4],  checks[5],  checks[6]]
    g3 = [checks[7],  checks[8],  checks[9],  checks[10]]
    g4 = [checks[11]]
    g5 = [checks[12], checks[13], checks[14], checks[15]]
    g6 = [checks[16], checks[17], checks[18]]
 
    def group_score(grp):
        return sum(c["score"] for c in grp) / len(grp)
 
    composite = (
        WEIGHTS["group1"] * group_score(g1) +
        WEIGHTS["group2"] * group_score(g2) +
        WEIGHTS["group3"] * group_score(g3) +
        WEIGHTS["group4"] * group_score(g4) +
        WEIGHTS["group5"] * group_score(g5) +
        WEIGHTS["group6"] * group_score(g6)
    )
 
    # ── Verdict ──
    if composite >= 0.65:
        verdict = "GENUINE"
    elif composite >= 0.40:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FAKE"
 
    # ── Summary stats ──
    passed   = sum(1 for c in checks if c["status"] == "PASS")
    warnings = sum(1 for c in checks if c["status"] == "WARNING")
    failed   = sum(1 for c in checks if c["status"] == "FAIL")
 
    return {
        "verdict":          verdict,
        "composite_score":  round(composite, 4),
        "checks":           checks,
        "passed":           passed,
        "warnings":         warnings,
        "failed":           failed,
        "total_checks":     len(checks),
        "identified_as":    ref_name,
        "identified_brand": ref_brand,
        "match_confidence": round(match_confidence, 3),
        "extracted_text":   text[:500] if text else None,
        "reference_used":   ref_img is not None,
    }
