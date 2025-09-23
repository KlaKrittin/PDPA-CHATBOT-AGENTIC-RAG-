import re
from typing import Dict, List, Tuple, Optional
import logging

class SecurityFilter:
    """
    Security filter for PDPA Assistant to prevent inappropriate content and restrict topics to PDPA only.
    """
    
    def __init__(self):
        # Realguide-style inappropriate patterns - only truly inappropriate content
        # 1) Define Thai inappropriate terms as a list for maintainability
        self.thai_inappropriate_terms = [
            "หี", "จิ๋ม", "จู๋", "ไข่", "หำ", "แตด", "หัวนม", "เย็ด",
            "ปี้", "เสียบ", "เสียว", "เงี่ยน", "ข่มขืน", "รุมโทรม",
            "น้ำแตก", "แตกใน", "สวิงกิ้ง", "สวิ้งกิ้ง", "ดูดปาก", "อมควย", "เลียหี",
            "ตูด", "ส้นตีน", "ตีน", "ตอแหล", "พ่อง", "พ่อมึง", "แม่มึง", "พ่อมึงตาย",
            "แม่มึงตาย", "เหี้ย", "เฮี้ย", "เห้", "ห่า", "สัด", "สัส", "เชี่ย",
            "เชี้ย", "เชี่ยเอ้ย", "เชี้ยเอ้ย", "แม่ง", "มึง", "กู", "ไอสัด", "ไอ้สัด",
            "ไอสัส", "ไอ้สัส", "ไอ้เหี้ย", "อีเหี้ย", "เหี้ยมาก", "เหี้ยสุด", "ไอ้ห่า",
            "ไอสาด", "ไอสาดด", "สัตว์", "ชาติหมา", "ไอ้ชาติหมา", "ไอ้เวร", "ไอ้เวรตะไล",
            "ไอ้ควาย", "ควาย", "ไอ้โง่", "โง่เง่า", "โง่ควาย", "ปัญญาอ่อน", "ไร้สมอง",
            "สมองหมา", "ไอ้บ้า", "สวะ", "เฮงซวย", "ระยำ", "ไอ้ระยำ", "สถุน", "ไอ้สถุน",
            "ต่ำตม", "อัปรีย์", "จัญไร", "กระหรี่", "กะหรี่", "กระรี่", "กะรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี",
            "ชิบหาย", "ชิบหายวายวอด", "ไอ้สารเลว", "สารเลว", "เลว", "สถุนมาก",
            "หน้าด้าน", "หน้าหนา", "หน้าตัวเมีย", "กาก", "ขยะสังคม", "มะเร็งสังคม",
            "ควยใหญ่", "ควยยาว", "ควยเด็ก", "หีบาน", "หีเด็ก", "หีเน่า", "จิ๋มเด็ก",
            "แตกคาปาก", "แตกคา", "แตกใส่หน้า", "แตกใส่ปาก", "เสร็จใน", "ขย่ม", "ขยี้",
            "ขึ้นคร่อม", "กระแทก", "เอากัน", "เอาอย่างว่า", "เอาแรงๆ", "เอาแรงแรง",
            "เลีย", "ดูด",  "แหย่", "เสียบเข้า", "สอดเข้า",
            "ต่ำตม", "อัปรีย์", "จัญไร", "กระหรี่", "กะหรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี"
        ]

        # 2) Additional variant regex fragments that aren't simple literals
        thai_variant_patterns = [
            r"ค\s*วย(?:ย+)?",              # allow spacing and elongation
            r"เอา(?:\s*กัน)?",             # with/without กัน, optional spaces
            r"(?:ไอ้|อี)?\s*เหี้ย(?:ย+)?",   # with/without prefix, elongation
            r"(?:ไอ้|อี)?\s*สัด",          
            r"(?:ไอ้|อี)?\s*สัส",          
            r"(?:ไอ้|อี)?\s*เวร(?:ตะไล)?", 
            r"(?:ไอ้|อี)?\s*ควาย",        
            r"(?:ไอ้|อี)?\s*ระยำ",        
            r"(?:ไอ้|อี)?\s*สถุน",        
            r"ชาติ\s*หมา",                 
            r"หน้า\s*ควย",                 
            r"หน้า\s*หี"                   
        ]

        # 3) Build a single Thai regex from the list + variants
        thai_literals_pattern = "|".join(map(re.escape, self.thai_inappropriate_terms))
        # For Thai, use simple pattern matching without strict word boundaries
        # This is more permissive but catches profanity in context
        thai_pattern = rf"({thai_literals_pattern}|{'|'.join(thai_variant_patterns)})"

        # 4) English inappropriate terms (literals) and variant fragments
        self.english_inappropriate_terms = [
            "fuck", "motherfucker", "mf", "shit", "bullshit", "bs", "bitch",
            "slut", "whore", "cunt", "pussy", "dick", "cock", "asshole",
            "ass", "bastard", "son of a bitch", "sob", "bloody hell",
            "goddamn", "god damn", "damn", "prick", "wanker", "twat"
        ]
        english_variant_patterns = [
            r"f\*+k", r"fxxk", r"sh\*t", r"b!tch"
        ]
        english_literals_pattern = "|".join(map(re.escape, self.english_inappropriate_terms))
        english_pattern = rf"\\b({english_literals_pattern}|{'|'.join(english_variant_patterns)})\\b"

        # 5) Security-related generic terms to avoid (non-PDPA-specific)
        self.security_avoid_terms = [
            "crack", "cracker", "cracking", "exploit", "vulnerability", "intrusion",
            "malware", "virus", "trojan", "ransomware", "phishing", "spam", "ddos",
            "sql injection", "xss", "csrf", "buffer overflow", "privilege escalation",
            "rootkit", "backdoor", "keylogger", "spyware", "adware", "botnet", "worm",
            "logic bomb", "time bomb", "easter egg", "trapdoor", "trap door", "trap-door",
            "trap_door"
        ]
        security_pattern = rf"\\b({'|'.join(map(re.escape, self.security_avoid_terms))})\\b"

        # 6) Violence-related terms (Thai/English)
        self.violence_terms_th = ["ฆ่า", "ฆาตกรรม", "ฆาตกร"]
        # Use proper word boundaries for Thai
        violence_th_pattern = rf"(?<![ก-๙])({'|'.join(map(re.escape, self.violence_terms_th))})(?![ก-๙])"
        self.violence_terms_en = [
            "kill", "murder", "murderer", "assassinate", "assassin", "execute",
            "execution", "suicide", "suicidal", "terrorism", "terrorist", "bomb",
            "explosion", "gun", "weapon", "violence", "violent", "attack", "assault",
            "threat", "threatening"
        ]
        violence_en_pattern = rf"\\b({'|'.join(map(re.escape, self.violence_terms_en))})\\b"

        # 7) Drug-related terms (Thai/English)
        self.drug_terms_th = ["ยาเสพติด", "ยาเสพ"]
        # Use proper word boundaries for Thai
        drug_th_pattern = rf"(?<![ก-๙])({'|'.join(map(re.escape, self.drug_terms_th))})(?![ก-๙])"
        self.drug_terms_en = [
            "drug", "drugs", "heroin", "cocaine", "marijuana", "weed", "meth",
            "amphetamine", "ecstasy", "lsd", "pills", "overdose", "addiction",
            "addict", "dealer", "trafficking", "smuggling"
        ]
        drug_en_pattern = rf"\\b({'|'.join(map(re.escape, self.drug_terms_en))})\\b"

        self.inappropriate_patterns = [
            thai_pattern,
            english_pattern,
            security_pattern,
            violence_th_pattern,
            violence_en_pattern,
            drug_th_pattern,
            drug_en_pattern,
        ]
        
        # PDPA-related keywords (allowed topics)
        self.pdpa_keywords = [
            # Thai PDPA terms
            "PDPA", "Personal Data Protection Act", "คุ้มครองข้อมูลส่วนบุคคล", "พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล", 
            "ข้อมูลส่วนบุคคล", "data controller", "data processor", "ผู้ควบคุมข้อมูล", "ผู้ประมวลผลข้อมูล",
            "สิทธิเจ้าของข้อมูล", "การประมวลผลข้อมูล", "การเก็บรวบรวมข้อมูล", "ฐานทางกฎหมาย",
            "มาตรา","การแจ้งเตือน", "การขอความยินยอม", "การถอนความยินยอม", "การโอนข้อมูล", "การส่งข้อมูลไปต่างประเทศ",
            "การรักษาความปลอดภัย", "การแจ้งเหตุ", "การแจ้งเตือน", "การขอความยินยอม", "การถอนความยินยอม",
            "การโอนข้อมูล", "การส่งข้อมูลไปต่างประเทศ", "การรักษาความปลอดภัย", "การแจ้งเหตุ",
            "คณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "สำนักงานคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล",
            "เจ้าหน้าที่คุ้มครองข้อมูลส่วนบุคคล", "ผู้ตรวจสอบ", "การตรวจสอบ", "การลงโทษ",
            "ค่าปรับ", "โทษทางอาญา", "โทษทางแพ่ง", "การฟ้องร้อง", "การชดเชย",
            # Security-related PDPA terms
            "การแฮกข้อมูลส่วนบุคคล", "การแฮคข้อมูลส่วนบุคคล", "การเจาะข้อมูลส่วนบุคคล",
            "การรั่วไหลข้อมูลส่วนบุคคล", "การรั่วไหลของข้อมูลส่วนบุคคล", "การรั่วไหลข้อมูล",
            "การรั่วไหลของข้อมูล", "การขโมยข้อมูลส่วนบุคคล", "การขโมยข้อมูล",
            "การโจมตีข้อมูลส่วนบุคคล", "การโจมตีข้อมูล", "การบุกรุกข้อมูลส่วนบุคคล",
            "การบุกรุกข้อมูล", "การแจ้งเหตุรั่วไหลข้อมูล", "การแจ้งเหตุรั่วไหลข้อมูลส่วนบุคคล",
            "การแจ้งเหตุการรั่วไหลข้อมูล", "การแจ้งเหตุการรั่วไหลข้อมูลส่วนบุคคล",
            "มาตรการรักษาความปลอดภัยข้อมูล", "มาตรการรักษาความปลอดภัยข้อมูลส่วนบุคคล",
            "การป้องกันข้อมูลส่วนบุคคล", "การป้องกันข้อมูล", "การรักษาความปลอดภัยข้อมูล",
            "การรักษาความปลอดภัยข้อมูลส่วนบุคคล", "การคุ้มครองข้อมูลส่วนบุคคล",
            "การคุ้มครองข้อมูล", "การป้องกันการรั่วไหลข้อมูล", "การป้องกันการรั่วไหลข้อมูลส่วนบุคคล",
            # English PDPA terms
            "personal data", "data protection", "data privacy", "data controller", "data processor",
            "data subject", "data subject rights", "consent", "withdrawal of consent", "data processing",
            "data collection", "data transfer", "cross-border data transfer", "data security",
            "data breach", "data breach notification", "data protection officer", "DPO",
            "data protection impact assessment", "DPIA", "legitimate interest", "legal basis",
            "data minimization", "purpose limitation", "storage limitation", "accuracy",
            "integrity and confidentiality", "accountability", "transparency",
            "right to access", "right to rectification", "right to erasure", "right to be forgotten",
            "right to data portability", "right to object", "right to restrict processing",
            "automated decision making", "profiling", "sensitive personal data", "special categories",
            "children's data", "employee data", "customer data", "vendor data", "third party data",
            "data sharing", "data disclosure", "data retention", "data disposal", "data destruction",
            "privacy policy", "privacy notice", "terms of service", "data processing agreement",
            "binding corporate rules", "standard contractual clauses", "adequacy decision",
            "supervisory authority", "data protection authority", "enforcement", "penalties",
            "fines", "administrative fines", "criminal penalties", "civil remedies",
            "compensation", "damages", "injunction", "cease and desist", "audit", "inspection",
            # English security-related PDPA terms
            "data breach", "data breach notification", "personal data breach", "data security breach",
            "data theft", "personal data theft", "data intrusion", "personal data intrusion",
            "data attack", "personal data attack", "data breach notification", "breach notification",
            "data security measures", "personal data security", "data protection measures",
            "data security", "personal data security", "data protection security",
            "data breach response", "personal data breach response", "breach response plan",
            "data incident response", "personal data incident response", "incident response",
            "data security incident", "personal data security incident", "security incident",
            "data protection incident", "personal data protection incident", "protection incident",
            # Image and visual data related terms
            "ภาพถ่าย", "ถ่ายภาพ", "รูปภาพ", "รูปถ่าย", "ภาพ", "การถ่ายภาพ", "การถ่ายรูป",
            "การแอบถ่าย", "แอบถ่าย", "ถ่ายโดยไม่ได้รับอนุญาต", "ถ่ายโดยไม่ยินยอม",
            "การเผยแพร่ภาพ", "เผยแพร่ภาพ", "การโพสต์ภาพ", "โพสต์ภาพ", "การแชร์ภาพ", "แชร์ภาพ",
            "การลงเน็ต", "ลงเน็ต", "การอัปโหลด", "อัปโหลด", "การเผยแพร่", "เผยแพร่",
            "การละเมิดความเป็นส่วนตัว", "ละเมิดความเป็นส่วนตัว", "การบุกรุกความเป็นส่วนตัว",
            "บุกรุกความเป็นส่วนตัว", "การรุกล้ำความเป็นส่วนตัว", "รุกล้ำความเป็นส่วนตัว",
            "การถ่ายภาพโดยไม่ได้รับอนุญาต", "การถ่ายภาพโดยไม่ยินยอม", "การถ่ายภาพโดยไม่ได้รับความยินยอม",
            "การถ่ายภาพโดยไม่ได้รับอนุญาต", "การถ่ายภาพโดยไม่ยินยอม", "การถ่ายภาพโดยไม่ได้รับความยินยอม",
            "photo", "photograph", "image", "picture", "taking photos", "photography", "camera",
            "unauthorized photography", "unauthorized photo", "unauthorized image", "unauthorized picture",
            "without consent", "without permission", "privacy violation", "privacy breach",
            "image sharing", "photo sharing", "image posting", "photo posting", "image upload",
            "photo upload", "image publication", "photo publication", "image dissemination",
            "photo dissemination", "image distribution", "photo distribution"
        ]
        
        # Compile regex patterns for efficiency
        self.inappropriate_regex = re.compile('|'.join(self.inappropriate_patterns), re.IGNORECASE)
        # Prompt-injection heuristics (Thai/English)
        self.injection_phrases = [
            r"ignore (all|any) (previous|prior) (instructions|messages)",
            r"disregard (the )?(rules|system|guardrails)",
            r"act as (?:an?|the) (?:admin|developer|system)",
            r"reveal (?:your )?(?:system|hidden) prompt",
            r"jailbreak|do-anything-now|DAN",
            r"override safety|bypass safety|disable safety",
            r"เพิกเฉยคำสั่งก่อนหน้า|ละเลยกฎ|แสดง system prompt|ปิดการทำงานความปลอดภัย|ข้ามข้อจำกัด",
        ]
        self.injection_regex = re.compile('|'.join(self.injection_phrases), re.IGNORECASE)

        # Basic PII patterns (best-effort)
        self.email_regex = re.compile(r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+")
        self.phone_regex = re.compile(r"(?:\+?66|0)[\s\-]?(?:\d[\s\-]?){8,10}")
        # Thai National ID (13 digits with hyphens optional)
        self.thai_id_regex = re.compile(r"\b\d{1}-?\d{4}-?\d{5}-?\d{2}-?\d{1}\b")
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """
        Realguide-style content safety check - only block truly inappropriate content.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        if not text:
            return True, []
        
        violations = []
        text_lower = text.lower()
        
        # Only check for truly inappropriate content (profanity, extreme violence, etc.)
        # Remove overly restrictive security term blocking
        matches = self.inappropriate_regex.findall(text_lower)
        if matches:
            # Flatten the matches list since regex groups return tuples
            flat_matches = []
            for match in matches:
                if isinstance(match, tuple):
                    flat_matches.extend([m for m in match if m])
                else:
                    flat_matches.append(match)
            
            if flat_matches:
                violations.append(f"พบคำที่ไม่เหมาะสม: {', '.join(set(flat_matches))}")
        
        # Only block extreme security terms that are clearly malicious
        extreme_security_terms = [
            "create malware", "create virus", "create trojan", "create ransomware",
            "make malware", "make virus", "make trojan", "make ransomware",
            "build malware", "build virus", "build trojan", "build ransomware",
            "develop malware", "develop virus", "develop trojan", "develop ransomware"
        ]
        
        found_extreme_terms = []
        for term in extreme_security_terms:
            if term.lower() in text_lower:
                found_extreme_terms.append(term)
        
        if found_extreme_terms:
            violations.append(f"พบคำที่เกี่ยวข้องกับการสร้างมัลแวร์: {', '.join(found_extreme_terms)}")
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            self.logger.warning(f"Content safety violation detected: {violations}")
        
        return is_safe, violations

    def _is_severe_profanity(self, text: str, violations: List[str]) -> bool:
        """
        Check if the profanity is severe enough to block the question.
        
        Args:
            text: The text to check
            violations: List of safety violations found
            
        Returns:
            True if the profanity is severe, False otherwise
        """
        if not violations:
            return False
        
        # Define severe profanity patterns that should always be blocked
        severe_patterns = [
            # Sexual content
            "เย็ด", "เอา", "ปี้", "สอด", "เสียบ", "เสียว", "เงี่ยน", "ข่มขืน", "รุมโทรม",
            "น้ำแตก", "แตกใน", "สวิงกิ้ง", "สวิ้งกิ้ง", "ดูดปาก", "อมควย", "เลียหี",
            "ควยใหญ่", "ควยยาว", "ควยเด็ก", "หีบาน", "หีเด็ก", "หีเน่า", "จิ๋มเด็ก",
            "แตกคาปาก", "แตกคา", "แตกใส่หน้า", "แตกใส่ปาก", "เสร็จใน", "ขย่ม", "ขยี้",
            "ขึ้นคร่อม", "กระแทก", "เอากัน", "เอาอย่างว่า", "เอาแรงๆ", "เอาแรงแรง",
            "เลีย", "ดูด", "อม", "ยัด", "แหย่", "เสียบเข้า", "สอดเข้า",
            
            # Extreme insults
            "พ่อมึงตาย", "แม่มึงตาย", "ชิบหาย", "ชิบหายวายวอด", "ไอ้สารเลว", "สารเลว",
            "หน้าด้าน", "หน้าหนา", "หน้าตัวเมีย", "กาก", "ขยะสังคม", "มะเร็งสังคม",
            "กระหรี่", "กะหรี่", "กระรี่", "กะรี่", "อีตัว", "อีแพศยา", "แพศยา",
            "อีดอก", "ดอกทอง", "หน้าควย", "หน้าหี", "ไอ้หน้าควย", "ไอ้หน้าหี",
            
            # English severe profanity
            "fuck", "motherfucker", "cunt", "pussy", "dick", "cock", "asshole",
            "bitch", "slut", "whore", "bastard", "son of a bitch"
        ]
        
        text_lower = text.lower()
        
        # Check for severe patterns
        for pattern in severe_patterns:
            if pattern.lower() in text_lower:
                return True
        
        # Check if multiple mild profanities are used together
        mild_profanities = ["มึง", "กู", "เหี้ย", "เฮี้ย", "เห้", "ห่า", "สัด", "สัส", "เชี่ย", "เชี้ย", "แม่ง"]
        mild_count = sum(1 for word in mild_profanities if word in text_lower)
        
        # If more than 2 mild profanities, consider it severe
        if mild_count > 2:
            return True
        
        return False

    def detect_prompt_injection(self, text: str) -> List[str]:
        """
        Detect likely prompt-injection attempts using heuristic patterns.
        Returns list of matched phrases (empty if none).
        """
        if not text:
            return []
        matches = self.injection_regex.findall(text)
        # Normalize matches to strings
        flat = []
        for m in matches:
            if isinstance(m, tuple):
                flat.extend([x for x in m if x])
            else:
                flat.append(m)
        return list(set(flat))

    def sanitize_pii(self, text: str) -> str:
        """
        Redact common PII patterns from text.
        """
        if not text:
            return text
        redacted = self.email_regex.sub('[REDACTED_EMAIL]', text)
        redacted = self.phone_regex.sub('[REDACTED_PHONE]', redacted)
        redacted = self.thai_id_regex.sub('[REDACTED_THAI_ID]', redacted)
        return redacted
    
    def check_topic_restriction(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if the text is related to PDPA topics only.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_pdpa_related, list_of_reasons)
        """
        if not text:
            return False, ["ข้อความว่างเปล่า"]
        
        text_lower = text.lower()
        
        # Check if text contains PDPA-related keywords
        pdpa_matches = []
        for keyword in self.pdpa_keywords:
            if keyword.lower() in text_lower:
                pdpa_matches.append(keyword)
        
        if pdpa_matches:
            return True, [f"เกี่ยวข้องกับ PDPA: {', '.join(pdpa_matches[:3])}"]
        
        # Check for general legal/privacy terms that might be PDPA-related
        legal_privacy_terms = [
            "กฎหมาย", "กฎหมายคุ้มครอง", "กฎหมายข้อมูล", "กฎหมายส่วนบุคคล",
            "privacy", "legal", "law", "regulation", "compliance", "governance",
            "ข้อมูล", "ข้อมูลส่วนบุคคล", "ข้อมูลส่วนตัว", "ข้อมูลส่วนบุคคล",
            "data", "personal", "private", "confidential", "sensitive",
            # Add more general terms that could be related to privacy/data protection
            "ความปลอดภัย", "security", "protection", "คุ้มครอง", "protect",
            "ความเป็นส่วนตัว", "privacy", "ส่วนตัว", "private",
            "ข้อมูล", "information", "data", "สารสนเทศ",
            "กฎหมาย", "law", "regulation", "กฎระเบียบ",
            "การจัดการ", "management", "การบริหาร", "administration"
        ]
        
        legal_matches = []
        for term in legal_privacy_terms:
            if term.lower() in text_lower:
                legal_matches.append(term)
        
        if legal_matches:
            return True, [f"เกี่ยวข้องกับกฎหมาย/ความเป็นส่วนตัว: {', '.join(legal_matches[:3])}"]
        
        return False, ["ไม่เกี่ยวข้องกับ PDPA หรือกฎหมายคุ้มครองข้อมูลส่วนบุคคล"]
    
    def filter_user_input(self, user_input: str) -> Dict[str, any]:
        """
        [CORRECTED LOGIC]
        ตรวจสอบ Input จากผู้ใช้ตามลำดับความสำคัญ:
        1. ตรวจจับการโจมตี (Injection) -> บล็อกทันที
        2. ตรวจจับเนื้อหาไม่เหมาะสม (Profanity) -> บล็อกทันที
        3. ตรวจสอบว่าหัวข้อเกี่ยวกับ PDPA หรือไม่ -> บล็อกถ้าไม่เกี่ยว
        """
        # --- เริ่มต้นด้วยค่าตั้งต้นที่ปลอดภัย ---
        result = {
            "is_safe": True,
            "is_pdpa_related": True,
            "violations": [],
            "reasons": [],
            "filtered_text": user_input,
            "should_respond": True,
            "response_message": ""
        }

        # --- ด่านที่ 1: ตรวจสอบการโจมตี Prompt Injection (สำคัญที่สุด) ---
        injection_hits = self.detect_prompt_injection(user_input)
        if injection_hits:
            result["should_respond"] = False
            result["response_message"] = "ตรวจพบความพยายามในการโจมตีระบบ"
            result["violations"].append(f"Prompt-injection attempt: {', '.join(injection_hits)}")
            return result

        # --- ด่านที่ 2: ตรวจสอบเนื้อหาไม่เหมาะสม / คำหยาบ (สำคัญที่สุด) ---
        is_safe, safety_violations = self.check_content_safety(user_input)
        if not is_safe:
            # บล็อกคำหยาบทันที ไม่ว่าจะเกี่ยวข้องกับ PDPA หรือไม่
            result["should_respond"] = False
            result["response_message"] = "🔴 [Guardrail] บล็อกคำถามเนื่องจากพบคำหยาบ/ไม่เหมาะสม"
            result["violations"].extend(safety_violations)
            return result

        # --- ด่านที่ 3: ตรวจสอบว่าหัวข้อเกี่ยวข้องกับ PDPA หรือไม่ ---
        is_pdpa_related, topic_reasons = self.check_topic_restriction(user_input)
        if not is_pdpa_related:
            result["should_respond"] = False
            result["response_message"] = "ขออภัยค่ะ ฉันสามารถให้ข้อมูลที่เกี่ยวข้องกับ พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล (PDPA) เท่านั้น"
            result["reasons"].extend(topic_reasons)
            return result

        # --- ถ้าผ่านทุกด่าน ถือว่า Input ปลอดภัยและเกี่ยวข้อง ---
        return result
    
    def filter_ai_response(self, ai_response: str) -> Dict[str, any]:
        """
        Filter AI-generated responses for safety.
        
        Args:
            ai_response: The AI's response text
            
        Returns:
            Dictionary with filtering results
        """
        result = {
            "is_safe": True,
            "violations": [],
            "filtered_text": ai_response,
            "should_display": True,
            "replacement_message": ""
        }
        
        # Redact PII first
        ai_response = self.sanitize_pii(ai_response)

        # Check content safety
        is_safe, safety_violations = self.check_content_safety(ai_response)
        result["is_safe"] = is_safe
        result["violations"].extend(safety_violations)
        
        if not is_safe:
            result["should_display"] = False
            result["replacement_message"] = (
                "ขออภัย ฉันไม่สามารถแสดงคำตอบที่มีเนื้อหาไม่เหมาะสมได้ "
                "โปรดลองถามคำถามใหม่ที่เกี่ยวข้องกับ PDPA"
            )
        
        # Detect prompt injection language in the response (unlikely, but safe)
        injection_hits = self.detect_prompt_injection(ai_response)
        if injection_hits:
            result["violations"].append(
                f"ตรวจพบข้อความที่อาจเป็น prompt-injection: {', '.join(injection_hits[:3])}"
            )
        return result
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing or replacing inappropriate content.
        """
        if not text:
            return text

        sanitized = self.inappropriate_regex.sub('[REDACTED]', text)
        sanitized = self.sanitize_pii(sanitized)
        return sanitized
