"""Deterministic safety detection and localized escalation guidance."""

from __future__ import annotations

import re

RISK_PATTERNS = {
    "possible malware or ransomware": (
        r"\bmalware\b|\bvirus\b|\bransomware\b|\bphishing\b|"
        r"\blink(?:u)?\s+i\s+dyshimt[eë]\b|\bemail\s+i\s+dyshimt[eë]\b|"
        r"\bschadsoftware\b|\btrojaner\b|\bverd[aä]chtige\s+mail\b"
    ),
    "possible account compromise": (
        r"\bhacked\b|\bcompromised\b|\bunauthorized\b|\bsuspicious login\b|"
        r"\bhakuar\b|\bkomprometuar\b|\bhyrje\s+e\s+dyshimt[eë]\b|"
        r"\bgehackt\b|\bkompromittiert\b|\bverd[aä]chtige\s+anmeldung\b"
    ),
    "possible data loss": (
        r"\bdata loss\b|\bdeleted files?\b|\bmissing files?\b|\blost files?\b|"
        r"\bhumb(?:ur|je)\s+(?:e\s+)?t[eë]\s+dh[eë]nave\b|\bskedar(?:i|[eë]t)?\s+(?:u\s+)?fshi|"
        r"\bdatenverlust\b|\bgel[oö]schte\s+dateien?\b|\bfehlende\s+dateien?\b"
    ),
    "possible company-wide outage": (
        r"\boutage\b|\beveryone\b.*\bdown\b|\bcompany[- ]wide\b|"
        r"\bnd[eë]rprerje\b|\bt[eë]\s+gjith[eë](?:ve)?\b.*\bnuk\s+(?:u\s+)?punon\b|"
        r"\bfirmenweiter\s+ausfall\b|\bbei\s+allen\b.*\bausgefallen\b"
    ),
    "possible physical danger": (
        r"\bsmoke\b|\bburning smell\b|\bsparks?\b|\boverheat(?:ing)?\b|\bfire\b|"
        r"\btym(?:i)?\b|\ber[eë]\s+(?:e\s+)?djeg(?:ies|ur)\b|\bshk[eë]ndij(?:a|ë)\b|"
        r"\bmbinxeh(?:et|je)\b|\bzjarr\b|"
        r"\brauch\b|\bbrandgeruch\b|\bfunken\b|\b[uü]berhitz(?:t|ung)\b|\bfeuer\b"
    ),
}

RISK_ACTIONS = {
    "en": {
        "possible malware or ransomware": [
            "Disconnect the affected device from Wi-Fi, Ethernet, and VPN if it is safe to do so.",
            "Stop entering credentials and contact your IT or security team immediately.",
        ],
        "possible account compromise": [
            "Use a different trusted device to contact IT or security and secure the account.",
            "Do not approve unexpected MFA prompts or share verification codes.",
        ],
        "possible data loss": [
            "Stop writing new data to the affected device or drive.",
            "Do not reinstall, reset, delete, or run cleanup tools before IT reviews the issue.",
        ],
        "possible company-wide outage": [
            "Report the incident through the official help-desk or outage channel.",
            "Avoid repeated resets or configuration changes across multiple devices.",
        ],
        "possible physical danger": [
            "Move away from the device and warn nearby people.",
            "If it is safe, disconnect power without touching a hot, wet, sparking, or smoking device.",
            "Contact building safety or emergency services when there is fire, smoke, or immediate danger.",
        ],
    },
    "sq": {
        "possible malware or ransomware": [
            "Shkëpute pajisjen nga Wi-Fi, Ethernet dhe VPN, nëse mund ta bësh pa rrezik.",
            "Mos shkruaj më kredenciale dhe kontakto menjëherë ekipin e IT-së ose sigurisë.",
        ],
        "possible account compromise": [
            "Përdor një pajisje tjetër të besueshme për ta kontaktuar IT-në dhe për ta siguruar llogarinë.",
            "Mos aprovo kërkesa të papritura MFA dhe mos ndaj kode verifikimi.",
        ],
        "possible data loss": [
            "Ndalo shkrimin e të dhënave të reja në pajisjen ose diskun e prekur.",
            "Mos riinstalo, reset, fshi ose përdor mjete pastrimi para se ta kontrollojë IT-ja.",
        ],
        "possible company-wide outage": [
            "Raporto incidentin në kanalin zyrtar të help-desk-ut ose të ndërprerjeve.",
            "Shmang reset-et dhe ndryshimet e përsëritura në shumë pajisje.",
        ],
        "possible physical danger": [
            "Largohu nga pajisja dhe paralajmëro njerëzit afër.",
            "Nëse është e sigurt, shkëpute energjinë pa prekur pajisje të nxehtë, të lagur, me shkëndija ose tym.",
            "Kontakto sigurinë e objektit ose shërbimet emergjente kur ka zjarr, tym apo rrezik të menjëhershëm.",
        ],
    },
    "de": {
        "possible malware or ransomware": [
            "Trenne das betroffene Gerät von WLAN, Ethernet und VPN, sofern dies sicher möglich ist.",
            "Gib keine weiteren Zugangsdaten ein und kontaktiere sofort IT oder Security.",
        ],
        "possible account compromise": [
            "Nutze ein anderes vertrauenswürdiges Gerät, um IT zu kontaktieren und das Konto zu sichern.",
            "Bestätige keine unerwarteten MFA-Anfragen und teile keine Bestätigungscodes.",
        ],
        "possible data loss": [
            "Schreibe keine neuen Daten auf das betroffene Gerät oder Laufwerk.",
            "Installiere, lösche oder setze nichts zurück, bevor die IT den Vorfall geprüft hat.",
        ],
        "possible company-wide outage": [
            "Melde den Vorfall über den offiziellen Helpdesk- oder Störungskanal.",
            "Vermeide wiederholte Resets oder Konfigurationsänderungen auf mehreren Geräten.",
        ],
        "possible physical danger": [
            "Entferne dich vom Gerät und warne Personen in der Nähe.",
            "Trenne, wenn gefahrlos möglich, die Stromversorgung, ohne ein heißes, nasses, funkendes oder rauchendes Gerät zu berühren.",
            "Kontaktiere bei Feuer, Rauch oder unmittelbarer Gefahr den Gebäudeschutz oder den Notruf.",
        ],
    },
}

HEADINGS = {
    "en": "Safety first — escalate this now:",
    "sq": "Siguria së pari — përshkallëzoje menjëherë:",
    "de": "Sicherheit zuerst — jetzt eskalieren:",
}


def detect_patterns(text: str, patterns: dict[str, str]) -> list[str]:
    return sorted(
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    )


def detect_risk_flags(text: str) -> list[str]:
    return detect_patterns(text, RISK_PATTERNS)


def detect_response_language(text: str) -> str:
    lowered = text.lower()
    albanian = len(
        re.findall(
            r"\b(?:nuk|është|eshte|kam|pajisj\w*|llogari\w*|fjalëkalim\w*|skedar\w*|"
            r"shkëput\w*|shkendij\w*|zjarr|tym\w*)\b",
            lowered,
        )
    )
    german = len(
        re.findall(
            r"\b(?:nicht|mein\w*|gerät\w*|geraet\w*|konto\w*|passwort\w*|datei\w*|"
            r"rauch\w*|funken\w*|feuer|ausfall)\b",
            lowered,
        )
    )
    if albanian >= 2 or any(char in lowered for char in "ëç"):
        return "sq"
    if german >= 2 or any(char in lowered for char in "äöüß"):
        return "de"
    return "en"


def build_escalation_notice(risk_flags: list[str], language: str = "en") -> str:
    language = language if language in RISK_ACTIONS else "en"
    actions: list[str] = []
    for risk_flag in risk_flags:
        for action in RISK_ACTIONS[language].get(risk_flag, []):
            if action not in actions:
                actions.append(action)
    if not actions:
        return ""
    return f"{HEADINGS[language]}\n" + "\n".join(f"- {action}" for action in actions)


def apply_escalation_notice(response: str, risk_flags: list[str], language: str = "en") -> str:
    notice = build_escalation_notice(risk_flags, language)
    return f"{notice}\n\n{response}".strip() if notice else response
