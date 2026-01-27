#!/usr/bin/env python3
# parse_tiktok_live.py
# Sync έκδοση για TikTok Live -> M3U/JSON

# Μήνυμα έναρξης
print("=========================================================================")
print("🚀 Εκκίνηση του TikTok Parse Live! 🚀")
print("=========================================================================")

import os
import io
import sys
import json
import time
import requests
from typing import Dict, List
from datetime import datetime, timezone

from lxml import html
from PIL import Image

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
    
    import sys
import time

def print_banner_once():
    banner = r"""
████████╗██╗██╗  ██╗████████╗ ██████╗ ██╗  ██╗
╚══██╔══╝██║██║ ██╔╝╚══██╔══╝██╔═══██╗██║ ██╔╝
   ██║   ██║█████╔╝    ██║   ██║   ██║█████╔╝ 
   ██║   ██║██╔═██╗    ██║   ██║   ██║██╔═██╗ 
   ██║   ██║██║  ██╗   ██║   ╚██████╔╝██║  ██╗
   ╚═╝   ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
        TikTok Live Scraper Engine
-----------------------------------------------
"""

    # Typing animation
    for char in banner:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.0015)  # smooth, fast typing

    print("\033[0m")  # reset colors
    
# ---------- GLOBAL MERGE STREAM DATA ----------
def merge_stream_data(src, dest):
    """
    Ενώνει FLV/HLS/CMAF/DASH/HEVC streams από streamData & hevcStreamData
    μέσα στο dest (info["stream_urls"]).
    """
    if not src:
        return

    pull = src.get("pull_data", {})
    sraw = pull.get("stream_data")

    # TikTok sometimes returns string instead of dict
    if isinstance(sraw, str):
        try:
            sraw = json.loads(sraw)
        except Exception:
            return

    if not isinstance(sraw, dict):
        return

    data_obj = sraw.get("data", {}) or {}

    for q, c in data_obj.items():
        if q not in dest or not isinstance(dest[q], dict):
            dest[q] = {}

        # Case: string → treat as FLV
        if isinstance(c, str):
            dest[q]["flv"] = c
            continue

        # Case: list → take first valid URL
        if isinstance(c, list):
            for item in c:
                if isinstance(item, str):
                    dest[q]["flv"] = item
                    break
            continue

        if not isinstance(c, dict):
            continue

        main = c.get("main")

        # Case: main is string → treat as FLV
        if isinstance(main, str):
            dest[q]["flv"] = main
            continue

        if not isinstance(main, dict):
            continue

        flv = main.get("flv")
        hls = main.get("hls")
        hevc = main.get("hevc")

        if flv:
            dest[q]["flv"] = flv
        if hls:
            dest[q]["hls"] = hls
        if hevc:
            dest[q]["hevc"] = hevc

# ---------- Φόρτωση config ----------
def load_config(base_dir: str) -> dict:
    cfg_path = os.path.join(base_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    def abs_path(p):
        return os.path.join(base_dir, p) if not os.path.isabs(p) else p

    for k in (
        "users_file", "m3u_file", "json_file", "log_file",
        "avatars_dir", "avatar_cache_file", "debug_dir"
    ):
        if k in cfg:
            cfg[k] = abs_path(cfg[k])

    return cfg

# ---------- ANSI COLORS ----------
class ANSI:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


# ---------- Logger ----------
class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def write(self, msg: str):
        ts = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Detect type
        if msg.startswith("✅"):
            color = ANSI.GREEN
        elif msg.startswith("❌"):
            color = ANSI.RED
        elif msg.startswith("⚠"):
            color = ANSI.YELLOW
        elif msg.startswith("ℹ"):
            color = ANSI.BLUE
        else:
            color = ANSI.RESET

        line = f"[{ts}] {color}{msg}{ANSI.RESET}"
        print(line)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                # Save WITHOUT ANSI codes in file
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ---------- URL check ----------
def check_url_active(url: str, cfg: dict) -> bool:
    if not cfg.get("check_url_active", True):
        return True

    timeout = cfg.get("timeout", 10)

    try:
        h = requests.head(url, timeout=timeout, allow_redirects=True)
        if 200 <= h.status_code < 400:
            return True
    except Exception:
        pass

    try:
        g = requests.get(url, timeout=timeout, stream=True)
        if 200 <= g.status_code < 400:
            return True
    except Exception:
        pass

    return False


# ---------- Debug save ----------
def save_sigi_state(username: str, page_text: str, cfg: dict, log: Logger) -> None:
    if not cfg.get("debug_mode", False):
        return

    try:
        debug_dir = cfg.get("debug_dir", os.path.dirname(cfg["log_file"]))
        os.makedirs(debug_dir, exist_ok=True)

        sel = html.fromstring(page_text)
        sigi = sel.xpath("//script[@id='SIGI_STATE']/text()")
        if not sigi:
            out = os.path.join(debug_dir, f"{username}_page.html")
            with open(out, "w", encoding="utf-8") as f:
                f.write(page_text)
            log.write(f"[DEBUG] Saved HTML (no SIGI_STATE): {out}")
            return

        out = os.path.join(debug_dir, f"{username}_sigi.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(sigi[0])
        log.write(f"[DEBUG] Saved SIGI_STATE: {out}")

    except Exception as e:
        log.write(f"[DEBUG] Error saving SIGI_STATE: {e}")


# ---------- Parse SIGI_STATE ----------
def parse_live_info_html(text: str, tzinfo) -> Dict:
    try:
        sel = html.fromstring(text)
        sigi = sel.xpath("//script[@id='SIGI_STATE']/text()")
        if not sigi:
            sigi = sel.xpath('//script[contains(text(), "SIGI_STATE")]/text()')

        data = json.loads(sigi[0])

        live_room = data.get("LiveRoom", {}) or {}
        block = (
            live_room.get("liveRoomUserInfo", {})
            or live_room.get("liveRoomUserInfoV2", {})
        )

        room = block.get("liveRoom", {}) or {}
        user = block.get("user", {}) or {}

        st = room.get("startTime")
        status = room.get("status")

        status_map = {2: "Σε εξέλιξη", 4: "Δεν υπάρχει live"}
        status_str = status_map.get(status, "–")

        only_time = ""
        if st:
            dt = datetime.fromtimestamp(st, tz=tzinfo)
            only_time = dt.strftime("%H:%M:%S")

        info = {
            "uniqueId": user.get("uniqueId"),
            "nickname": user.get("nickname"),
            "avatar": user.get("avatarLarger") or user.get("avatarThumb"),
            "live_title": room.get("title") or "",
            "start_time": st,
            "start_time_only_time": only_time,
            "status": status,
            "status_str": status_str,
            "stream_urls": {}
        }
        playinfo = data.get("LiveRoom", {}).get("liveRoomUserInfo", {}).get("liveRoom", {}).get("playInfo", {})
        ls = playinfo.get("liveStream", {})
        urls = ls.get("playUrls", {})

        for q, c in urls.items():
            if q not in info["stream_urls"]:
                info["stream_urls"][q] = {}
            if isinstance(c, dict):
                if c.get("flv"):
                    info["stream_urls"][q]["flv"] = c["flv"]
                if c.get("hls"):
                    info["stream_urls"][q]["hls"] = c["hls"]
                if c.get("hevc"):
                    info["stream_urls"][q]["hevc"] = c["hevc"]

        merge_stream_data(room.get("streamData"), info["stream_urls"])
        merge_stream_data(room.get("hevcStreamData"), info["stream_urls"])

        return info

    except Exception:
        return {}


# ---------- OutputEngine (FULL MULTI-MODE) ----------
class OutputEngine:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log

        self.template = cfg.get(
            "m3u_extinf_template",
            "{uid} | {nick} - {title} | Ώρα έναρξης: {start_time} | {quality} | {protocol}"
        )

        self.fields = cfg.get(
            "m3u_extinf_fields",
            {
                "uid": True,
                "nick": True,
                "title": True,
                "start_time": True,
                "status": False,
                "quality": True,
                "protocol": True
            }
        )

        self.enable_extfilter = cfg.get("enable_extfilter", True)
        self.empty_header = cfg.get("save_empty_m3u_header", True)
        self.empty_message = cfg.get("empty_m3u_message", "#EXTINF:-1, Δεν βρέθηκαν live ή υπήρξε σφάλμα")

    # ---------- BUILD EXTINF ----------
    def build_extinf(self, info, quality, protocol, logo):
        def get_value(key, default=""):
            return info.get(key, default) or ""

        values = {
            "uid": get_value("uniqueId"),
            "nick": get_value("nickname"),
            "title": get_value("live_title"),
            "start_time": get_value("start_time_only_time"),
            "status": get_value("status_str"),
            "quality": quality or "",
            "protocol": protocol or ""
        }

        # Disable fields based on config
        for key, enabled in self.fields.items():
            if not enabled:
                values[key] = ""

        # Apply template
        text = self.template.format(**values)
        text = " ".join(text.split())

        # ExtFilter
        extfilter = ' $ExtFilter="Tikitok live"' if self.enable_extfilter else ""

        # Logo
        logo_attr = f' tvg-logo="{logo}"' if logo else ""

        return f'#EXTINF:-1 group-title="TikTok Live"{logo_attr} tvg-id="simpleTVFakeEpgId"{extfilter},{text}'

    # ---------- EMPTY OUTPUT ----------
    def write_empty(self, m3u_path):
        try:
            with open(m3u_path, "a", encoding="utf-8") as m3u:
                m3u.write(f"{self.empty_message}\n")
        except Exception as e:
            self.log.write(f"⚠ Σφάλμα εγγραφής empty message στο M3U: {e}")


# ---------- QualityEngine (FULL MULTI-MODE) ----------
class QualityEngine:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log

        # Config values
        self.quality = cfg.get("quality", "best")
        self.manual_quality = cfg.get("manual_quality", "")
        self.worst_rank = cfg.get("worst_quality_rank", {})
        self.ignore_ao = cfg.get("ignore_ao_in_video_modes", True)
        self.use_ao_only = cfg.get("use_ao_only_when_requested", True)
        self.allow_ao_fallback = cfg.get("allow_ao_fallback", False)
        self.fallback_lower = cfg.get("fallback_to_lower_quality", True)

        # Default ranking (if worst_quality_rank missing)
        self.default_rank = {
            "ld": 1, "sd": 2, "360p": 3, "480p": 4,
            "720p": 5, "hd": 6, "1080p": 7, "2k": 8, "4k": 9
        }

    # ---------- INTERNAL HELPERS ----------
    def _rank(self, q):
        return self.worst_rank.get(q, self.default_rank.get(q, 999))

    def _sort_best(self, qualities):
        order = ["4k", "2k", "1080p", "hd", "or4", "720p", "480p", "360p", "sd", "ld"]
        return [q for q in order if q in qualities]

    def _sort_worst(self, qualities):
        return sorted(qualities, key=lambda q: self._rank(q))

    # ---------- MAIN SELECTOR ----------
    def select(self, streams):
        if not isinstance(streams, dict) or not streams:
            return []

        qualities = list(streams.keys())

        # Remove AO if ignore_ao_in_video_modes = true
        if self.ignore_ao and "ao" in qualities:
            qualities.remove("ao")

        # Manual override
        if self.manual_quality:
            if self.manual_quality in qualities:
                return [self.manual_quality]
            elif self.fallback_lower:
                # fallback to lower qualities
                sorted_q = self._sort_best(qualities)
                if sorted_q:
                    return sorted_q
            return []

        # Worst video mode
        if self.quality == "worst_video":
            sorted_q = self._sort_worst(qualities)
            return sorted_q

        # Best mode (absolute best)
        if self.quality == "best":
            sorted_q = self._sort_best(qualities)
            return sorted_q

        # Specific quality
        if self.quality in qualities:
            return [self.quality]

        # Fallback to lower qualities
        if self.fallback_lower:
            sorted_q = self._sort_best(qualities)
            return sorted_q

        return []


# ---------- ProtocolEngine (FULL MULTI-MODE) ----------
class ProtocolEngine:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log

        self.protocol = cfg.get("protocol", "all")
        self.fallback_enabled = cfg.get("fallback_to_other_protocols", True)

        # Default fallback order
        self.order = ["flv", "hls", "hevc"]

    def select(self, urls):
        """
        urls = dict:
            {
                "flv": "...",
                "hls": "...",
                "hevc": "..."
            }
        """

        if not isinstance(urls, dict):
            return []

        # If protocol = all → try everything in default order
        if self.protocol == "all":
            return [(p, urls[p]) for p in self.order if urls.get(p)]

        # If specific protocol requested
        if self.protocol in urls and urls.get(self.protocol):
            return [(self.protocol, urls[self.protocol])]

        # If fallback disabled → return empty
        if not self.fallback_enabled:
            return []

        # Fallback to other protocols
        fallback_list = []
        for p in self.order:
            if urls.get(p):
                fallback_list.append((p, urls[p]))

        return fallback_list


# ---------- Scrape (sync) ----------
def scrape_users(users: List[str], cfg: dict, log: Logger, tzinfo) -> List[Dict]:
    results = []

    real_live_count = 0
    ghost_count = 0
    paused_count = 0
    region_locked_count = 0
    reconnecting_count = 0
    audio_only_count = 0

    total = len(users)
    timeout = cfg.get("timeout", 10)
    headers = {
        "User-Agent": cfg.get("user_agent", "Mozilla/5.0"),
        "Accept-Language": cfg.get("accept_language", "en-US,en;q=0.9")
    }

    for idx, username in enumerate(users, 1):
        try:
            url = f"https://www.tiktok.com/@{username}/live"
            r = requests.get(url, headers=headers, timeout=timeout)

            save_sigi_state(username, r.text, cfg, log)

            info = parse_live_info_html(r.text, tzinfo)
            uid = info.get("uniqueId") or username
            nick = info.get("nickname") or ""
            title = info.get("live_title") or ""
            status = info.get("status")
            status_str = info.get("status_str")
            only_time = info.get("start_time_only_time", "")

            room = info.get("room") or {}
            merge_stream_data(room.get("streamData"), info["stream_urls"])
            merge_stream_data(room.get("hevcStreamData"), info["stream_urls"])

            if status != 2 or not info.get("stream_urls"):
                try:
                    r2 = requests.get(url, headers=headers, timeout=timeout)
                    save_sigi_state(username, r2.text, cfg, log)
                    info2 = parse_live_info_html(r2.text, tzinfo)

                    room2 = info2.get("room") or {}
                    merge_stream_data(room2.get("streamData"), info2["stream_urls"])
                    merge_stream_data(room2.get("hevcStreamData"), info2["stream_urls"])

                    status2 = info2.get("status")
                    streams2 = info2.get("stream_urls")

                    if status2 == 2 and streams2:
                        log.write(f"[{idx}/{total}] ✅ 2nd pass: Το live του {uid} επιβεβαιώθηκε!")
                        info = info2
                        status = status2
                    else:
                        if status != 2:
                            log.write(f"[{idx}/{total}] ❌ {uid} | {nick} - {title} | {status_str}")
                        elif not info.get("stream_urls"):
                            log.write(f"[{idx}/{total}] ❌ {uid} | {nick} - {title} | Status=2 αλλά χωρίς streams")
                        continue

                except Exception as e:
                    log.write(f"[{idx}/{total}] ❌ 2nd pass error για {uid}: {e}")
                    continue

            streams = info.get("stream_urls", {})

            if not streams:
                ghost_count += 1
                log.write(f"[{idx}/{total}] ⚠ Ghost live: status=2 αλλά το TikTok δεν δίνει ΚΑΝΕΝΑ stream URL")
                continue

            has_video = False
            has_audio_only = False
            region_locked = False
            reconnecting = False

            for q, urls in streams.items():
                if not isinstance(urls, dict):
                    continue

                if q == "ao":
                    has_audio_only = True

                for proto, u in urls.items():
                    if not u:
                        continue

                    try:
                        h = requests.head(u, timeout=timeout, allow_redirects=True)
                        if h.status_code in (403, 451):
                            region_locked = True
                            continue
                    except:
                        pass

                    if check_url_active(u, cfg):
                        if q != "ao":
                            has_video = True
                        else:
                            has_audio_only = True
                    else:
                        reconnecting = True

            if region_locked:
                region_locked_count += 1
                log.write(f"[{idx}/{total}] ⚠ Region-locked live: URLs υπάρχουν αλλά η πρόσβαση είναι μπλοκαρισμένη (403/451)")
                continue

            if has_audio_only and not has_video:
                audio_only_count += 1
                log.write(f"[{idx}/{total}] ⚠ Audio-only live: Το live έχει μόνο ήχο χωρίς video stream")
                continue

            if reconnecting and not has_video:
                reconnecting_count += 1
                log.write(f"[{idx}/{total}] ⚠ Reconnecting live: Το live είναι status=2 αλλά το video pipeline δεν έχει σταθεροποιηθεί")
                continue

            if not has_video:
                paused_count += 1
                log.write(f"[{idx}/{total}] ⚠ Paused live: Το live είναι status=2 αλλά δεν υπάρχει ενεργό video stream")
                continue

            real_live_count += 1
            log.write(f"[{idx}/{total}] ✅ {uid} | {nick} - {title} | Ώρα έναρξης: {only_time} | Σε εξέλιξη")
            results.append(info)

        except Exception as e:
            log.write(f"[{idx}/{total}] ❌ {username} - σφάλμα αιτήματος/ανάλυσης: {e}")

    return results, {
        "real_live": real_live_count,
        "ghost": ghost_count,
        "paused": paused_count,
        "region_locked": region_locked_count,
        "reconnecting": reconnecting_count,
        "audio_only": audio_only_count
    }

# ---------- AvatarEngine (FULL MULTI-MODE) ----------
class AvatarEngine:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log

        self.mode = cfg.get("avatar_mode", "smart")
        self.refresh_days = cfg.get("avatar_refresh_days", 3)
        self.download_once = cfg.get("avatar_download_once", False)

        self.github_user = cfg.get("github_user", "")
        self.github_repo = cfg.get("github_repo", "")

        self.avatars_dir = cfg.get("avatars_dir", "avatars")
        self.cache_file = cfg.get("avatar_cache_file", "avatars_cache.json")

        os.makedirs(self.avatars_dir, exist_ok=True)

        # Load cache
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except:
            self.cache = {}

    # ---------- SAVE CACHE ----------
    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=4)
        except:
            pass

    # ---------- GITHUB URL ----------
    def github_url(self, uid):
        if not self.github_user or not self.github_repo:
            return None
        folder = os.path.basename(self.avatars_dir)
        return f"https://raw.githubusercontent.com/{self.github_user}/{self.github_repo}/main/{folder}/{uid}.jpg"

    # ---------- LOCAL PATH ----------
    def local_path(self, uid):
        return os.path.join(self.avatars_dir, f"{uid}.jpg")

    # ---------- SHOULD DOWNLOAD ----------
    def should_download(self, uid):
        if self.download_once:
            return not os.path.exists(self.local_path(uid))

        entry = self.cache.get(uid)
        if not entry:
            return True

        last = entry.get("last_download")
        if not last:
            return True

        try:
            last_dt = datetime.fromisoformat(last)
            if (datetime.now() - last_dt).days >= self.refresh_days:
                return True
        except:
            return True

        return False

    # ---------- DOWNLOAD AVATAR ----------
    def download(self, uid, url):
        if not url:
            return None

        if not self.should_download(uid):
            return self.local_path(uid)

        try:
            r = requests.get(url, timeout=self.cfg.get("timeout", 10))
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img = img.resize(
                    (self.cfg.get("avatar_resize", 256),
                     self.cfg.get("avatar_resize", 256)),
                    Image.LANCZOS
                )
                img.save(self.local_path(uid), "JPEG", quality=90)

                self.cache[uid] = {"last_download": datetime.now().isoformat()}
                self.save_cache()
                self.cfg["__summary_avatars"] = self.cfg.get("__summary_avatars", 0) + 1

                return self.local_path(uid)
        except Exception as e:
            self.log.write(f"[{uid}] ⚠ Avatar download error: {e}")

        return None

    # ---------- MAIN SELECTOR ----------
    def select(self, uid, avatar_url):
        # 1. Mode none
        if self.mode == "none":
            return ""

        # 2. Mode favicon
        if self.mode == "favicon":
            return "https://www.tiktok.com/favicon.ico"

        # 3. Mode remote (Το προσωρινό link του TikTok)
        if self.mode == "remote":
            return avatar_url or "https://www.tiktok.com/favicon.ico"

        # 4. Mode local / smart
        if self.mode in ["local", "smart"]:
            # Κατεβάζει τη φωτό τοπικά στον φάκελο avatars/
            success = self.download(uid, avatar_url)
            if success:
                # Επιστρέφει το link από το repo parse_live (εκεί που είναι οι φωτό)
                return f"https://raw.githubusercontent.com/Blueddo/parse_live/main/avatars/{uid}.jpg"
            
            return avatar_url or "https://www.tiktok.com/favicon.ico"

        # Default fallback
        return "https://www.tiktok.com/favicon.ico"

# ---------- Γράψιμο M3U & JSON ----------
def write_outputs(live_infos: List[Dict], cfg: dict, log: Logger) -> None:
    # JSON metadata
    try:
        with open(cfg["json_file"], "w", encoding="utf-8") as jf:
            json.dump(live_infos, jf, ensure_ascii=False, indent=2)
        log.write(f"Αποθηκεύτηκαν metadata: {cfg['json_file']}")
    except Exception as e:
        log.write(f"⚠ Σφάλμα αποθήκευσης JSON: {e}")

    # Header M3U
    try:
        with open(cfg["m3u_file"], "w", encoding="utf-8") as m3u:
            m3u.write('#EXTM3U $BorpasFileFormat="1" $NestedGroupsSeparator="/"\n')
    except Exception as e:
        log.write(f"⚠ Αδυναμία δημιουργίας M3U: {e}")
        return

    written = 0
    qe = QualityEngine(cfg, log)

    try:
        with open(cfg["m3u_file"], "a", encoding="utf-8") as m3u:
            for info in live_infos:
                uid = info.get("uniqueId") or ""
                avatar_url = info.get("avatar") or ""
                streams = info.get("stream_urls") or {}

                if not isinstance(streams, dict) or not streams:
                    continue

                # Avatar
                ae = AvatarEngine(cfg, log)
                logo = ae.select(uid, avatar_url)

                # QUALITY selection via QualityEngine (FULL CONFIG CONTROL)
                engine_qualities = qe.select(streams)

                if not engine_qualities:
                    log.write(
                        f"[{uid}] ⚠ Καμία διαθέσιμη ποιότητα από QualityEngine "
                        f"(streams κατεστραμμένα ή config επιλογές)"
                    )
                    continue

                available_qualities = engine_qualities

                chosen = False

                for quality in available_qualities:
                    urls = streams.get(quality)

                    if not urls or not isinstance(urls, dict):
                        log.write(
                            f"[{uid}] ⚠ Παράκαμψη quality={quality} επειδή urls δεν είναι dict "
                            f"({type(urls).__name__})"
                        )
                        continue

                    # PROTOCOL selection via ProtocolEngine
                    pe = ProtocolEngine(cfg, log)
                    candidates = pe.select(urls)

                    if not candidates:
                        log.write(f"[{uid}] ⚠ Καμία διεύθυνση για quality={quality} (ProtocolEngine άδειο)")
                        continue

                    for protocol, url in candidates:
                        if not url or not isinstance(url, str):
                            continue
                        if not check_url_active(url, cfg):
                            continue

                        oe = OutputEngine(cfg, log)
                        extinf = oe.build_extinf(info, quality, protocol, logo)
                        m3u.write(extinf + "\n")
                        m3u.write(f"{url}\n")
                        written += 1
                        cfg["__summary_entries"] = cfg.get("__summary_entries", 0) + 1
                        chosen = True
                        break

                    if chosen:
                        break

    except Exception as e:
        log.write(f"⚠ Σφάλμα εγγραφής M3U: {e}")
        return

    if written == 0:
        oe = OutputEngine(cfg, log)
        oe.write_empty(cfg["m3u_file"])
        log.write("ℹ Το M3U δημιουργήθηκε χωρίς ενεργές εγγραφές (written=0).")


# ---------- Auto-clean avatars ----------
def clean_orphan_avatars(cfg: dict, users: List[str], log: Logger) -> None:
    avatars_dir = cfg.get("avatars_dir")
    if not avatars_dir or not os.path.isdir(avatars_dir):
        return

    user_set = set(u.strip().lower() for u in users if u.strip())

    removed = 0
    try:
        for name in os.listdir(avatars_dir):
            path = os.path.join(avatars_dir, name)
            if not os.path.isfile(path):
                continue
            if not name.lower().endswith(".jpg"):
                continue

            uid = os.path.splitext(name)[0].lower()
            if uid not in user_set:
                try:
                    os.remove(path)
                    removed += 1
                except Exception as e:
                    log.write(f"⚠ Σφάλμα διαγραφής avatar {name}: {e}")

        if removed > 0:
            log.write(f"🧹 Διαγράφηκαν {removed} orphan avatars που δεν υπάρχουν πια στο userstiktok.txt")

    except Exception as e:
        log.write(f"⚠ Σφάλμα κατά τον καθαρισμό orphan avatars: {e}")


# ---------- MainEngine (FULL MULTI-MODE) ----------
class MainEngine:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log

        self.run_mode = cfg.get("run_mode", "single")
        self.interval = cfg.get("interval", 60)
        self.timezone_name = cfg.get("timezone", "Europe/Athens")
        self.locale = cfg.get("locale", "el")
        self.log_level = cfg.get("log_level", "info")

    # ---------- TIMEZONE ----------
    def get_timezone(self):
        try:
            if ZoneInfo:
                return ZoneInfo(self.timezone_name)
        except:
            pass
        return timezone.utc

    # ---------- RUN SINGLE ----------
    def run_single(self, users, cfg):
        tzinfo = self.get_timezone()
        start_time = time.time()

        infos, stats = scrape_users(users, cfg, self.log, tzinfo)

        write_outputs(infos, cfg, self.log)
        clean_orphan_avatars(cfg, users, self.log)

        duration = time.time() - start_time

        self.log.write("═══════════════════════════════════════════════════════")
        self.log.write("📊 Στατιστικά κύκλου:")
        self.log.write(f"    • Πραγματικά live: {stats['real_live']}")
        self.log.write(f"    • Ghost: {stats['ghost']}")
        self.log.write(f"    • Paused: {stats['paused']}")
        self.log.write(f"    • Region-locked: {stats['region_locked']}")
        self.log.write(f"    • Reconnecting: {stats['reconnecting']}")
        self.log.write(f"    • Audio-only: {stats['audio_only']}")
        self.log.write(f"    • Εγγραφές M3U: {cfg.get('__summary_entries', 0)}")
        self.log.write(f"    • Avatars ενημερώθηκαν: {cfg.get('__summary_avatars', 0)}")
        self.log.write(f"    • Διάρκεια scraping: {duration:.1f} sec")
        self.log.write("═══════════════════════════════════════════════════════")

    # ---------- RUN LOOP ----------
    def run_loop(self, users, cfg):
        tzinfo = self.get_timezone()
        self.log.write("Το πρόγραμμα τρέχει μέχρι να το κλείσεις (Ctrl+C).\n")

        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # ANSI colors
        C1 = "\033[96m"
        C2 = "\033[93m"
        C3 = "\033[95m"
        C4 = "\033[92m"
        C5 = "\033[94m"
        RESET = "\033[0m"

        cycle = 0

        while True:
            # --- 1) Scrape cycle ---
            cycle += 1
            cycle_start = time.time()
            total_users = len(users)

            self.log.write(f"▶ Νέος κύκλος scraping #{cycle} ξεκίνησε ({total_users} χρήστες)")

            # Reset counters
            cfg["__summary_entries"] = 0
            cfg["__summary_avatars"] = 0

            infos, stats = scrape_users(users, cfg, self.log, tzinfo)
            write_outputs(infos, cfg, self.log)
            clean_orphan_avatars(cfg, users, self.log)

            cycle_end = time.time()
            cycle_duration = round(cycle_end - cycle_start, 2)

            # --- 2) Stats block ---
            print(C4)
            print("    ═══════════════════════════════════════════════════════")
            print(f"    📊 Στατιστικά κύκλου #{cycle}:")
            print(f"        • Βρεθηκαν live: {ANSI.GREEN}{stats['real_live']}{ANSI.RESET}")
            print(f"        • Ghost: {ANSI.YELLOW}{stats['ghost']}{ANSI.RESET}")
            print(f"        • Paused: {ANSI.YELLOW}{stats['paused']}{ANSI.RESET}")
            print(f"        • Region-locked: {ANSI.RED}{stats['region_locked']}{ANSI.RESET}")
            print(f"        • Reconnecting: {ANSI.YELLOW}{stats['reconnecting']}{ANSI.RESET}")
            print(f"        • Audio-only: {ANSI.BLUE}{stats['audio_only']}{ANSI.RESET}")
            print(f"        • Εγγραφές M3U: {ANSI.GREEN}{cfg.get('__summary_entries', 0)}{ANSI.RESET}")
            print(f"        • Avatars ενημερώθηκαν: {ANSI.GREEN}{cfg.get('__summary_avatars', 0)}{ANSI.RESET}")
            print(f"        • Διάρκεια scraping: {ANSI.GREEN}{cycle_duration} sec{ANSI.RESET}")
            print("    ═══════════════════════════════════════════════════════")
            print(RESET)

            # --- 3) Countdown with spinner + progress bar ---
            total = self.interval

            try:
                term_width = os.get_terminal_size().columns
            except:
                term_width = 80

            bar_max = max(10, term_width - 45)

            # Μήνυμα οδηγιών
            print(f"{C5}Πατήστε {C2}r{C5} για επανεκκίνηση χρονομέτρου, "
                  f"{C2}g{C5} για άμεσο νέο κύκλο.{RESET}")

            remaining = total

            while True:
                reset_timer = False
                force_cycle = False

                # --- Keyboard instant actions ---
                if os.name == "nt":
                    import msvcrt
                    if msvcrt.kbhit():
                        raw = msvcrt.getch()
                        try:
                            key = raw.decode(errors="ignore").lower()
                        except:
                            key = chr(raw[0]).lower()

                        if key == "r":
                            print("\n🔁 Επανεκκίνηση χρονομέτρου!")
                            reset_timer = True

                        elif key == "g":
                            print("\n⚡ Άμεσος νέος κύκλος!")
                            force_cycle = True

                else:
                    import sys, select
                    dr, dw, de = select.select([sys.stdin], [], [], 0)
                    if dr:
                        key = sys.stdin.read(1).lower()

                        if key == "r":
                            print("\n🔁 Επανεκκίνηση χρονομέτρου!")
                            reset_timer = True

                        elif key == "g":
                            print("\n⚡ Άμεσος νέος κύκλος!")
                            force_cycle = True

                # --- r → reset timer (ΔΕΝ κάνει scraping) ---
                if reset_timer:
                    remaining = total
                    continue

                # --- g → άμεσο scraping ---
                if force_cycle:
                    break

                # --- countdown finished → νέο scraping ---
                if remaining <= 0:
                    break

                # --- Spinner + progress bar ---
                mins = remaining // 60
                secs = remaining % 60

                spin = spinner_frames[remaining % len(spinner_frames)]

                filled = int((total - remaining) / total * bar_max)
                bar = "█" * filled + "░" * (bar_max - filled)

                line = (
                    f"{C1}{spin}{RESET} "
                    f"{C2}Επόμενος έλεγχος σε {mins:02d}:{secs:02d}{RESET} "
                    f"{C3}|{bar}|{RESET}"
                )

                print(line.ljust(term_width), end="\r", flush=True)
                time.sleep(1)
                remaining -= 1

            print(" " * term_width, end="\r")

    # ---------- MAIN DISPATCH ----------
    def start(self, users, cfg):
        self.log.write(
            f"▶ Εκκίνηση sync: users={cfg['users_file']} "
            f"mode={self.run_mode} interval={self.interval}s "
            f"quality='{cfg.get('quality', '')}' "
            f"protocol={cfg.get('protocol', 'all')} "
            f"avatar_mode={cfg.get('avatar_mode', 'local')} "
            f"debug={cfg.get('debug_mode', False)}"
        )

        if self.run_mode == "single":
            self.run_single(users, cfg)
        else:
            self.run_loop(users, cfg)

# ---------- Main ----------
def main():
    print_banner_once() # <<< ΕΜΦΑΝΙΖΕΤΑΙ ΜΟΝΟ ΜΙΑ ΦΟΡΑ
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(base_dir)

    # Ensure required directories exist
    for p in ("m3u_file", "json_file", "log_file"):
        os.makedirs(os.path.dirname(cfg[p]), exist_ok=True)

    os.makedirs(cfg.get("avatars_dir", os.path.join(base_dir, "avatars")), exist_ok=True)

    if cfg.get("debug_mode", False):
        os.makedirs(cfg.get("debug_dir", os.path.dirname(cfg["log_file"])), exist_ok=True)

    log = Logger(cfg["log_file"])

    # Load users
    try:
        with open(cfg["users_file"], "r", encoding="utf-8") as f:
            users = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        log.write(f"❌ Δεν βρέθηκε το αρχείο χρηστών: {cfg['users_file']}")
        return

    # Start MainEngine
    engine = MainEngine(cfg, log)
    engine.start(users, cfg)


if __name__ == "__main__":
    main()


