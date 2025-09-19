#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import feedparser
import re
from difflib import SequenceMatcher
from telegram import Bot
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import html
import json
import hashlib
import contextlib
import math
from io import BytesIO
import aiohttp
from PIL import Image

# ================== CONFIG ==================
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_RU = "-1002597393191"

    # Длины
    MIN_LEN = 900
    MAX_LEN = 1000
    TG_HARD_LIMIT = 4000               # общий лимит Telegram
    TG_PHOTO_CAPTION_LIMIT = 1024      # лимит подписи к фото

    # LLM
    LLM_CONCURRENCY = 3
    LLM_INPUT_CHARS = 320
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMP = float(os.getenv("LLM_TEMP", "0.45"))  # чуть живее для лёгкого юмора

    # Кеш
    CACHE_FILE = "rewrite_cache.json"

    # Свежесть новостей
    LOCAL_TZ = ZoneInfo("Europe/Vienna")
    FRESHNESS_HOURS = int(os.getenv("FRESHNESS_HOURS", "19"))
    SOFT_DECAY_POWER = 1.1  # штраф возраста в скоринге

    # Изображение
    TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "750"))

    # Фиды
    CRYPTO_FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
    ]
    FINANCE_FEEDS = [
        "https://www.bloomberg.com/feed/podcast/etf-report.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.reuters.com/finance/rss",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "https://www.investing.com/rss/news_301.rss",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.morningbrew.com/feed.xml",
    ]

# ================== UTIL ==================
def html_escape(s: str) -> str:
    s = s or ""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s

def needs_link(text: str) -> bool:
    """Добавляем ссылку, если текст слишком общий/поверхностный"""
    if not re.search(
        r"(\d+|%|\$|₿|Ξ|bitcoin|btc|eth|google|fed|рынок|компания|usd|eur|фрс|ецб|биржа|рост|падени|ставк|инфляц)",
        text,
        re.IGNORECASE,
    ):
        return True
    return False

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def normalize_title(t: str) -> str:
    t = re.sub(r"<.*?>", "", t or "")
    t = re.sub(r"[\[\]\(\){}“”\"'«»•·\-–—:;,.!?]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def sha_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def best_entry_time(entry) -> Optional[datetime]:
    """
    Возвращает datetime (UTC) из published_parsed или updated_parsed.
    """
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if not st:
        return None
    try:
        dt = datetime(*st[:6], tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def should_add_wit(text: str) -> bool:
    """
    ~10% лёгкого юмора: детерминированно от содержимого.
    """
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return (int(h[:2], 16) % 10) == 0  # 1 из 10

# ================== CACHED STORAGE ==================
class DiskCache:
    def __init__(self, path: str):
        self.path = path
        self._data = {}
        self._load()

    def _load(self):
        with contextlib.suppress(Exception):
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self):
        with contextlib.suppress(Exception):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str):
        self._data[key] = value

# ================== NEWS PROCESSOR ==================
class NewsProcessor:
    @staticmethod
    def add_emoji_flair(text: str) -> str:
        emoji_map = {
            r"(?i)биткоин|bitcoin|btc": "₿",
            r"(?i)эфир|ethereum|eth": "Ξ",
            r"(?i)бык|bull": "🐂",
            r"(?i)медведь|bear": "🐻",
            r"(?i)рынок|market": "📈",
            r"(?i)фрс|fed|ецб|ecb": "🏦",
        }
        out = text
        for pattern, emoji in emoji_map.items():
            out = re.sub(pattern, f"{emoji} \\g<0>", out)
        return out

# ================== SCORING ==================
def score_news_item(item: Dict, all_news: List[Dict]) -> int:
    score = 0
    text = (item["title"] + " " + item["summary"]).lower()

    # ключевые слова
    if any(
        kw in text
        for kw in [
            "биткоин","bitcoin","btc","ethereum","eth","фрс","fed","treasury",
            "рынок","ставка","cpi","инфляция","центробанк","ecb","eps","выручка",
            "прибыль","gdp","pmi"
        ]
    ):
        score += 10

    # цифры/проценты/валюта
    if re.search(r"(\d+%|\$\d+|\d+\.\d+)", text):
        score += 5

    # источник
    source_url = item.get("source", "").lower()
    if any(src in source_url for src in ["bloomberg", "reuters", "cnbc", "coindesk", "cointelegraph"]):
        score += 7
    elif any(src in source_url for src in ["marketwatch", "investing"]):
        score += 5
    elif "morningbrew" in source_url:
        score += 2

    # свежесть (экспоненциальный штраф)
    if item.get("published_dt"):
        age_hours = (datetime.now(timezone.utc) - item["published_dt"]).total_seconds() / 3600
        score -= int(math.floor(age_hours ** Config.SOFT_DECAY_POWER))
    else:
        score -= 24  # без даты — далеко вниз

    # цитируемость (похожие заголовки)
    for other in all_news:
        if other is item:
            continue
        if similar(item["title_norm"], other["title_norm"]) > 0.7:
            score += 5
            break

    return score

# ================== FETCH NEWS ==================
async def fetch_news(feeds: List[str], max_news: int = 30) -> List[Dict]:
    entries = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:max_news]:
                published_dt = best_entry_time(entry)
                if published_dt is None:
                    continue
                age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
                if age_hours > Config.FRESHNESS_HOURS:
                    continue  # жёсткий отсев

                title = re.sub(r"<.*?>", "", (entry.get("title") or "").strip())
                summary = re.sub(r"<.*?>", "", (entry.get("summary") or "").strip())
                entries.append(
                    {
                        "title": title,
                        "title_norm": normalize_title(title),
                        "summary": summary,
                        "link": (entry.get("link") or "").strip(),
                        "published": entry.get("published_parsed"),
                        "published_dt": published_dt,  # aware UTC
                        "source": url,
                    }
                )
        except Exception as e:
            print(f"⚠️ Feed error ({url}): {str(e)[:100]}...")

    # дедуп по ссылке
    seen = set()
    uniq = []
    for e in entries:
        if e["link"] and e["link"] not in seen:
            seen.add(e["link"])
            uniq.append(e)

    # семантическая дедупликация заголовков
    filtered = []
    for e in uniq:
        if any(similar(e["title_norm"], x["title_norm"]) > 0.92 for x in filtered):
            continue
        filtered.append(e)

    return filtered

# ================== OPENAI (REWRITER) ==================
client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
_cache = DiskCache(Config.CACHE_FILE)
_sema = asyncio.Semaphore(Config.LLM_CONCURRENCY)

SYSTEM_MSG = (
    "Ты — финансовый редактор уровня Bloomberg/NYT: строго по фактам, чёткая терминология, "
    "дружественный человеческий тон без канцелярита. Русский язык. "
    "Смысл — первичен, цифры/проценты/суммы не терять."
)

def _build_prompt(text: str, allow_wit: bool) -> str:
    wit_rules = (
        "- Лёгкий, деликатный штрих иронии допустим, но только один короткий приём и только при уместности.\n"
        "- Если ирония неуместна — не добавляй её вовсе.\n"
    ) if allow_wit else "- Без какого-либо юмора или метафор.\n"

    return (
        "Перепиши новость для телеграм-дайджеста.\n"
        "- Формат: одно информативное предложение, ≈12–18 слов.\n"
        "- Стиль: Bloomberg/NYT — факты, ясно и сдержанно; человеческий, без воды.\n"
        f"{wit_rules}"
        "- Сохрани ключевые цифры/проценты/суммы/тикеры.\n"
        "- Избегай общих фраз и клише. Без эмодзи.\n\n"
        f"Оригинал: {text}"
    )

async def _rewrite_one(text: str) -> str:
    trimmed = text.strip()
    if len(trimmed) > Config.LLM_INPUT_CHARS:
        trimmed = trimmed[:Config.LLM_INPUT_CHARS].rsplit(" ", 1)[0] + "…"

    allow_wit = should_add_wit(trimmed)  # ~10% кейсов
    prompt = _build_prompt(trimmed, allow_wit)

    key = sha_key(Config.LLM_MODEL, SYSTEM_MSG, prompt)
    cached = _cache.get(key)
    if cached:
        return cached

    # быстрый фолбэк, если текст уже короткий с цифрами
    if len(trimmed.split()) <= 14 and re.search(r"[\d$%]", trimmed):
        _cache.set(key, trimmed)
        return trimmed

    async with _sema:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=Config.LLM_TEMP,
                    max_tokens=48,
                )
                out = (resp.choices[0].message.content or "").strip()
                _cache.set(key, out)
                return out
            except Exception:
                await asyncio.sleep(0.6 * (attempt + 1))

        _cache.set(key, trimmed)
        return trimmed

async def ai_rewrite_batch(bases: List[str]) -> List[str]:
    tasks = [asyncio.create_task(_rewrite_one(b)) for b in bases]
    results = await asyncio.gather(*tasks)
    _cache.save()
    return results

# ================== IMAGE (как в твоём втором скрипте) ==================
def _resize_image_height(buf: BytesIO, target_height: int) -> BytesIO:
    try:
        img = Image.open(buf)
        w, h = img.size
        if h <= target_height:
            buf.seek(0)
            return buf
        resized = img.resize((w, target_height), Image.Resampling.LANCZOS)
        out = BytesIO()
        resized.save(out, format="PNG")
        out.seek(0)
        return out
    except Exception:
        buf.seek(0)
        return buf

async def ai_generate_image(prompt: str) -> BytesIO:
    """
    DALL·E 3, современная плоская иллюстрация, без текста.
    Итог: высота сжата до TARGET_IMAGE_HEIGHT, ширина без изменений.
    """
    img_prompt = (
        f"Digital illustration for a finance/crypto daily digest: '{prompt}'. "
        "Fun but professional, modern flat style, soft colors, no text."
    )
    resp = await client.images.generate(
        model="dall-e-3",
        prompt=img_prompt,
        n=1,
        size="1024x1024",
    )
    img_url = resp.data[0].url
    async with aiohttp.ClientSession() as session:
        async with session.get(img_url) as r:
            r.raise_for_status()
            buf = BytesIO(await r.read())
    buf = _resize_image_height(buf, Config.TARGET_IMAGE_HEIGHT)
    return buf

def build_image_prompt_from_news(items: List[Dict]) -> str:
    titles = [re.sub(r"\s+", " ", i["title"]).strip() for i in items[:3] if i.get("title")]
    return " | ".join(titles) if titles else "Markets and crypto today"

# ================== BUILD CAPTION ==================
def enforce_fin_quota(selected: List[Dict], ranked: List[Dict], min_fin: int = 2) -> List[Dict]:
    fin = [n for n in ranked if n["source"] in Config.FINANCE_FEEDS]
    current_fin = [n for n in selected if n in fin]
    need = max(0, min_fin - len(current_fin))
    if need == 0:
        return selected

    to_add = []
    for n in ranked:
        if n in fin and n not in selected:
            to_add.append(n)
            if len(to_add) >= need:
                break

    # вытесняем крипто из конца
    for add in to_add:
        idx = None
        for i in range(len(selected) - 1, -1, -1):
            if selected[i]["source"] in Config.CRYPTO_FEEDS:
                idx = i
                break
        if idx is not None:
            selected[idx] = add
        else:
            selected.append(add)
    return selected

def assemble_caption(header: str, blocks: List[str], footer: str,
                     target_min: int, target_max: int) -> str:
    def join(h, bl, f):
        body = ("\n\n").join(bl)
        parts = [h.strip()]
        if body:
            parts += ["", body]
        if f:
            parts += ["", f]
        return "\n".join(parts)

    used = []
    for b in blocks:
        trial = join(header, used + [b], footer)
        if len(trial) <= target_max or len(trial) < target_min:
            used.append(b)
        else:
            break

    caption = join(header, used, footer)

    i = len(used)
    while len(caption) < target_min and i < len(blocks):
        trial = join(header, used + [blocks[i]], footer)
        if len(trial) <= target_max:
            used.append(blocks[i])
            caption = trial
            i += 1
        else:
            break

    if len(caption) < target_min and used:
        deficit = target_min - len(caption)
        used[-1] = used[-1] + (" " * min(deficit, 30))
        caption = join(header, used, footer)

    if len(caption) > target_max and used:
        base = join(header, used[:-1], footer)
        allowance = target_max - len(base) - 1
        if allowance > 20:
            used[-1] = used[-1][:allowance].rstrip() + "…"
            caption = join(header, used, footer)
        else:
            used = used[:-1]
            caption = join(header, used, footer)

    hard_cap = min(Config.TG_HARD_LIMIT, Config.TG_PHOTO_CAPTION_LIMIT)
    if len(caption) > hard_cap:
        caption = caption[: hard_cap - 1] + "…"

    print(f"📊 Итог: {len(caption)} символов (цель {target_min}-{target_max}, hard {hard_cap})")
    return caption

# ================== MAIN ==================
async def send_daily_digest():
    now_local = datetime.now(Config.LOCAL_TZ)
    print(f"🚀 Building 15:30 digest… Local time: {now_local:%Y-%m-%d %H:%M%z}")

    # 1) Пул новостей
    crypto_pool = await fetch_news(Config.CRYPTO_FEEDS, 30)
    finance_pool = await fetch_news(Config.FINANCE_FEEDS, 30)
    all_news = crypto_pool + finance_pool

    if not all_news:
        print("⚠️ No news found within freshness window.")
        return

    # 2) Скоринг + сортировка
    for item in all_news:
        item["score"] = score_news_item(item, all_news)
    all_news.sort(
        key=lambda x: (x["score"], x.get("published_dt") or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )

    # 3) Баланс 70/30 из топ-12 и минимум 2 финновости
    top = all_news[:12]
    crypto_top = [n for n in top if n["source"] in Config.CRYPTO_FEEDS]
    finance_top = [n for n in top if n["source"] in Config.FINANCE_FEEDS]

    total_target = min(12, len(top))
    crypto_target = int(total_target * 0.7)
    finance_target = total_target - crypto_target

    prelim = crypto_top[:crypto_target] + finance_top[:finance_target]
    selected = enforce_fin_quota(prelim, top, min_fin=2)

    # 4) Базы для перефразировки
    bases: List[str] = []
    for item in selected:
        first_sentence = (item["summary"] or "").split(".")[0]
        base = f"{item['title']}. {first_sentence}".strip()
        bases.append(base)

    # 5) Переписать LLM
    rewrites = await ai_rewrite_batch(bases)

    # 6) Сборка блоков
    blocks = []
    for item, rewritten in zip(selected, rewrites):
        pretty = NewsProcessor.add_emoji_flair(rewritten)  # эмодзи только здесь
        safe = html_escape(pretty)
        if needs_link(pretty):
            block = f"🏷️ {safe}\n<a href='{html.escape(item['link'], quote=True)}'>Источник</a>"
        else:
            block = f"🏷️ {safe}"
        blocks.append(block)

    # 7) Заголовок/подвал
    header = f"📊 Дневной дайджест — {now_local.strftime('%d.%m.%Y')}"
    footer = "С вами был ReserveOne ☕️"

    # 8) Подпись
    caption = assemble_caption(header, blocks, footer, target_min=Config.MIN_LEN, target_max=Config.MAX_LEN)
    print(f"🧾 Символов в подписи: {len(caption)}")

    # 9) Картинка (тот же стиль/размер)
    img_prompt = build_image_prompt_from_news(selected)
    image = None
    try:
        image = await ai_generate_image(img_prompt)
    except Exception as e:
        print(f"⚠️ Image generation failed: {e}")

    # 10) Отправка
    bot = Bot(token=Config.TELEGRAM_TOKEN)
    if image is not None:
        await bot.send_photo(
            chat_id=Config.TELEGRAM_CHANNEL_RU,
            photo=image,
            caption=caption[: Config.TG_PHOTO_CAPTION_LIMIT],
            parse_mode="HTML",
        )
    else:
        await bot.send_message(
            chat_id=Config.TELEGRAM_CHANNEL_RU,
            text=caption,
            parse_mode="HTML",
            disable_web_page_preview=False,
        )

    # 11) Статистика кеша (оценочно)
    hits = misses = 0
    for b in bases:
        trimmed = (b[:Config.LLM_INPUT_CHARS].rsplit(" ", 1)[0] + "…") if len(b) > Config.LLM_INPUT_CHARS else b
        key = sha_key(Config.LLM_MODEL, SYSTEM_MSG, _build_prompt(trimmed, should_add_wit(trimmed)))
        if _cache.get(key):
            hits += 1
        else:
            misses += 1
    print(f"✅ Digest sent. Cache ~ hits: {hits}, misses: {misses}")

if __name__ == "__main__":
    asyncio.run(send_daily_digest())
