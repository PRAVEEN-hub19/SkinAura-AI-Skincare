"""SkinAura Flask web application.

Run with: python app.py
"""

import base64
import hmac
import io
import json
import os
import secrets
import sqlite3
import textwrap
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import date, timedelta
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from groq import Groq
except ImportError:
    Groq = None

from inference import get_model_warning, predict_skin_type_details
from recommend import (
    DISCLAIMER,
    PROFILE_OPTIONS,
    RECOMMENDATIONS,
    WEATHER_CONDITIONS,
    get_recommendation,
    get_weather_advice,
    resolve_profile,
)

DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(DOTENV_PATH)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SKINAURA_SECRET") or "skinaura-production-stable-secret-2026"
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["DATABASE"] = os.path.join(app.instance_path, "skinaura.sqlite3")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
IMAGE_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
}
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "").strip() or "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = 1800
CHAT_MAX_MESSAGE_LENGTH = 3000
CHAT_MAX_HISTORY_ITEM_LENGTH = 10000
CHAT_MAX_HISTORY_MESSAGES = 10


def is_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@contextmanager
def database():
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    connection = sqlite3.connect(app.config["DATABASE"])
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database():
    with database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                skin_profile TEXT NOT NULL DEFAULT 'image',
                skin_goal TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS product_checks (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                product_key TEXT NOT NULL,
                checked INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, product_key)
            );
            CREATE TABLE IF NOT EXISTS analysis_results (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                result_json TEXT NOT NULL,
                weather_condition TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS routine_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                task_name TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_date TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS routine_task_completions (
                task_id INTEGER NOT NULL REFERENCES routine_tasks(id) ON DELETE CASCADE,
                completion_date TEXT NOT NULL,
                PRIMARY KEY (task_id, completion_date)
            );
            CREATE TABLE IF NOT EXISTS routine_days (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                completion_date TEXT NOT NULL,
                PRIMARY KEY (user_id, completion_date)
            );
            """
        )
        user_columns = {
            column["name"] for column in connection.execute("PRAGMA table_info(users)")
        }
        for column_name, definition in (
            ("display_name", "TEXT NOT NULL DEFAULT ''"),
            ("skin_profile", "TEXT NOT NULL DEFAULT 'image'"),
            ("skin_goal", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column_name not in user_columns:
                connection.execute(f"ALTER TABLE users ADD COLUMN {column_name} {definition}")


def current_user_id():
    return session.get("user_id")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user_id() is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def require_csrf():
    submitted_token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    session_token = session.get("csrf_token", "")
    if not submitted_token or not session_token or not hmac.compare_digest(submitted_token, session_token):
        abort(400, description="The form expired or could not be verified. Please refresh and try again.")


@app.context_processor
def inject_template_values():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["csrf_token"]}


def validated_image(raw: bytes, expected_format: str | None = None):
    try:
        with Image.open(io.BytesIO(raw)) as candidate:
            image_format = candidate.format
            width, height = candidate.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image that is smaller than 20 megapixels.")
            if expected_format and image_format != expected_format:
                raise ValueError("The file extension does not match the image content.")
            candidate.verify()
        with Image.open(io.BytesIO(raw)) as decoded:
            return decoded.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The image is invalid or cannot be opened. Choose a JPG, PNG, WEBP, or BMP image.") from error


def routine_summary(user_id: int, completion_date: date | None = None):
    today = completion_date or date.today()
    with database() as connection:
        tasks = connection.execute(
            """
            SELECT task.id, task.task_name,
                   EXISTS (
                       SELECT 1 FROM routine_task_completions AS completion
                       WHERE completion.task_id = task.id AND completion.completion_date = ?
                   ) AS completed
            FROM routine_tasks AS task
            WHERE task.user_id = ? AND task.active = 1
            ORDER BY task.id
            """,
            (today.isoformat(), user_id),
        ).fetchall()
        week_dates = {
            row["completion_date"]
            for row in connection.execute(
                """
                SELECT completion_date FROM routine_days
                WHERE user_id = ? AND completion_date >= ? AND completion_date <= ?
                ORDER BY completion_date DESC
                """,
                (user_id, (today - timedelta(days=6)).isoformat(), today.isoformat()),
            )
        }
        all_completed_dates = {
            row["completion_date"]
            for row in connection.execute(
                "SELECT completion_date FROM routine_days WHERE user_id = ?",
                (user_id,),
            )
        }

    streak = 0
    streak_day = today if today.isoformat() in all_completed_dates else today - timedelta(days=1)
    while streak_day.isoformat() in all_completed_dates:
        streak += 1
        streak_day -= timedelta(days=1)
    last_date = max(all_completed_dates, default="")
    completed_count = sum(bool(task["completed"]) for task in tasks)
    total_count = len(tasks)
    return {
        "tasks": tasks,
        "week_days": [
            {
                "date": (today - timedelta(days=day_offset)).isoformat(),
                "completed": (today - timedelta(days=day_offset)).isoformat() in week_dates,
            }
            for day_offset in range(6, -1, -1)
        ],
        "streak_count": streak,
        "last_date": last_date,
        "completed_count": completed_count,
        "total_count": total_count,
        "today_percent": round(completed_count * 100 / total_count) if total_count else 0,
        "week_count": len(week_dates),
        "week_percent": round(len(week_dates) * 100 / 7),
        "today": today.isoformat(),
    }


def refresh_routine_day(connection, user_id: int, completion_date: str):
    totals = connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(EXISTS (
                   SELECT 1 FROM routine_task_completions AS completion
                   WHERE completion.task_id = task.id
                     AND completion.completion_date = ?
               )) AS completed
        FROM routine_tasks AS task
        WHERE task.user_id = ? AND task.active = 1
          AND task.created_date <= ?
        """,
        (completion_date, user_id, completion_date),
    ).fetchone()
    if totals["total"] and totals["completed"] == totals["total"]:
        connection.execute(
            "INSERT OR IGNORE INTO routine_days (user_id, completion_date) VALUES (?, ?)",
            (user_id, completion_date),
        )
    else:
        connection.execute(
            "DELETE FROM routine_days WHERE user_id = ? AND completion_date = ?",
            (user_id, completion_date),
        )


def recommendation_context(user_id: int):
    with database() as connection:
        user = connection.execute(
            "SELECT skin_profile FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        analysis = connection.execute(
            "SELECT result_json, weather_condition FROM analysis_results WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    stored_profile = user["skin_profile"] if user else "image"
    if stored_profile != "image":
        profile = stored_profile
    elif analysis:
        profile = json.loads(analysis["result_json"])["profile_key"]
    else:
        profile = "normal"
    weather = analysis["weather_condition"] if analysis else "mild"
    recommendation = get_recommendation(profile, weather)
    return profile, weather, recommendation


SUGGESTED_PROMPTS = {
    "oily": [
        "Suggest an oil-control morning skincare routine",
        "Salicylic acid vs Niacinamide: which is better for pores?",
        "How to prevent midday shine without dehydrating skin?",
        "What lightweight sunscreen won't clog my pores?",
    ],
    "dry": [
        "How to repair a compromised skin moisture barrier?",
        "Hyaluronic acid vs Ceramides: what do I need?",
        "How to adjust my routine for dry or cold weather?",
        "What cleanser won't make my face feel tight or flaky?",
    ],
    "combination": [
        "How to treat an oily T-zone while hydrating dry cheeks?",
        "What type of moisturizer works best for combination skin?",
        "How often should I exfoliate combination skin?",
        "How does changing weather affect combination skin?",
    ],
    "sensitive": [
        "How do I patch-test new products safely?",
        "Which common ingredients trigger redness and stinging?",
        "What is a minimal 3-step routine for reactive skin?",
        "How can I soothe irritated or burning skin?",
    ],
    "normal": [
        "How do I maintain my skin barrier and natural balance?",
        "Daily SPF protection and reapplication essentials",
        "Should I incorporate antioxidants like Vitamin C?",
        "How to keep my skin healthy through seasonal changes?",
    ],
}


def assistant_context(user_id: int) -> dict:
    with database() as connection:
        user = connection.execute(
            "SELECT display_name, skin_profile, skin_goal FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        analysis = connection.execute(
            "SELECT result_json, weather_condition FROM analysis_results WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        tasks = [
            row["task_name"]
            for row in connection.execute(
                "SELECT task_name FROM routine_tasks WHERE user_id = ? AND active = 1 ORDER BY id ASC",
                (user_id,),
            )
        ]
    stored_profile = user["skin_profile"] if user else "image"
    analysis_data = json.loads(analysis["result_json"]) if analysis else None

    if stored_profile != "image":
        profile = stored_profile
        profile_origin = "Self-assessed profile"
    elif analysis_data:
        profile = analysis_data.get("profile_key", "normal")
        conf = analysis_data.get("confidence", 0)
        profile_origin = f"Photo scan estimate ({round(conf, 1)}% match)"
    else:
        profile = "normal"
        profile_origin = "Default profile"

    weather = analysis["weather_condition"] if analysis else "mild"
    weather_label = WEATHER_CONDITIONS.get(weather, weather.title())
    recommendation = get_recommendation(profile, weather)

    return {
        "user_name": user["display_name"] if user and user["display_name"] else "",
        "skin_goal": user["skin_goal"] if user and user["skin_goal"] else "",
        "profile_key": profile,
        "profile_origin": profile_origin,
        "weather_key": weather,
        "weather_label": weather_label,
        "recommendation": recommendation,
        "analysis_data": analysis_data,
        "tasks": tasks,
    }


def groq_chat_completion(api_key: str, messages: list[dict[str, str]]) -> str:
    # First attempt: Official Groq Python SDK if installed
    if Groq is not None:
        try:
            client = Groq(api_key=api_key, timeout=35)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=GROQ_MAX_TOKENS,
            )
            answer = completion.choices[0].message.content
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        except Exception as sdk_err:
            app.logger.info("Groq SDK call failed or returned empty, trying fallback: %s", sdk_err)

    # HTTP Fallback with User-Agent header (required to prevent Cloudflare 403 blocks)
    payload = json.dumps(
        {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.6,
            "max_completion_tokens": GROQ_MAX_TOKENS,
        }
    ).encode("utf-8")
    api_request = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SkinAura-Assistant/1.0 (Windows; Python-urllib)",
        },
        method="POST",
    )
    with urllib.request.urlopen(api_request, timeout=35) as response:
        result = json.loads(response.read().decode("utf-8"))
    answer = result["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Groq returned an empty chat response.")
    return answer.strip()


def groq_error_detail(error: urllib.error.HTTPError, api_key: str) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace")
        response_data = json.loads(body)
        detail = response_data.get("error", {}).get("message", "")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        detail = ""
    if not isinstance(detail, str):
        detail = ""
    detail = detail.replace(api_key, "[redacted]").strip()
    return detail[:500]


def make_pdf(lines: list[str]) -> bytes:
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=82) or [""])
    page_lines = [wrapped_lines[index:index + 43] for index in range(0, len(wrapped_lines), 43)] or [[]]
    objects: list[bytes | None] = [
        None,
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_object_ids = []
    for page_number, page in enumerate(page_lines):
        page_id = 4 + page_number * 2
        content_id = page_id + 1
        page_object_ids.append(f"{page_id} 0 R")
        commands = ["BT /F1 16 Tf 50 790 Td (SkinAura Skin Analysis Report) Tj ET"]
        y_position = 758
        for line in page:
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            escaped = safe_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F1 10 Tf 50 {y_position} Td ({escaped}) Tj ET")
            y_position -= 16
        content = "\n".join(commands).encode("ascii")
        objects.extend([
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii"),
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
        ])
    objects[2] = (
        f"<< /Type /Pages /Kids [{' '.join(page_object_ids)}] /Count {len(page_lines)} >>"
    ).encode("ascii")
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, content in enumerate(objects[1:], start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(content or b"")
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user_id() is not None:
        return redirect(url_for("index"))
    if request.method == "POST":
        require_csrf()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(email) > 254 or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash("Enter a valid email address.", "error")
        elif len(password) < 8 or len(password) > 128:
            flash("Choose a password between 8 and 128 characters.", "error")
        else:
            try:
                with database() as connection:
                    connection.execute(
                        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                        (email, generate_password_hash(password)),
                    )
            except sqlite3.IntegrityError:
                flash("An account with that email already exists. Please log in.", "error")
            else:
                flash("Your account is ready. Log in to continue.", "success")
                return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user_id() is not None:
        return redirect(url_for("index"))
    if request.method == "POST":
        require_csrf()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with database() as connection:
            user = connection.execute(
                "SELECT id, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Email or password is incorrect.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("index"))
    return render_template("login.html")


@app.post("/logout")
@login_required
def logout():
    require_csrf()
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
    user_id = current_user_id()
    routine = routine_summary(user_id)
    with database() as connection:
        user = connection.execute(
            "SELECT skin_profile FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return render_template(
        "index.html",
        selected_weather="mild",
        selected_profile=user["skin_profile"],
        profile_options=PROFILE_OPTIONS,
        disclaimer=DISCLAIMER,
        routine=routine,
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = current_user_id()
    with database() as connection:
        user = connection.execute(
            "SELECT email, display_name, skin_profile, skin_goal FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if request.method == "POST":
        require_csrf()
        display_name = request.form.get("display_name", "").strip()
        skin_profile = request.form.get("skin_profile", "")
        skin_goal = request.form.get("skin_goal", "").strip()
        if len(display_name) > 60:
            flash("Keep your name to 60 characters or fewer.", "error")
        elif len(skin_goal) > 180:
            flash("Keep your skin goal to 180 characters or fewer.", "error")
        elif skin_profile not in {"image", *RECOMMENDATIONS}:
            flash("Choose a supported skin profile.", "error")
        else:
            with database() as connection:
                connection.execute(
                    """
                    UPDATE users
                    SET display_name = ?, skin_profile = ?, skin_goal = ?
                    WHERE id = ?
                    """,
                    (display_name, skin_profile, skin_goal, user_id),
                )
            flash("Your profile has been updated.", "success")
            return redirect(url_for("profile"))
    return render_template(
        "profile.html",
        user=user,
        profile_choices={
            **PROFILE_OPTIONS,
            "oily": "Oily",
            "dry": "Dry",
            "normal": "Normal",
        },
    )


@app.get("/recommendations")
@login_required
def recommendations():
    profile_key, weather, recommendation = recommendation_context(current_user_id())
    return render_template(
        "recommendations.html",
        profile_key=profile_key,
        weather=weather,
        weather_label=WEATHER_CONDITIONS[weather],
        recommendation=recommendation,
        weather_advice=get_weather_advice(weather, profile_key),
        disclaimer=DISCLAIMER,
    )


@app.get("/chat")
@login_required
def skin_chat():
    load_dotenv(DOTENV_PATH, override=True)
    ctx = assistant_context(current_user_id())
    products = [
        {"name": ctx["recommendation"][f"product_{slot}"], "url": ctx["recommendation"][f"product_{slot}_url"]}
        for slot in ("a", "b")
    ]
    prompts = SUGGESTED_PROMPTS.get(ctx["profile_key"], SUGGESTED_PROMPTS["normal"])
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return render_template(
        "chat.html",
        context=ctx,
        profile_key=ctx["profile_key"],
        products=products,
        suggested_prompts=prompts,
        disclaimer=DISCLAIMER,
        groq_configured=bool(api_key),
    )


@app.post("/chat/api")
@login_required
def skin_chat_api():
    require_csrf()
    load_dotenv(DOTENV_PATH, override=True)
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        return jsonify({"error": "SkinAura chat is not configured yet. Add GROQ_API_KEY to the server environment (.env file)."}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a valid chat message."}), 400
    question = payload.get("message")
    history = payload.get("history", [])
    if not isinstance(question, str) or not question.strip() or len(question) > CHAT_MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Enter a question of 1 to {CHAT_MAX_MESSAGE_LENGTH} characters."}), 400
    if not isinstance(history, list) or len(history) > CHAT_MAX_HISTORY_MESSAGES:
        return jsonify({"error": "This conversation is too long. Start a new chat and try again."}), 400

    cleaned_history = []
    for item in history:
        if (
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
            or not item["content"].strip()
            or len(item["content"]) > CHAT_MAX_HISTORY_ITEM_LENGTH
        ):
            return jsonify({"error": "The conversation contains an invalid message. Start a new chat and try again."}), 400
        cleaned_history.append({"role": item["role"], "content": item["content"].strip()})

    ctx = assistant_context(current_user_id())
    rec = ctx["recommendation"]

    context_details = [
        f"Active skin profile: {ctx['profile_key'].upper()} ({ctx['profile_origin']}).",
        f"Local weather context: {ctx['weather_label']}.",
        f"Recommended weather advice: {get_weather_advice(ctx['weather_key'], ctx['profile_key'])}.",
    ]
    if ctx["user_name"]:
        context_details.append(f"User's name: {ctx['user_name']}.")
    if ctx["skin_goal"]:
        context_details.append(f"User's primary skin goal: {ctx['skin_goal']}.")
    if ctx["tasks"]:
        context_details.append(f"Active daily routine tasks: {', '.join(ctx['tasks'][:5])}.")

    product_a = rec.get("product_a", "Gentle Cleanser")
    product_b = rec.get("product_b", "Moisturizer")
    context_details.append(f"SkinAura recommended products for this profile: {product_a} and {product_b}.")

    system_prompt = (
        "You are SkinAura's intelligent skincare assistant — a kind, knowledgeable, and scientifically grounded guide. "
        "Your role is to help users understand their skin, basic skincare routines, ingredients, and cosmetic product choices. "
        "Use the user's specific SkinAura profile context below to give tailored, practical, and gentle recommendations:\n"
        + "\n".join(f"- {detail}" for detail in context_details) +
        "\n\nRules and Guidelines:\n"
        "1. Structure your answers with clear formatting: use short paragraphs, bullet points, or numbered steps so advice is easy to read.\n"
        "2. Explain active ingredients (e.g., Salicylic Acid, Niacinamide, Hyaluronic Acid, Ceramides, SPF) simply, including when and how to apply them.\n"
        "3. Emphasize patch-testing new products and protecting the skin barrier. Never recommend harsh scrubs or stripping alcohol-heavy products.\n"
        "4. Medical boundaries: You are an educational AI, not a doctor. Never diagnose skin conditions (e.g. cystic acne, eczema, rosacea, melanoma), prescribe drugs, or guarantee cures.\n"
        "5. For severe, painful, rapidly spreading, inflamed, or worrying symptoms, advise the user to consult a board-certified dermatologist.\n"
        "6. If asked about unrelated non-skincare topics, politely steer the conversation back to skin health, routines, and skincare."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *cleaned_history,
        {"role": "user", "content": question.strip()},
    ]
    try:
        answer = groq_chat_completion(api_key, messages)
    except urllib.error.HTTPError as error:
        detail = groq_error_detail(error, api_key)
        app.logger.warning(
            "Groq chat returned HTTP %s for model %s: %s",
            error.code,
            GROQ_MODEL,
            detail or "No provider error details were returned.",
        )
        if error.code == 429:
            return jsonify({"error": "The free Groq request limit has been reached. Please wait a little and try again."}), 503
        if error.code in {401, 403}:
            if error.code == 401:
                return jsonify({"error": "Groq API key is invalid or expired (HTTP 401). Verify GROQ_API_KEY in the .env file."}), 502
            return jsonify({"error": "Groq denied access (HTTP 403). Check that the API key has access to the configured model."}), 502
        if error.code == 404:
            return jsonify({"error": f"The Groq model '{GROQ_MODEL}' is unavailable. Set GROQ_MODEL in .env to an active model."}), 502
        if error.code == 400:
            message = "Groq rejected this chat request."
            if detail:
                message = f"{message} Provider details: {detail}"
            return jsonify({"error": message}), 502
        return jsonify({"error": "The skincare assistant could not respond right now. Please try again shortly."}), 502
    except TimeoutError:
        app.logger.warning("Groq chat request timed out for model %s.", GROQ_MODEL)
        return jsonify({"error": "The Groq request timed out. Please try again."}), 504
    except urllib.error.URLError as error:
        app.logger.warning("Could not connect to Groq chat service: %s", error.reason)
        return jsonify({"error": "Could not connect to Groq. Check your internet connection and try again."}), 502
    except Exception as error:
        app.logger.exception("Groq returned an invalid chat response for model %s: %s", GROQ_MODEL, error)
        return jsonify({"error": f"The skincare assistant could not respond right now: {error}"}), 502

    products = [
        {
            "name": rec[f"product_{slot}"],
            "url": rec[f"product_{slot}_url"],
        }
        for slot in ("a", "b")
    ]
    return jsonify({
        "answer": answer,
        "products": products,
        "profile_key": ctx["profile_key"],
        "weather_label": ctx["weather_label"],
    })


@app.route("/routine", methods=["GET", "POST"])
@login_required
def routine():
    user_id = current_user_id()
    if request.method == "POST":
        require_csrf()
        task_name = request.form.get("task_name", "").strip()
        if not task_name:
            flash("Enter a task for your daily routine.", "error")
        elif len(task_name) > 120:
            flash("Keep routine tasks to 120 characters or fewer.", "error")
        else:
            today = date.today().isoformat()
            with database() as connection:
                connection.execute(
                    """
                    INSERT INTO routine_tasks (user_id, task_name, created_date)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, task_name, today),
                )
                refresh_routine_day(connection, user_id, today)
            flash("Your daily task has been added.", "success")
            return redirect(url_for("routine"))

    profile_key, _weather, recommendation = recommendation_context(user_id)
    today = date.today().isoformat()
    with database() as connection:
        existing_task_count = connection.execute(
            "SELECT COUNT(*) AS total FROM routine_tasks WHERE user_id = ?",
            (user_id,),
        ).fetchone()["total"]
        if not existing_task_count:
            recommended_tasks = [
                *(f"Morning · {task}" for task in recommendation["morning_tasks"]),
                *(f"Evening · {task}" for task in recommendation["evening_tasks"]),
            ]
            connection.executemany(
                """
                INSERT INTO routine_tasks (user_id, task_name, created_date)
                VALUES (?, ?, ?)
                """,
                [(user_id, task, today) for task in recommended_tasks],
            )
    return render_template(
        "routine.html",
        routine=routine_summary(user_id),
        profile_key=profile_key,
    )


@app.post("/routine/tasks/<int:task_id>/complete")
@login_required
def complete_routine_task(task_id: int):
    require_csrf()
    user_id = current_user_id()
    today = date.today().isoformat()
    completed = request.form.get("completed")
    if completed not in {"true", "false"}:
        abort(400, description="Invalid routine task update.")
    with database() as connection:
        task = connection.execute(
            """
            SELECT id FROM routine_tasks
            WHERE id = ? AND user_id = ? AND active = 1
            """,
            (task_id, user_id),
        ).fetchone()
        if task is None:
            abort(404)
        if completed == "true":
            connection.execute(
                """
                INSERT OR IGNORE INTO routine_task_completions (task_id, completion_date)
                VALUES (?, ?)
                """,
                (task_id, today),
            )
        else:
            connection.execute(
                """
                DELETE FROM routine_task_completions
                WHERE task_id = ? AND completion_date = ?
                """,
                (task_id, today),
            )
        refresh_routine_day(connection, user_id, today)
    return redirect(url_for("routine"))


@app.post("/routine/tasks/<int:task_id>/remove")
@login_required
def remove_routine_task(task_id: int):
    require_csrf()
    user_id = current_user_id()
    today = date.today().isoformat()
    with database() as connection:
        cursor = connection.execute(
            "UPDATE routine_tasks SET active = 0 WHERE id = ? AND user_id = ? AND active = 1",
            (task_id, user_id),
        )
        if not cursor.rowcount:
            abort(404)
        refresh_routine_day(connection, user_id, today)
    flash("The routine task was removed.", "success")
    return redirect(url_for("routine"))


@app.post("/analyze")
@login_required
def analyze_skin():
    require_csrf()
    if request.form.get("privacy_consent") != "yes":
        flash("Please confirm the image privacy notice before analysis.", "error")
        return redirect(url_for("index"))

    selected_weather = request.form.get("weather_condition", "mild")
    selected_profile = request.form.get("skin_profile", "image")
    if selected_profile not in PROFILE_OPTIONS:
        flash("Choose a supported skin profile.", "error")
        return redirect(url_for("index"))
    if selected_weather not in WEATHER_CONDITIONS:
        flash("Choose a supported weather condition.", "error")
        return redirect(url_for("index"))

    uploaded = request.files.get("photo")
    camera_data = request.form.get("camera_data", "")
    image = None
    if camera_data:
        try:
            prefix, encoded_image = camera_data.split(",", 1)
            if prefix != "data:image/jpeg;base64":
                raise ValueError("Use a valid camera capture and try again.")
            raw_image = base64.b64decode(encoded_image, validate=True)
            if len(raw_image) > MAX_IMAGE_BYTES:
                raise ValueError("That image is larger than 8 MB. Please choose a smaller file.")
            image = validated_image(raw_image, "JPEG")
        except (ValueError, TypeError):
            flash("The camera capture is invalid. Please capture a new image.", "error")
            return redirect(url_for("index"))
    elif not uploaded or not uploaded.filename:
        flash("Choose a face image before starting the analysis.", "error")
        return redirect(url_for("index"))
    elif not is_allowed(uploaded.filename):
        flash("Use JPG, PNG, WEBP, or BMP images up to 8 MB.", "error")
        return redirect(url_for("index"))
    else:
        extension = uploaded.filename.rsplit(".", 1)[1].lower()

    try:
        if image is None:
            raw_image = uploaded.stream.read(MAX_IMAGE_BYTES + 1)
            if len(raw_image) > MAX_IMAGE_BYTES:
                raise ValueError("That image is larger than 8 MB. Please choose a smaller file.")
            image = validated_image(raw_image, IMAGE_FORMATS[extension])

        prediction_breakdown = predict_skin_type_details(image)
        predicted_label = prediction_breakdown[0]["label"]
        confidence = prediction_breakdown[0]["confidence"]
        profile, manual_profile = resolve_profile(selected_profile, predicted_label)
        recommendation = get_recommendation(profile, selected_weather)
        products = [
            {
                "key": f"{profile}:{selected_weather}:product_{slot}",
                "name": recommendation[f"product_{slot}"],
                "url": recommendation[f"product_{slot}_url"],
            }
            for slot in ("a", "b")
        ]
        result = {
            "label": profile.title(),
            "predicted_label": predicted_label.title(),
            "manual_profile": manual_profile,
            "confidence": confidence,
            "prediction_breakdown": prediction_breakdown,
            "is_uncertain": confidence < 50.0,
            "model_warning": get_model_warning(),
            "recommendation": recommendation,
            "products": products,
            "profile_key": profile,
            "weather_advice": get_weather_advice(selected_weather, profile),
        }
        with database() as connection:
            connection.execute(
                """
                INSERT INTO analysis_results (user_id, result_json, weather_condition)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    result_json = excluded.result_json,
                    weather_condition = excluded.weather_condition
                """,
                (current_user_id(), json.dumps(result), selected_weather),
            )
    except (ValueError, FileNotFoundError) as error:
        flash(str(error), "error")
        return redirect(url_for("index"))
    except Exception as error:
        flash(f"Analysis failed: {error}", "error")
        return redirect(url_for("index"))

    return redirect(url_for("analysis_results"))


@app.get("/results")
@login_required
def analysis_results():
    with database() as connection:
        row = connection.execute(
            "SELECT result_json, weather_condition FROM analysis_results WHERE user_id = ?",
            (current_user_id(),),
        ).fetchone()
        checked_products = {
            item["product_key"]: bool(item["checked"])
            for item in connection.execute(
                "SELECT product_key, checked FROM product_checks WHERE user_id = ?",
                (current_user_id(),),
            )
        }

    if row is None:
        flash("Analyze a photo to see your results.", "error")
        return redirect(url_for("index"))

    result = json.loads(row["result_json"])
    selected_weather = row["weather_condition"]
    recommendation = get_recommendation(result["profile_key"], selected_weather)
    result["recommendation"] = recommendation
    result["weather_advice"] = get_weather_advice(selected_weather, result["profile_key"])
    result["products"] = [
        {
            "key": f"{result['profile_key']}:{selected_weather}:product_{slot}",
            "name": recommendation[f"product_{slot}"],
            "url": recommendation[f"product_{slot}_url"],
        }
        for slot in ("a", "b")
    ]
    for product in result["products"]:
        product["checked"] = checked_products.get(product["key"], False)
    return render_template(
        "results.html",
        result=result,
        selected_weather=selected_weather,
        weather_label=WEATHER_CONDITIONS[selected_weather],
        disclaimer=DISCLAIMER,
    )


@app.post("/products/check")
@login_required
def update_product_check():
    require_csrf()
    product_key = request.form.get("product_key", "")
    checked = request.form.get("checked", "")
    valid_keys = {
        f"{profile}:{weather}:product_{slot}"
        for profile in RECOMMENDATIONS
        for weather in WEATHER_CONDITIONS
        for slot in ("a", "b")
    }
    if product_key not in valid_keys or checked not in {"true", "false"}:
        abort(400, description="Invalid product checklist update.")
    with database() as connection:
        connection.execute(
            """
            INSERT INTO product_checks (user_id, product_key, checked)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, product_key) DO UPDATE SET checked = excluded.checked
            """,
            (current_user_id(), product_key, int(checked == "true")),
        )
    return jsonify({"checked": checked == "true"})


@app.post("/report")
@login_required
def download_report():
    require_csrf()
    profile = request.form.get("profile", "")
    predicted_label = request.form.get("predicted_label", "")
    weather = request.form.get("weather", "")
    try:
        confidence = float(request.form.get("confidence", ""))
    except ValueError:
        abort(400, description="Invalid report confidence.")
    if (
        profile not in RECOMMENDATIONS
        or predicted_label.lower() not in {"oily", "dry", "normal"}
        or weather not in WEATHER_CONDITIONS
        or not 0 <= confidence <= 100
    ):
        abort(400, description="The analysis report details are invalid.")
    recommendation = get_recommendation(profile, weather)
    lines = [
        f"Generated: {date.today().isoformat()}",
        f"Skin profile: {profile.title()}",
        f"Photo estimate: {predicted_label.title()} ({confidence:.1f}% confidence)",
        "",
        "PRODUCT SUGGESTIONS",
        f"- {recommendation['product_a']}: {recommendation['product_a_url']}",
        f"- {recommendation['product_b']}: {recommendation['product_b_url']}",
        "",
        "MORNING ROUTINE",
        *[f"- {task}" for task in recommendation["morning_tasks"]],
        "",
        "EVENING ROUTINE",
        *[f"- {task}" for task in recommendation["evening_tasks"]],
        "",
        "DAILY NOTES",
        *[f"- {tip}" for tip in recommendation["tips"]],
        "",
        "WEATHER",
        f"{WEATHER_CONDITIONS[weather]}: {get_weather_advice(weather, profile)}",
        "",
        DISCLAIMER,
        "Your uploaded image is not included in or stored with this report.",
    ]
    response = make_response(make_pdf(lines))
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = 'attachment; filename="skinaura-skin-analysis.pdf"'
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def request_too_large(_error):
    flash("That image is larger than 8 MB. Please choose a smaller file.", "error")
    if current_user_id() is None:
        return redirect(url_for("login"))
    return redirect(url_for("index"))


init_database()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )
