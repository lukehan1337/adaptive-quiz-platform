import os
import json
import random
import pathlib
import platform
import subprocess
import tempfile
import datetime
import time
import matplotlib.pyplot as plt

import streamlit as st
from gtts import gTTS

# ─────────────────────────── Paths ───────────────────────────
# BASE_DIR: 현재 파일 기준으로 경로 잡음 (상대경로 관리 편하게)
BASE_DIR = pathlib.Path(__file__).parent
# 유저별 데이터 저장 파일
DATA_FILE = BASE_DIR / "user_data.json"
# 받아쓰기 문제 데이터 파일
DICT_FILE = BASE_DIR / "dictation_data.json"
# 영단어 문제 데이터 파일
VOCAB_FILE = BASE_DIR / "vocab_data.json"
# 도형 퀴즈는 별도 데이터 파일 없이 코드로 생성

# ───────────────────────── Utilities ─────────────────────────
import json
import pathlib

def load_json(path: pathlib.Path, default):
    """
    json 파일을 읽어서 파싱.
    - 파일이 없거나 깨진 경우 default 리턴
    - 인코딩 문제(utf-8 실패 시 cp949)도 예외 처리
    """
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            try:
                return json.loads(path.read_text(encoding="cp949"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # 인코딩 실패 or JSON 파싱 실패
        except json.JSONDecodeError:
            pass  # JSON 형식 오류
    return default

def save_json(path: pathlib.Path, obj):
    # json 파일로 저장. 한글 깨짐 방지 위해 ensure_ascii=False
    # indent=2로 저장하면 보기 편함
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ───────────────────────── Skill Helpers ─────────────────────
def skill_level(best_ratio: float) -> str:
    # 점수 비율(0~1) 받아서 등급 문자열로 변환
    # 50% 미만: 초보, 90% 미만: 중수, 그 이상: 고수
    pct = best_ratio * 100
    if pct < 50:
        return "초보"
    elif pct < 90:
        return "중수"
    return "고수"

def questions_for_level(level: str) -> int:
    # 등급별로 문제 개수 다르게 반환
    # dict로 관리하면 나중에 등급 추가할 때 편함
    # ex) 초보: 5, 중수: 10, 고수: 15
    return {"초보": 5, "중수": 10, "고수": 15}[level]

# --------------------- Shape Utilities ---------------------
def generate_polygon_image(sides: int) -> str:
    """
    주어진 변의 수로 정다각형을 그린 후, 임시 png 파일 경로를 리턴.
    """
    import numpy as np
    theta = np.linspace(0, 2*np.pi, sides + 1)
    x = np.cos(theta); y = np.sin(theta)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(x, y, "-o"); ax.axis("off"); ax.set_aspect("equal")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return tmp.name

def tts_play(text: str):
    # TTS로 한글 읽어주는 함수
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        gTTS(text, lang="ko").save(tmp.name)

    system_name = platform.system()

    if system_name == "Windows":
        # 윈도우는 기본 플레이어로 실행
        os.startfile(tmp.name)
    elif system_name == "Darwin":
        # macOS는 afplay 사용
        subprocess.run(["afplay", tmp.name])
    else:
        # Linux 등은 mpg123 사용 (설치되어 있어야 함)
        subprocess.run(["mpg123", tmp.name])

def clear_quiz_state():
    # 퀴즈 상태 초기화. 새로 시작할 때 사용
    # session_state에 남아있는 퀴즈 관련 키들 싹 지움
    QUIZ_KEYS = (
        "dict_active", "math_active", "voc_active", "shape_active",
        "dict_questions", "dict_answers", "dict_idx",
        "math_questions", "math_answers",
        "voc_pairs", "voc_answers",
        "shape_questions", "shape_answers",
        "math_submitted", "voc_submitted", "shape_submitted",
    )
    # session_state에 있는 키들 중 퀴즈 관련된 것만 삭제
    for k in list(st.session_state.keys()):
        if k.startswith(("dict_", "math_", "voc_", "shape_")) or k in QUIZ_KEYS:
            del st.session_state[k]
    # username과 현재 페이지는 유지 - 대시보드로 가지 않음

# ───────────────────── Persistent Storage ────────────────────
# 유저 데이터 전체를 USERS에 로드
USERS = load_json(DATA_FILE, {})

def get_user(name: str):
    # 유저 정보 가져오는 함수. 없으면 새로 만들어줌
    # user_data.json에 저장/불러오기
    if name not in USERS:
        # 유저 없으면 기본 구조로 생성
        USERS[name] = {"dict": [], "math": [], "vocab": [], "shape": [], "history": []}
        save_json(DATA_FILE, USERS)
    user = USERS[name]

    # 업데이트 후 첫 실행이면 기존 기록을 history에 채워줌
    # history가 비어있을 때만 동작
    if not user.get("history"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # dict, math, vocab, shape 각각의 점수 리스트를 history로 옮김
        for quiz_key in ("dict", "math", "vocab", "shape"):
            for s in user.get(quiz_key, []):
                user["history"].append(
                    {
                        "time": timestamp,
                        "type": quiz_key,
                        "score": round(s * 100, 1),
                    }
                )
        save_json(DATA_FILE, USERS)

    return user

# ───────────────────────────  UI  ────────────────────────────
# streamlit 페이지 설정. 제목, 아이콘, 레이아웃
st.set_page_config(page_title="학습 게임 모음", page_icon="🎲", layout="centered")
st.title("📝 학습 게임 모음")

def quiz_locked() -> bool:
    # 퀴즈 진행 중에는 (제출 완료 후 피드백 화면까지 포함) 메뉴 이동을 막는다
    return any([
        st.session_state.get("dict_active", False),
        st.session_state.get("math_active", False),
        st.session_state.get("voc_active", False),
        st.session_state.get("shape_active", False),
    ])

def main():
    # ---------- 로그인 ----------
    # 이름 입력 안 했으면 입력받고, 없으면 멈춤
    if "username" not in st.session_state or st.session_state.username == "":
        st.session_state.username = st.text_input("이름을 입력하세요")
        if not st.session_state.username:
            st.stop()

    # 유저 정보 불러오기
    user = get_user(st.session_state.username)

    # 퀴즈 중인지 확인
    lock = quiz_locked()

    if not lock:
        # 사이드바 메뉴. 대시보드/받아쓰기/수학/단어/도형/단어 암기/로그아웃
        st.session_state.page = st.sidebar.radio(
            "메뉴",
            ["대시보드", "받아쓰기", "수학", "단어", "도형", "단어 암기", "로그아웃"],
            key="nav",
            help="메뉴를 선택하세요",
        )
    else:
        # 퀴즈 중이면 메뉴 못 바꾸게 막음
        st.sidebar.info("퀴즈 진행 중 ‑ 종료 전까지 이동 불가")

    # 현재 페이지 결정
    page = st.session_state.page if "page" in st.session_state else "대시보드"

    # 각 메뉴별로 함수 분기
    if page == "대시보드":
        show_dashboard(user)
    elif page == "받아쓰기":
        show_dictation(user)
    elif page == "수학":
        show_math(user)
    elif page == "단어":
        show_vocab(user)
    elif page == "도형":
        show_shape(user)
    elif page == "단어 암기":
        show_flashcards()
    else:  # 로그아웃
        clear_quiz_state()
        st.session_state.username = ""
        st.rerun()

# ──────────────────────── Page Views ─────────────────────────
def show_dashboard(user):
    # 대시보드. 내 최고점수, 평균, 최근 기록 보여줌
    st.subheader(f"👋 {st.session_state.username} 님의 성적표")

    # 각 퀴즈별 최고점수 계산 (없으면 0)
    best_dict = max(user["dict"] or [0])
    best_math = max(user["math"] or [0])
    best_vocab = max(user["vocab"] or [0])
    best_shape = max(user["shape"] or [0])
    # 평균 점수 계산
    avg_best = round((best_dict + best_math + best_vocab + best_shape) / 4, 3)

    # 4개 컬럼으로 점수 표시
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📖 받아쓰기", f"{best_dict*100:.1f}%")
    col1.caption(skill_level(best_dict))
    col2.metric("➕ 수학", f"{best_math*100:.1f}%")
    col2.caption(skill_level(best_math))
    col3.metric("🗣 단어", f"{best_vocab*100:.1f}%")
    col3.caption(skill_level(best_vocab))
    col4.metric("🔺 도형", f"{best_shape*100:.1f}%")
    col4.caption(skill_level(best_shape))
    col4.metric("⭐ 평균", f"{avg_best*100:.1f}%")

    # 최근 10회 기록 테이블로 보여줌
    history = sorted(user.get("history", []), key=lambda x: x["time"], reverse=True)[:10]
    if history:
        st.markdown("#### 최근 10회 시험 기록")
        table_data = {
            "시간": [],
            "시험 종류": [],
            "정답률(%)": [],
        }
        for h in history:
            # 날짜 파싱 실패하면 원본 출력
            try:
                ts = datetime.datetime.strptime(h["time"].replace("‑", "-"), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts = h["time"]
            table_data["시간"].append(ts)
            type_display = {"dict":"받아쓰기","math":"수학","vocab":"단어","shape":"도형","flash":"단어 암기"}.get(h["type"], h["type"])
            table_data["시험 종류"].append(type_display)
            table_data["정답률(%)"].append(h["score"])
        st.table(table_data)

    st.info("왼쪽 사이드바 메뉴에서 게임을 선택해보세요")

# --------------------------- Dictation -----------------------
def show_dictation(user):
    st.header("📖 받아쓰기")

    # ---------- 준비 단계 ----------
    # dict_active, dict_questions 없으면 준비상태
    if (
        not st.session_state.get("dict_active")
        or "dict_questions" not in st.session_state
    ):
        # 등급에 따라 문제 개수 다르게
        level = skill_level(max(user["dict"] or [0]))
        n_q = questions_for_level(level)
        st.write(f"{level} 등급 · {n_q}문제 받아쓰기를 시작합니다. 준비되면 [응시하기]를 눌러주세요.")
        if st.button("응시하기", key="dict_start"):
            # 문제 풀기 시작하면 상태값 세팅
            st.session_state.dict_active = True
            # 받아쓰기 문제 pool에서 랜덤 추출
            pool = load_json(DICT_FILE, [])[:30]
            k = min(n_q, len(pool))  # ← 여기서 '당구'가 아니라 'pool'이어야 함
            st.session_state.dict_questions = random.sample(pool, k)
            st.session_state.dict_answers = [""] * k
            st.session_state.dict_idx = 0
            st.rerun()
        return  # 시작 전엔 여기서 멈춤
    # ---------- 문제 푸는 단계 ----------
    idx = st.session_state.dict_idx
    if idx < len(st.session_state.dict_questions):
        sent = st.session_state.dict_questions[idx]
        st.write(f"문제 {idx + 1} / {len(st.session_state.dict_questions)}")
        # TTS로 문제 읽어주기
        st.button("🔊 들려주기", on_click=lambda: tts_play(sent))
        # 받아쓰기 입력받기
        st.session_state.dict_answers[idx] = st.text_input(
            "받아쓰기 결과를 입력하세요",
            value=st.session_state.dict_answers[idx],
            key=f"dict_input_{idx}",
        )
        # 다음 문제로 이동
        if st.button("다음", key="dict_next"):
            st.session_state.dict_idx += 1
            st.rerun()
    else:
        # 다 풀었으면 결과 체크
        check_and_show_result(
            user,
            quiz="dict",
            questions=st.session_state.dict_questions,
            user_inputs=st.session_state.dict_answers,
            answer_func=lambda q: q,
        )

# ---------------------------- Math ---------------------------
def show_math(user):
    st.header("➕ 수학 퀴즈")

    # ---------- 준비 단계 ----------
    if (
        not st.session_state.get("math_active")
        or "math_questions" not in st.session_state
    ):
        # 등급에 따라 문제 개수 다르게
        level = skill_level(max(user["math"] or [0]))
        n_q = questions_for_level(level)
        st.write(f"{level} 등급 · {n_q}문제를 풉니다. [응시하기]를 눌러 시작하세요.")
        if st.button("응시하기", key="math_start"):
            st.session_state.math_active = True
            # 랜덤으로 수식 생성 (1~9, + 또는 -)
            st.session_state.math_questions = [
                (random.randint(1, 9), random.randint(1, 9), random.choice(["+", "-"]))
                for _ in range(n_q)
            ]
            st.session_state.math_answers = [""] * n_q
            st.session_state.math_submitted = False  # 제출 상태 초기화
            st.rerun()
        return

    # ---------- 제출 후 결과 표시 ----------
    if st.session_state.get("math_submitted"):
        # 정답 계산
        correct = [str(eval(f"{a}{op}{b}")) for a, b, op in st.session_state.math_questions]
        check_and_show_result(
            user,
            quiz="math",
            questions=[f"{a}{op}{b}" for a, b, op in st.session_state.math_questions],
            user_inputs=st.session_state.math_answers,
            correct_answers=correct,
        )
        return

    # ---------- 문제 푸는 단계 ----------
    for i, (a, b, op) in enumerate(st.session_state.math_questions):
        # 각 문제별로 입력받기
        st.session_state.math_answers[i] = st.text_input(
            f"{a} {op} {b} =",
            value=st.session_state.math_answers[i],
            key=f"math_input_{i}",
        )

    if st.button("제출", key="math_submit"):
        st.session_state.math_submitted = True
        st.rerun()

# -------------------------- Shape ---------------------------
def show_shape(user):
    st.header("🔺 도형 맞추기")
    
    # ---------- 준비 단계 ----------
    if (not st.session_state.get("shape_active")
        or "shape_questions" not in st.session_state):
        level = skill_level(max(user["shape"] or [0]))
        n_q = questions_for_level(level)
        st.write(f"{level} 등급 · {n_q}문제 도형 퀴즈입니다. [응시하기]를 눌러 시작하세요.")
        if st.button("응시하기", key="shape_start"):
            st.session_state.shape_active = True
            st.session_state.shape_questions = [random.randint(3,8) for _ in range(n_q)]
            st.session_state.shape_answers = [""] * n_q
            st.session_state.shape_submitted = False  # 제출 상태 초기화
            st.rerun()
        return

    # ---------- 제출 후 결과 표시 ----------
    if st.session_state.get("shape_submitted"):
        correct = [str(s) for s in st.session_state.shape_questions]
        check_and_show_result(
            user,
            quiz="shape",
            questions=[f"{sides}각형" for sides in st.session_state.shape_questions],
            user_inputs=st.session_state.shape_answers,
            correct_answers=correct,
        )
        return

    # ---------- 문제 푸는 단계 ----------
    for i, sides in enumerate(st.session_state.shape_questions):
        img_path = generate_polygon_image(sides)
        st.image(img_path, width=150)
        st.session_state.shape_answers[i] = st.text_input(
            f"{i+1}. 몇각형인가요?", value=st.session_state.shape_answers[i], key=f"shape_input_{i}"
        )
    
    if st.button("제출", key="shape_submit"):
        st.session_state.shape_submitted = True
        st.rerun()

# --------------------------- Vocabulary ----------------------
def show_vocab(user):
    st.header("🗣 단어 맞추기")

    # ---------- 준비 단계 ----------
    if (
        not st.session_state.get("voc_active")
        or "voc_pairs" not in st.session_state
    ):
        # 등급에 따라 문제 개수 다르게
        level = skill_level(max(user["vocab"] or [0]))
        n_q = questions_for_level(level)
        st.write(f"{level} 등급 · {n_q}문제 영단어 퀴즈입니다. [응시하기] 버튼을 눌러 시작하세요.")
        if st.button("응시하기", key="voc_start"):
            st.session_state.voc_active = True
            vocab = load_json(VOCAB_FILE, {})
            population = list(vocab.items())
            # 단어가 너무 적으면 에러 표시
            if len(population) < 5:
                st.error("vocab_data.json 파일에 최소 5개 이상의 단어가 필요합니다.")
                st.session_state.voc_active = False
            else:
                k = min(n_q, len(population))
                # 랜덤으로 단어 추출
                st.session_state.voc_pairs = random.sample(population, k)
                st.session_state.voc_answers = [""] * k
                st.session_state.voc_submitted = False  # 제출 상태 초기화
                st.rerun()
        return

    # ---------- 제출 후 결과 표시 ----------
    if st.session_state.get("voc_submitted"):
        correct = [kor for _, kor in st.session_state.voc_pairs]
        check_and_show_result(
            user,
            quiz="vocab",
            questions=[eng for eng, _ in st.session_state.voc_pairs],
            user_inputs=st.session_state.voc_answers,
            correct_answers=correct,
        )
        return

    # ---------- 문제 푸는 단계 ----------
    for i, (eng, kor) in enumerate(st.session_state.voc_pairs):
        # 영어 단어 보여주고 한글 입력받기
        st.session_state.voc_answers[i] = st.text_input(
            f"{eng} →",
            value=st.session_state.voc_answers[i],
            key=f"voc_input_{i}",
        )

    if st.button("제출", key="voc_submit"):
        st.session_state.voc_submitted = True
        st.rerun()

# ------------------------- Flashcards -----------------------
def show_flashcards():
    st.header("📚 단어 암기 (5초 자동 전환)")
    vocab = load_json(VOCAB_FILE, {})
    if not vocab:
        st.error("vocab_data.json 데이터가 없습니다.")
        return
    if "flash_idx" not in st.session_state:
        st.session_state.flash_idx = 0
    words = list(vocab.items())
    eng, kor = words[st.session_state.flash_idx % len(words)]
    st.info(f"**{eng}**  →  {kor}")
    st.write("5초마다 새로운 단어가 표시됩니다. [멈추기]를 누르면 정지합니다.")
    if st.button("멈추기"):
        st.session_state.flash_running = False
    if st.session_state.get("flash_running", True):
        time.sleep(5)
        st.session_state.flash_idx += 1
        st.rerun()

# ────────────────── Generic Result Renderer ──────────────────
def check_and_show_result(
    user,
    quiz: str,
    questions,
    user_inputs,
    answer_func=None,
    correct_answers=None,
):
    # 결과 보여주고 점수 저장하는 함수
    # answer_func로 정답 만들거나, correct_answers 직접 넘겨도 됨
    if correct_answers is None and answer_func is not None:
        correct_answers = [answer_func(q) for q in questions]

    # 정답 체크 (strip으로 공백 제거)
    correct_mask = [u.strip() == c for u, c in zip(user_inputs, correct_answers)]
    score = sum(correct_mask) / len(questions)

    st.subheader("결과")
    # 각 문제별로 정답/오답 표시
    for i, (q, u, c, ok) in enumerate(zip(questions, user_inputs, correct_answers, correct_mask), 1):
        st.write(f"{i}. {q} ⇒ {u or '❌ 미입력'} | 정답: {c} {'✅' if ok else '❌'}")

    st.success(f"정확도 : {score * 100:.1f}%")

    # 점수랑 기록 저장
    user[quiz].append(score)
    # 상세 기록도 history에 추가
    user.setdefault("history", []).append(
        {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": quiz,
            "score": round(score * 100, 1),
        }
    )
    save_json(DATA_FILE, USERS)

    # 완료하기 버튼: 성적 확인 후 대시보드로 이동
    if st.button("완료하기"):
        clear_quiz_state()
        # 대시보드로 자동 이동
        st.session_state.page = "대시보드"
        st.rerun()

# ─────────────────────────── Main ────────────────────────────
if __name__ == "__main__":
    # 메인 함수 실행 (streamlit에서 직접 실행할 때만)
    main()