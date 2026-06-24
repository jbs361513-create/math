"""
복소수 무리함수 3D 시각화 - Streamlit 버전 (평행이동 기능 추가)
사용자 정의 수식 √(expr), ∛(expr), expr^(p/q) 지원
여러 함수 동시 비교 및 각 함수별 3D 평행이동 지원
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import re as _re

st.set_page_config(
    page_title="복소수 무리함수 시각화 (평행이동)",
    page_icon="🔢",
    layout="wide",
)

st.title("복소수 무리함수 3D 시각화")
st.markdown(
    "실수 **x** 입력 → f(x) 의 **실수부(y축)** 와 **허수부(z축)** 를 3D 곡선으로 표시 (함수별 평행이동 가능)"
)

# ── 수식 파서 ──────────────────────────────────────────────────
SAFE_NAMES = {
    "x": None,          # 런타임에 주입
    "pi": np.pi,
    "e":  np.e,
    "sin":  np.sin,  "cos": np.cos,  "tan": np.tan,
    "exp":  np.exp,  "log": np.log,  "log2": np.log2, "log10": np.log10,
    "abs":  np.abs,  "sqrt": np.sqrt,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
}

def preprocess(expr: str) -> str:
    s = expr.strip()
    s = s.replace("π", "pi").replace("×", "*").replace("÷", "/")
    s = _re.sub(r'\^', '**', s)
    s = _re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)
    s = _re.sub(r'(\))([a-zA-Z(])', r'\1*\2', s)
    s = _re.sub(r'(?<![a-zA-Z])(x)\(', r'\1*(', s)
    return s

def safe_eval(expr_str: str, x_arr: np.ndarray):
    processed = preprocess(expr_str)
    env = {**SAFE_NAMES, "x": x_arr.astype(complex)}
    try:
        result = eval(compile(processed, "<string>", "eval"), {"__builtins__": {}}, env)
        return np.asarray(result, dtype=complex)
    except Exception as ex:
        raise ValueError(f"수식 오류 ({processed}): {ex}")

def parse_function(fn_type: str, inner: str, p: int, q: int, x_arr: np.ndarray, 
                   x_shift: float = 0.0, y_shift: float = 0.0, z_shift: float = 0.0):
    """
    평행이동(x_shift, y_shift, z_shift) 매개변수 추가
    """
    # 1. X축 평행이동 반영 (수학적으로 f(x - a) 형태가 되도록 x 대신 x - x_shift 대입)
    adjusted_x = x_arr - x_shift

    if fn_type == "expr":
        w = safe_eval(inner, adjusted_x)
    else:
        w_inner = safe_eval(inner, adjusted_x)
        r  = np.abs(w_inner)
        th = np.angle(w_inner)

        if fn_type == "sqrt":
            exp = 0.5
        elif fn_type == "cbrt":
            exp = 1/3
        else:
            exp = p / q
        w = (r ** exp) * np.exp(1j * exp * th)
    
    # 2. Y축(실수부), Z축(허수부) 평행이동 반영 (+b, +c 형태)
    # 복소수 형태로 더해주면 Re(w) + y_shift, Im(w) + z_shift 가 동시에 처리됩니다.
    return w + (y_shift + 1j * z_shift)


# ── 색상 팔레트 ────────────────────────────────────────────────
COLORS = ["#3b82f6","#f97316","#22c55e","#a855f7",
          "#ef4444","#06b6d4","#eab308","#ec4899"]

# ── 세션 스테이트: 그래프 목록 ────────────────────────────────
if "graphs" not in st.session_state:
    st.session_state.graphs = [
        {
            "fn_type": "sqrt", "inner": "x", "p": 2, "q": 3, "label": "√x", "active": True,
            "x_shift": 0.0, "y_shift": 0.0, "z_shift": 0.0
        },
    ]

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("x 범위")
    x_min = st.number_input("x 최솟값", value=-6.0, step=0.5, format="%.1f")
    x_max = st.number_input("x 최댓값", value=6.0,  step=0.5, format="%.1f")
    n_pts = st.slider("포인트 수", 300, 3000, 1000, 100)

    st.divider()
    show_planes = st.checkbox("기준면 표시", value=True)
    show_axes   = st.checkbox("축선 표시",   value=True)
    line_width  = st.slider("선 굵기", 1, 8, 4)

# ── 그래프 추가/편집 패널 ──────────────────────────────────────
st.subheader("그래프 추가 / 편집")

# 새 그래프 추가
with st.expander("➕  새 그래프 추가", expanded=False):
    new_type = st.selectbox(
        "함수 유형",
        ["expr  f(x)", "sqrt  √( )", "cbrt  ∛( )", "pq  ( )^(p/q)"],
        key="new_type",
    )
    fn_type_key = new_type.split()[0]

    inner_label = "수식 f(x)" if fn_type_key == "expr" else "근호 안 수식"
    inner_default = "x - 3" if fn_type_key == "expr" else "x**2 - 1"
    new_inner = st.text_input(
        inner_label, value=inner_default, key="new_inner",
        help="x, sin, cos, exp, log, pi, e 등 사용 가능",
    )

    if fn_type_key == "pq":
        cp, cq = st.columns(2)
        with cp: new_p = st.number_input("p (분자)", value=2, min_value=1, step=1, key="new_p")
        with cq: new_q = st.number_input("q (분모)", value=3, min_value=1, step=1, key="new_q")
    else:
        new_p, new_q = 2, 3

    # 평행이동 입력 UI 추가
    st.markdown("**평행이동 설정**")
    sh1, sh2, sh3 = st.columns(3)
    with sh1: new_xs = st.number_input("X축 이동 (x축 방향)", value=0.0, step=0.5, key="new_xs")
    with sh2: new_ys = st.number_input("Y축 이동 (Re 방향)", value=0.0, step=0.5, key="new_ys")
    with sh3: new_zs = st.number_input("Z축 이동 (Im 방향)", value=0.0, step=0.5, key="new_zs")

    new_label = st.text_input("범례 이름 (선택)", value="", key="new_label", placeholder="비워두면 자동 생성")

    if fn_type_key == "expr": auto_label = f"f(x) = {new_inner}"
    elif fn_type_key == "sqrt": auto_label = f"√({new_inner})"
    elif fn_type_key == "cbrt": auto_label = f"∛({new_inner})"
    else: auto_label = f"({new_inner})^({new_p}/{new_q})"

    shift_info = ""
    if new_xs != 0 or new_ys != 0 or new_zs != 0:
        shift_info = f" (이동: {new_xs}, {new_ys}, {new_zs})"

    final_label = (new_label.strip() or auto_label) + shift_info
    st.caption(f"레이블 미리보기: **{final_label}**")

    if st.button("그래프 추가 ↗"):
        try:
            test_x = np.array([1.0])
            parse_function(fn_type_key, new_inner, int(new_p), int(new_q), test_x, new_xs, new_ys, new_zs)
            st.session_state.graphs.append({
                "fn_type": fn_type_key, "inner": new_inner, "p": int(new_p), "q": int(new_q),
                "label": final_label, "active": True,
                "x_shift": float(new_xs), "y_shift": float(new_ys), "z_shift": float(new_zs)
            })
            st.success(f"'{final_label}' 추가됨!")
            st.rerun()
        except Exception as ex:
            st.error(str(ex))

st.divider()

# 기존 그래프 목록 편집
if st.session_state.graphs:
    st.markdown("**등록된 그래프**")
    to_delete = []
    for idx, g in enumerate(st.session_state.graphs):
        col_chk, col_info, col_edit, col_del = st.columns([0.5, 3, 1.5, 1])
        with col_chk:
            g["active"] = st.checkbox("", value=g["active"], key=f"chk_{idx}", label_visibility="collapsed")
        with col_info:
            color_dot = f'<span style="color:{COLORS[idx % len(COLORS)]};font-size:18px">●</span>'
            st.markdown(f'{color_dot} **{g["label"]}**', unsafe_allow_html=True)
        with col_edit:
            if st.button("수정", key=f"edit_{idx}"):
                st.session_state[f"editing_{idx}"] = not st.session_state.get(f"editing_{idx}", False)
        with col_del:
            if st.button("삭제", key=f"del_{idx}"):
                to_delete.append(idx)

        # 인라인 수정 폼 (평행이동 항목 포함)
        if st.session_state.get(f"editing_{idx}", False):
            with st.container():
                ec1, ec2 = st.columns([1, 2])
                with ec1:
                    e_type = st.selectbox(
                        "유형", ["expr  f(x)", "sqrt  √( )", "cbrt  ∛( )", "pq  ( )^(p/q)"],
                        index=["expr","sqrt","cbrt","pq"].index(g["fn_type"]), key=f"etype_{idx}"
                    )
                with ec2: e_inner = st.text_input("수식", value=g["inner"], key=f"einner_{idx}")

                ep, eq_ = st.columns(2)
                with ep: e_p = st.number_input("p", value=g["p"], min_value=1, key=f"ep_{idx}")
                with eq_: e_q = st.number_input("q", value=g["q"], min_value=1, key=f"eq_{idx}")

                st.markdown("  └ 평행이동 수정")
                esh1, esh2, esh3 = st.columns(3)
                with esh1: e_xs = st.number_input("X축 이동", value=g.get("x_shift", 0.0), step=0.5, key=f"exs_{idx}")
                with esh2: e_ys = st.number_input("Y축 이동(Re)", value=g.get("y_shift", 0.0), step=0.5, key=f"eys_{idx}")
                with esh3: e_zs = st.number_input("Z축 이동(Im)", value=g.get("z_shift", 0.0), step=0.5, key=f"ezs_{idx}")

                e_label = st.text_input("레이블", value=g["label"], key=f"elabel_{idx}")

                if st.button("저장", key=f"save_{idx}"):
                    try:
                        e_fn_type = e_type.split()[0]
                        parse_function(e_fn_type, e_inner, int(e_p), int(e_q), np.array([1.0]), e_xs, e_ys, e_zs)
                        g["fn_type"] = e_fn_type
                        g["inner"]   = e_inner
                        g["p"]       = int(e_p)
                        g["q"]       = int(e_q)
                        g["label"]   = e_label
                        g["x_shift"] = float(e_xs)
                        g["y_shift"] = float(e_ys)
                        g["z_shift"] = float(e_zs)
                        st.session_state[f"editing_{idx}"] = False
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))

    for idx in reversed(to_delete):
        st.session_state.graphs.pop(idx)
    if to_delete:
        st.rerun()

st.divider()

# ── 계산 및 그래프 ─────────────────────────────────────────────
if x_min >= x_max:
    st.error("x 최솟값 < 최댓값 이어야 합니다.")
    st.stop()

x = np.linspace(x_min, x_max, n_pts)
active_graphs = [g for g in st.session_state.graphs if g["active"]]

if not active_graphs:
    st.info("위에서 그래프를 하나 이상 활성화하세요.")
    st.stop()

rng_re, rng_im = 0.0, 0.0
results = []
for g in active_graphs:
    try:
        # 평행이동 값 안전하게 꺼내기 (기본값 0.0)
        xs = g.get("x_shift", 0.0)
        ys = g.get("y_shift", 0.0)
        zs = g.get("z_shift", 0.0)
        
        W = parse_function(g["fn_type"], g["inner"], g["p"], g["q"], x, xs, ys, zs)
        Re_W = np.real(W); Im_W = np.imag(W)
        finite = np.isfinite(Re_W) & np.isfinite(Im_W)
        if np.any(finite):
            rng_re = max(rng_re, np.max(np.abs(Re_W[finite])))
            rng_im = max(rng_im, np.max(np.abs(Im_W[finite])))
        results.append((g, W, Re_W, Im_W))
    except Exception as ex:
        st.warning(f"'{g['label']}' 계산 오류: {ex}")

rng_re = max(rng_re, 1.0) + 0.3
rng_im = max(rng_im, 1.0) + 0.3
traces = []

if show_planes:
    xg = np.array([x_min, x_max, x_max, x_min])
    traces.append(go.Mesh3d(
        x=xg, y=[0,0,0,0], z=np.array([-rng_im,-rng_im,rng_im,rng_im]),
        color="lightblue", opacity=0.07, name="Im=0 평면", showlegend=True, hoverinfo="skip",
    ))
    traces.append(go.Mesh3d(
        x=xg, y=np.array([-rng_re,-rng_re,rng_re,rng_re]), z=[0,0,0,0],
        color="lightyellow", opacity=0.07, name="Re=0 평면", showlegend=True, hoverinfo="skip",
    ))

if show_axes:
    for xs,ys,zs,xe,ye,ze,name in [
        (x_min,0,0,x_max,0,0,"x축"),
        (0,-rng_re,0,0,rng_re,0,"Re축"),
        (0,0,-rng_im,0,0,rng_im,"Im축"),
    ]:
        traces.append(go.Scatter3d(
            x=[xs,xe], y=[ys,ye], z=[zs,ze], mode="lines",
            line=dict(color="gray", width=1, dash="dash"), name=name, showlegend=False, hoverinfo="skip",
        ))

for idx, (g, W, Re_W, Im_W) in enumerate(results):
    color = COLORS[idx % len(COLORS)]
    finite = np.isfinite(Re_W) & np.isfinite(Im_W)
    xf, rf, imf = x[finite], Re_W[finite], Im_W[finite]

    # 원래 그래프 기준으로 실수구간/복소구간을 나누고 싶다면 shift 빼기 전 수치로 봐야 하나,
    # 직관적으로 현재 그려진 결과의 허수부가 0(또는 z_shift) 근처인지로 판별하도록 보완 가능합니다.
    # 여기서는 최종 결과물의 허수부가 g["z_shift"]와 같은지를 기준으로 실수/복소 구간을 표기합니다.
    target_z = g.get("z_shift", 0.0)
    mask_real = np.abs(imf - target_z) < 1e-9

    if np.any(mask_real):
        traces.append(go.Scatter3d(
            x=xf[mask_real], y=rf[mask_real], z=imf[mask_real], mode="lines",
            line=dict(color=color, width=line_width), name=f"{g['label']} (실수구간)",
            hovertemplate=f"<b>{g['label']}</b><br>x=%{{x:.4f}}<br>Re=%{{y:.4f}}<br>Im=%{{z:.4f}}<extra></extra>",
        ))
    if np.any(~mask_real):
        traces.append(go.Scatter3d(
            x=xf[~mask_real], y=rf[~mask_real], z=imf[~mask_real], mode="lines",
            line=dict(color=color, width=line_width, dash="dot"), name=f"{g['label']} (복소구간)",
            hovertemplate=f"<b>{g['label']}</b><br>x=%{{x:.4f}}<br>Re=%{{y:.4f}}<br>Im=%{{z:.4f}}<extra></extra>",
        ))

fn_titles = " ,  ".join(g["label"] for g in active_graphs)
fig = go.Figure(data=traces)
fig.update_layout(
    height=660, margin=dict(l=0, r=0, t=50, b=0),
    title=dict(text=f"{fn_titles}", x=0.5, xanchor="center", font=dict(size=14)),
    scene=dict(
        xaxis=dict(title="x (실수 입력)"), yaxis=dict(title="y = Re(f(x))"), zaxis=dict(title="z = Im(f(x))"),
        bgcolor="rgba(14,17,23,1)", camera=dict(eye=dict(x=1.8, y=1.4, z=1.0)), aspectmode="auto",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(bgcolor="rgba(30,30,30,0.7)", font=dict(color="white"), bordercolor="#444", borderwidth=1),
)

st.plotly_chart(fig, use_container_width=True)

# ── 특정 x값 계산 ──────────────────────────────────────────────
st.divider()
st.subheader("특정 x 값에서 함수값 확인")
px = st.number_input("x =", value=2.0, step=0.1, format="%.4f", key="px_calc")
cols = st.columns(max(len(active_graphs), 1))
for i, (g, W, Re_W, Im_W) in enumerate(results):
    xs = g.get("x_shift", 0.0)
    ys = g.get("y_shift", 0.0)
    zs = g.get("z_shift", 0.0)
    pw = parse_function(g["fn_type"], g["inner"], g["p"], g["q"], np.array([px]), xs, ys, zs)[0]
    with cols[i % len(cols)]:
        color = COLORS[i % len(COLORS)]
        st.markdown(f'<span style="color:{color};font-weight:600">{g["label"]}</span>', unsafe_allow_html=True)
        st.metric("Re(f)", f"{np.real(pw):.6f}")
        st.metric("Im(f)", f"{np.imag(pw):.6f}")
        st.metric("|f|",   f"{abs(pw):.6f}")
        st.caption(f"arg = {np.degrees(np.angle(pw)):.3f}°")
