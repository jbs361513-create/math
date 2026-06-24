"""
복소수 무리함수 3D 시각화 - Streamlit 버전
실행: streamlit run math.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="복소수 무리함수 3D 시각화",
    page_icon="🔢",
    layout="wide",
)

st.title("복소수 무리함수 3D 시각화")
st.markdown(
    "입력 z = x + iy 에 대해 f(z) 의 **실수부(Re)**, **허수부(Im)**, "
    "**절댓값|f|** 곡면을 3D로 표시합니다."
)

# ── 무리함수 정의 ─────────────────────────────────────────────
def f_sqrt(z):
    return np.sqrt(z.astype(complex))

def f_cbrt(z):
    r = np.abs(z)
    th = np.angle(z)
    return (r ** (1 / 3)) * np.exp(1j * th / 3)

def f_pow23(z):
    r = np.abs(z)
    th = np.angle(z)
    return (r ** (2 / 3)) * np.exp(1j * 2 * th / 3)

def f_sqrt_x2iy(z):
    x, y = np.real(z), np.imag(z)
    return np.sqrt(x ** 2 + 1j * y)

def f_sqrt_xiiy2(z):
    x, y = np.real(z), np.imag(z)
    return np.sqrt(x + 1j * y ** 2)

def f_pow_half_i(z):
    """z^(1/2 + i) = e^((1/2+i)·ln z)"""
    z = z.astype(complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnz = np.log(np.where(z == 0, np.nan + 0j, z))
    return np.exp((0.5 + 1j) * lnz)

FUNCTIONS = {
    "√z  (주 제곱근)":    f_sqrt,
    "∛z  (세제곱근)":     f_cbrt,
    "z^(2/3)":            f_pow23,
    "√(x² + iy)":         f_sqrt_x2iy,
    "√(x + iy²)":         f_sqrt_xiiy2,
    "z^(½+i)":            f_pow_half_i,
}

# ── 사이드바 컨트롤 ───────────────────────────────────────────
with st.sidebar:
    st.header("설정")

    fn_name = st.radio("함수 선택", list(FUNCTIONS.keys()))

    st.divider()

    rng = st.slider("x, y 범위 (±)", 1.0, 8.0, 3.0, 0.5)
    n_pts = st.slider("격자 해상도", 40, 150, 80, 10,
                      help="높을수록 정밀하지만 느려집니다")

    st.divider()

    show_re  = st.checkbox("Re(f)  실수부 곡면 표시",  value=True)
    show_im  = st.checkbox("Im(f)  허수부 곡면 표시",  value=True)
    show_abs = st.checkbox("|f(z)| 절댓값 곡면 표시", value=False)
    show_zero_plane = st.checkbox("z = 0 기준면 표시", value=True)

    st.divider()
    opacity = st.slider("곡면 투명도", 0.1, 1.0, 0.75, 0.05)

    colorscale_re  = st.selectbox("Re 색상맵",  ["Blues",  "Viridis", "Plasma", "Cividis"], index=0)
    colorscale_im  = st.selectbox("Im 색상맵",  ["Oranges","RdBu",    "Hot",    "Magma"],   index=0)
    colorscale_abs = st.selectbox("|f| 색상맵", ["Greens", "Turbo",   "YlOrRd", "Electric"],index=0)

# ── 계산 ─────────────────────────────────────────────────────
@st.cache_data(max_entries=32)
def compute(fn_name, rng, n_pts):
    x = np.linspace(-rng, rng, n_pts)
    y = np.linspace(-rng, rng, n_pts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    W = FUNCTIONS[fn_name](Z)
    Re_W = np.real(W)
    Im_W = np.imag(W)
    Abs_W = np.abs(W)
    # NaN / Inf 클리핑
    clip = rng * 3
    for arr in (Re_W, Im_W, Abs_W):
        arr[~np.isfinite(arr)] = np.nan
        np.clip(arr, -clip, clip, out=arr)
    return X, Y, Re_W, Im_W, Abs_W

X, Y, Re_W, Im_W, Abs_W = compute(fn_name, rng, n_pts)

# ── Plotly 3D 그래프 ──────────────────────────────────────────
traces = []

if show_re:
    traces.append(go.Surface(
        x=X, y=Y, z=Re_W,
        surfacecolor=Re_W,
        colorscale=colorscale_re,
        opacity=opacity,
        name="Re(f)",
        showscale=True,
        colorbar=dict(x=1.0, title="Re(f)", thickness=14, len=0.5, y=0.8),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>Re(f)=%{z:.4f}<extra>Re(f)</extra>",
    ))

if show_im:
    traces.append(go.Surface(
        x=X, y=Y, z=Im_W,
        surfacecolor=Im_W,
        colorscale=colorscale_im,
        opacity=opacity,
        name="Im(f)",
        showscale=True,
        colorbar=dict(x=1.08, title="Im(f)", thickness=14, len=0.5, y=0.25),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>Im(f)=%{z:.4f}<extra>Im(f)</extra>",
    ))

if show_abs:
    traces.append(go.Surface(
        x=X, y=Y, z=Abs_W,
        surfacecolor=Abs_W,
        colorscale=colorscale_abs,
        opacity=opacity * 0.7,
        name="|f(z)|",
        showscale=True,
        colorbar=dict(x=1.16, title="|f|", thickness=14, len=0.5, y=0.0),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>|f|=%{z:.4f}<extra>|f(z)|</extra>",
    ))

if show_zero_plane:
    traces.append(go.Surface(
        x=X, y=Y,
        z=np.zeros_like(X),
        colorscale=[[0, "rgba(200,200,200,0.15)"], [1, "rgba(200,200,200,0.15)"]],
        showscale=False,
        opacity=0.25,
        name="z=0 기준면",
        hoverinfo="skip",
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    height=680,
    margin=dict(l=0, r=0, t=40, b=0),
    title=dict(
        text=f"f(z) = {fn_name} &nbsp;&nbsp;|&nbsp;&nbsp; z = x + iy",
        x=0.5, xanchor="center", font=dict(size=15)
    ),
    scene=dict(
        xaxis=dict(title="x  (Re 입력)", gridcolor="#333", showbackground=False),
        yaxis=dict(title="y  (Im 입력)", gridcolor="#333", showbackground=False),
        zaxis=dict(title="f(z) 값",      gridcolor="#333", showbackground=False),
        bgcolor="rgba(14,17,23,1)",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)", font=dict(color="white")),
)

st.plotly_chart(fig, use_container_width=True)

# ── 점 입력 → 함수값 계산 ─────────────────────────────────────
st.divider()
st.subheader("특정 점에서 함수값 계산")

col1, col2 = st.columns(2)
with col1:
    px = st.number_input("x (실수부)", value=1.0, step=0.1, format="%.3f")
with col2:
    py = st.number_input("y (허수부)", value=1.0, step=0.1, format="%.3f")

pz = px + 1j * py
pw = FUNCTIONS[fn_name](np.array([[pz]]))[0, 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("입력 z",      f"{px:.3f} + {py:.3f}i")
c2.metric("Re(f(z))",   f"{np.real(pw):.6f}")
c3.metric("Im(f(z))",   f"{np.imag(pw):.6f}")
c4.metric("|f(z)|",     f"{abs(pw):.6f}")

arg_deg = np.degrees(np.angle(pw))
st.caption(f"극형식:  |f| = {abs(pw):.6f},  arg(f) = {arg_deg:.4f}°")

# ── 함수 설명 ─────────────────────────────────────────────────
st.divider()
with st.expander("함수 설명 및 수학적 배경"):
    st.markdown("""
| 함수 | 정의 | 특이점 |
|------|------|--------|
| **√z** | 주치 제곱근: r^½ · e^(iθ/2) | z=0, 음의 실수축(분지컷) |
| **∛z** | 주치 세제곱근: r^⅓ · e^(iθ/3) | z=0, 분지컷 |
| **z^(2/3)** | r^(2/3) · e^(2iθ/3) | z=0 |
| **√(x²+iy)** | √(Re²+i·Im) | y=0, x=0 근방 |
| **√(x+iy²)** | √(Re+i·Im²) | 실수부 음수 구간 |
| **z^(½+i)** | e^((½+i)·ln z) | z=0, 분지컷 |

- **파란 곡면** → Re(f(z)) 실수부
- **주황 곡면** → Im(f(z)) 허수부  
- **색상** → 해당 곡면의 함수값 크기
- 음의 실수축 근방에서 불연속(분지컷)이 나타납니다.
    """)
