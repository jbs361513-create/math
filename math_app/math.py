"""
복소수 무리함수 3D 시각화 - Streamlit 버전
x (실수 입력) → y = Re(f(x)), z = Im(f(x)) 3D 곡선
실행: streamlit run math.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="복소수 무리함수 시각화",
    page_icon="🔢",
    layout="wide",
)

st.title("복소수 무리함수 시각화")
st.markdown(
    "실수 **x** 를 입력하면 f(x) 의 실수부(y축)와 허수부(z축)를 3D 곡선으로 표시합니다."
)

# ── 무리함수 정의 ─────────────────────────────────────────────
def f_sqrt(x):
    """√x"""
    x = x.astype(complex)
    return np.sqrt(x)

def f_cbrt(x):
    """∛x (주 세제곱근)"""
    x = x.astype(complex)
    r = np.abs(x)
    th = np.angle(x)
    return (r ** (1/3)) * np.exp(1j * th / 3)

def f_pow23(x):
    """x^(2/3)"""
    x = x.astype(complex)
    r = np.abs(x)
    th = np.angle(x)
    return (r ** (2/3)) * np.exp(1j * 2 * th / 3)

def f_pow13(x):
    """x^(1/3) - 실수부만"""
    x = x.astype(complex)
    r = np.abs(x)
    th = np.angle(x)
    return (r ** (1/3)) * np.exp(1j * th / 3)

def f_sqrt_x2_1(x):
    """√(x² - 1)"""
    x = x.astype(complex)
    return np.sqrt(x**2 - 1)

def f_sqrt_neg_x2(x):
    """√(1 - x²)"""
    x = x.astype(complex)
    return np.sqrt(1 - x**2)

def f_pow_half_plus_i(x):
    """x^(1/2 + i) = e^((1/2+i)·ln x)"""
    x = x.astype(complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnx = np.log(np.where(x == 0, np.nan + 0j, x))
    return np.exp((0.5 + 1j) * lnx)

FUNCTIONS = {
    "√x":            f_sqrt,
    "∛x":            f_cbrt,
    "x^(2/3)":       f_pow23,
    "x^(1/3)":       f_pow13,
    "√(x² − 1)":    f_sqrt_x2_1,
    "√(1 − x²)":    f_sqrt_neg_x2,
    "x^(½+i)":       f_pow_half_plus_i,
}

# ── 사이드바 컨트롤 ───────────────────────────────────────────
with st.sidebar:
    st.header("설정")

    selected_fns = st.multiselect(
        "함수 선택 (여러 개 가능)",
        list(FUNCTIONS.keys()),
        default=["√x", "√(x² − 1)"],
    )

    st.divider()

    x_min = st.number_input("x 최솟값", value=-5.0, step=0.5, format="%.1f")
    x_max = st.number_input("x 최댓값", value=5.0,  step=0.5, format="%.1f")
    n_pts = st.slider("포인트 수", 200, 2000, 800, 100)

    st.divider()

    show_axes   = st.checkbox("x, y, z 축선 표시", value=True)
    show_re_plane = st.checkbox("Im=0 기준면 (xz평면)", value=True)
    show_im_plane = st.checkbox("Re=0 기준면 (xy평면)", value=True)

    st.divider()
    line_width = st.slider("선 굵기", 1, 8, 4)

# ── 색상 팔레트 ───────────────────────────────────────────────
COLORS = [
    "#3b82f6",  # blue
    "#f97316",  # orange
    "#22c55e",  # green
    "#a855f7",  # purple
    "#ef4444",  # red
    "#06b6d4",  # cyan
    "#eab308",  # yellow
]

# ── 계산 ─────────────────────────────────────────────────────
if x_min >= x_max:
    st.error("x 최솟값이 최댓값보다 작아야 합니다.")
    st.stop()

x = np.linspace(x_min, x_max, n_pts)

# ── 그래프 구성 ───────────────────────────────────────────────
traces = []

# 기준면
rng_re = 0
rng_im = 0
for fn_name in selected_fns:
    W = FUNCTIONS[fn_name](x)
    rng_re = max(rng_re, np.nanmax(np.abs(np.real(W))))
    rng_im = max(rng_im, np.nanmax(np.abs(np.imag(W))))

pad = 0.3
rng_re = rng_re + pad if rng_re > 0 else 1.0
rng_im = rng_im + pad if rng_im > 0 else 1.0

if show_re_plane:
    # Im = 0 평면 (xz 평면, y=0)
    xg = np.array([x_min, x_max, x_max, x_min])
    yg = np.array([0, 0, 0, 0])
    zg = np.array([-rng_im, -rng_im, rng_im, rng_im])
    traces.append(go.Mesh3d(
        x=xg, y=yg, z=zg,
        color="lightblue", opacity=0.08,
        name="Im=0 평면", showlegend=True,
        hoverinfo="skip",
    ))

if show_im_plane:
    # Re = 0 평면 (xy 평면, z=0)
    xg = np.array([x_min, x_max, x_max, x_min])
    yg = np.array([-rng_re, -rng_re, rng_re, rng_re])
    zg = np.array([0, 0, 0, 0])
    traces.append(go.Mesh3d(
        x=xg, y=yg, z=zg,
        color="lightyellow", opacity=0.08,
        name="Re=0 평면", showlegend=True,
        hoverinfo="skip",
    ))

# 축선
if show_axes:
    # x 축
    traces.append(go.Scatter3d(
        x=[x_min, x_max], y=[0, 0], z=[0, 0],
        mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        name="x축", showlegend=False, hoverinfo="skip",
    ))
    # y 축 (Re)
    traces.append(go.Scatter3d(
        x=[0, 0], y=[-rng_re, rng_re], z=[0, 0],
        mode="lines",
        line=dict(color="#3b82f6", width=1, dash="dash"),
        name="Re축", showlegend=False, hoverinfo="skip",
    ))
    # z 축 (Im)
    traces.append(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-rng_im, rng_im],
        mode="lines",
        line=dict(color="#f97316", width=1, dash="dash"),
        name="Im축", showlegend=False, hoverinfo="skip",
    ))

# 함수 곡선
for i, fn_name in enumerate(selected_fns):
    W = FUNCTIONS[fn_name](x)
    Re_W = np.real(W)
    Im_W = np.imag(W)

    color = COLORS[i % len(COLORS)]

    # 실수부가 있는 구간 (Im=0)
    mask_real = np.abs(Im_W) < 1e-10
    # 허수부가 있는 구간 (Re≠0 or Im≠0, 분기 구간)
    mask_complex = ~mask_real

    # 실수 구간 곡선
    if np.any(mask_real):
        xr, yr, zr = x[mask_real], Re_W[mask_real], Im_W[mask_real]
        traces.append(go.Scatter3d(
            x=xr, y=yr, z=zr,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=f"{fn_name}  (실수)",
            hovertemplate=(
                f"<b>{fn_name}</b><br>"
                "x = %{x:.4f}<br>"
                "Re(f) = %{y:.4f}<br>"
                "Im(f) = %{z:.4f}<extra></extra>"
            ),
        ))

    # 복소수 구간 곡선
    if np.any(mask_complex):
        xc, yc, zc = x[mask_complex], Re_W[mask_complex], Im_W[mask_complex]
        traces.append(go.Scatter3d(
            x=xc, y=yc, z=zc,
            mode="lines",
            line=dict(color=color, width=line_width, dash="dot"),
            name=f"{fn_name}  (복소)",
            hovertemplate=(
                f"<b>{fn_name}</b><br>"
                "x = %{x:.4f}<br>"
                "Re(f) = %{y:.4f}<br>"
                "Im(f) = %{z:.4f}<extra></extra>"
            ),
        ))

fig = go.Figure(data=traces)
fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, t=50, b=0),
    title=dict(
        text="f(x) 의 3D 궤적  —  x: 실수 입력 / y: Re(f) / z: Im(f)",
        x=0.5, xanchor="center", font=dict(size=14),
    ),
    scene=dict(
        xaxis=dict(title="x  (실수 입력)", showgrid=True, gridcolor="#333"),
        yaxis=dict(title="y = Re(f(x))", showgrid=True, gridcolor="#333"),
        zaxis=dict(title="z = Im(f(x))", showgrid=True, gridcolor="#333"),
        bgcolor="rgba(14,17,23,1)",
        camera=dict(eye=dict(x=1.8, y=1.4, z=1.0)),
        aspectmode="auto",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        bgcolor="rgba(30,30,30,0.7)",
        font=dict(color="white"),
        bordercolor="#444",
        borderwidth=1,
    ),
)

if not selected_fns:
    st.info("왼쪽 사이드바에서 함수를 선택하세요.")
else:
    st.plotly_chart(fig, use_container_width=True)

# ── 특정 x값에서 함수값 계산 ─────────────────────────────────
st.divider()
st.subheader("특정 x 값에서 함수값 계산")

px = st.number_input("x 입력", value=2.0, step=0.1, format="%.4f")

if selected_fns:
    cols = st.columns(len(selected_fns))
    for i, fn_name in enumerate(selected_fns):
        pw = FUNCTIONS[fn_name](np.array([px]))[0]
        re, im = np.real(pw), np.imag(pw)
        abv = abs(pw)
        arg = np.degrees(np.angle(pw))
        with cols[i]:
            st.markdown(f"**{fn_name}**")
            st.metric("Re(f)",  f"{re:.6f}")
            st.metric("Im(f)",  f"{im:.6f}")
            st.metric("|f|",    f"{abv:.6f}")
            st.caption(f"arg = {arg:.3f}°")

# ── 함수 설명 ─────────────────────────────────────────────────
st.divider()
with st.expander("함수 및 3D 축 설명"):
    st.markdown("""
**3D 축 의미**
| 축 | 의미 |
|----|------|
| **x축** | 실수 입력값 |
| **y축** | Re(f(x)) — 함수값의 실수부 |
| **z축** | Im(f(x)) — 함수값의 허수부 |

**실선**: 실수 구간 (Im = 0)  
**점선**: 복소수 구간 (Im ≠ 0, 분지컷 이후)

**함수 목록**
| 함수 | 복소수 발생 조건 |
|------|-----------------|
| √x | x < 0 |
| ∛x | x < 0 (위상 π/3) |
| x^(2/3) | x < 0 |
| x^(1/3) | x < 0 |
| √(x²−1) | −1 < x < 1 |
| √(1−x²) | x < −1 또는 x > 1 |
| x^(½+i) | x < 0 (항상 복소수) |
    """)
