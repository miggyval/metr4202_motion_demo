import numpy as np
import cv2 as cv
import math

SIZE = 800
CIRC_SIZE = 30
xmin, xmax = -1, 1
ymin, ymax = -1, 1

GRID_SIZE = 400
v_min, v_max = 0.0, 1.0
w_min, w_max = -2.0, 2.0
goals = []
enable_samples = False
dist_type = "gaussian"

FONT_SIZE = 0.8
FONT_THICKNESS = 1
t_sim = 0.0

BG_COLOR = (0, 0, 0)
FG_COLOR = (255, 255, 255)

alpha_select = 0
alpha_exp = [-3, -3, -3, -3]
alpha_val = [1, 1, 1, 1]
alpha_vel = [0] * 4
alpha_odom = [0] * 4
for alpha_select in range(4):
    alpha_odom[alpha_select] = (10 ** alpha_exp[alpha_select]) * alpha_val[alpha_select]
    alpha_vel[alpha_select] = (10 ** alpha_exp[alpha_select]) * alpha_val[alpha_select]

mouse_x_pdf, mouse_y_pdf = SIZE // 2, SIZE // 2

def p2c(px, py=None):
    if py is None:
        return px / SIZE * (xmax - xmin)
    return (
        px / SIZE * (xmax - xmin) + xmin,
        (SIZE - 1 - py) / SIZE * (ymax - ymin) + ymin
    )

def c2p(x, y=None):
    if y is None:
        return int(SIZE * x / (xmax - xmin))
    return (
        int(SIZE * (x - xmin) / (xmax - xmin)),
        int(SIZE * (1 - (y - ymin) / (ymax - ymin)))
    )

def sample_noise(distribution, mean, sigma, n_samples):
    if distribution == "gaussian":
        return np.random.normal(mean, sigma, n_samples)
    elif distribution == "uniform":
        half_range = np.sqrt(12) * sigma / 2
        return np.random.uniform(mean - half_range, mean + half_range, n_samples)
    elif distribution == "triangular":
        sigma += 1e-6
        half_range = np.sqrt(6) * sigma
        return np.random.triangular(mean - half_range, mean, mean + half_range, n_samples)
    else:
        raise ValueError("Unknown distribution: choose 'gaussian', 'uniform', or 'triangular'")

def base_img():
    return np.ones((SIZE, SIZE, 3), dtype=np.uint8) * np.array(BG_COLOR, dtype=np.uint8)

def ctrl_base_img():
    return np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) * np.array(BG_COLOR, dtype=np.uint8)

def text(img, s, xy):
    cv.putText(img, s, xy, cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FG_COLOR, FONT_THICKNESS, cv.LINE_AA)

def draw_robot(img, center_px, theta):
    cv.circle(img, center_px, CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)
    cv.line(img, center_px, (center_px[0] + int(CIRC_SIZE * np.cos(theta)), center_px[1] - int(CIRC_SIZE * np.sin(theta))), FG_COLOR, 3, cv.LINE_AA)

def hue_color(theta):
    hue = int(((theta % (2 * math.pi)) / (2 * math.pi)) * 179)
    return cv.cvtColor(np.uint8([[[hue, 255, 255]]]), cv.COLOR_HSV2BGR)[0, 0].tolist()

def draw_samples_points(img, points, colors, enable):
    if not enable:
        return
    for (px, py), color in zip(points, colors):
        if 0 <= px < SIZE and 0 <= py < SIZE:
            img[py, px] = color

def arc_points(v_cmd, w_cmd, theta0, dt):
    if abs(w_cmd) > 1e-6:
        arc_x = -(v_cmd / w_cmd) * math.sin(theta0) + (v_cmd / w_cmd) * math.sin(theta0 + w_cmd * dt)
        arc_y = (v_cmd / w_cmd) * math.cos(theta0) - (v_cmd / w_cmd) * math.cos(theta0 + w_cmd * dt)
        return arc_x, arc_y
    else:
        return v_cmd * dt * math.cos(theta0), v_cmd * dt * math.sin(theta0)

def draw_arc(img, v_cmd, w_cmd, theta0, dt):
    if abs(w_cmd) <= 1e-6:
        cv.line(img, c2p(0, 0), c2p(v_cmd * dt * math.cos(theta0), v_cmd * dt * math.sin(theta0)), (128, 128, 128), 2)
        return
    arc_length = abs(v_cmd) * dt
    ppm = 200
    n = 2 * max(10, int(ppm * arc_length))
    grey = (230, 230, 230) if BG_COLOR == (0, 0, 0) else (128, 128, 128)
    for t in np.linspace(0, dt, n):
        ax = -(v_cmd / w_cmd) * math.sin(theta0) + (v_cmd / w_cmd) * math.sin(theta0 + w_cmd * t)
        ay = (v_cmd / w_cmd) * math.cos(theta0) - (v_cmd / w_cmd) * math.cos(theta0 + w_cmd * t)
        px, py = c2p(ax, ay)
        if 0 <= px < SIZE and 0 <= py < SIZE:
            cv.circle(img, (px, py), 1, grey, -1, cv.LINE_AA)

def draw_alpha_labels(img, alphas, alpha_select, y0=150):
    if not enable_samples:
        return
    for i, a in enumerate(alphas):
        cv.putText(img, f"alpha{i+1} = {a:.4f}", (20, y0 + 30 * i), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FG_COLOR, 2 if alpha_select == i else FONT_THICKNESS, cv.LINE_AA)

dt = 1.0
theta0 = 0.0
n_samples = 10000
mode = 1

mouse_x_ctrl, mouse_y_ctrl = GRID_SIZE // 2, GRID_SIZE // 2

def on_mouse_control(event, x, y, flags, param):
    global mouse_x_ctrl, mouse_y_ctrl
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x_ctrl, mouse_y_ctrl = x, y

def on_mouse_motion(event, x, y, flags, param):
    global mouse_x_pdf, mouse_y_pdf, goals
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x_pdf, mouse_y_pdf = x, y
    if event == cv.EVENT_LBUTTONDOWN and mode == 2:
        goals.append((x, y))

cv.namedWindow("Motion PDF")
cv.namedWindow("Control Space")
cv.setMouseCallback("Control Space", on_mouse_control)
cv.setMouseCallback("Motion PDF", on_mouse_motion)

delta_t = 0.01

print("""
Interactive Motion Model Visualizer
-----------------------------------
- Two windows open: 
    * "Motion PDF" shows sampled motion outcomes.
    * "Control Space" is where you change steering/angles.

Modes:
  m  : switch between Velocity model and Odometry model
  e  : toggle drawing of sample points
  d  : toggle dark / light background
  1-4: select which alpha parameter to adjust
  =  : increase selected alpha
  -  : decrease selected alpha
  5-6: select distribution (gaussian/triangular/uniform)
  ESC: quit

Mouse:
  - In Control Space: X = velocity/angular velocity (depending on mode)
  - In Motion PDF   : move crosshair; in odometry mode, move to set goal
""")

while True:
    w_cmd = w_min + (mouse_x_ctrl / GRID_SIZE) * (w_max - w_min)
    v_cmd = v_max - (mouse_y_ctrl / GRID_SIZE) * (v_max - v_min)

    t_sim += delta_t
    t_sim = t_sim % (dt + 0.4)

    img = base_img()
    text(img, f"Mode: {'Velocity' if mode == 0 else 'Odometry'}", (20, 30))

    origin_px = c2p(0, 0)
    draw_robot(img, origin_px, theta0)

    if mode == 0:
        text(img, f"v={v_cmd:.2f}, w={w_cmd:.2f}", (20, 70))
        text(img, f"th0={theta0:.2f}, dt={dt:.2f}", (20, 100))

        sigma_v = np.sqrt(alpha_vel[0] * v_cmd**2 + alpha_vel[1] * w_cmd**2)
        sigma_w = np.sqrt(alpha_vel[2] * v_cmd**2 + alpha_vel[3] * w_cmd**2)

        v_samples = sample_noise(dist_type, v_cmd, sigma_v, n_samples)
        w_samples = sample_noise(dist_type, w_cmd, sigma_w, n_samples)

        start_dx = (mouse_x_pdf / SIZE) * (xmax - xmin) + xmin
        start_dy = ((SIZE - mouse_y_pdf) / SIZE) * (ymax - ymin) + ymin
        theta0 = math.atan2(start_dy, start_dx)

        pts, cols = [], []
        for v, w in zip(v_samples, w_samples):
            if abs(w) > 1e-6:
                dx = -(v / w) * math.sin(theta0) + (v / w) * math.sin(theta0 + w * dt)
                dy = (v / w) * math.cos(theta0) - (v / w) * math.cos(theta0 + w * dt)
            else:
                dx = v * dt * math.cos(theta0)
                dy = v * dt * math.sin(theta0)
            final_theta = theta0 + w_cmd * dt
            px, py = c2p(dx, dy)
            pts.append((px, py))
            cols.append(hue_color(final_theta))
        draw_samples_points(img, pts, cols, enable_samples)

        draw_arc(img, v_cmd, w_cmd, theta0, dt)
        end_x, end_y = arc_points(v_cmd, w_cmd, theta0, dt)

        cv.circle(img, c2p(0, 0), 6, FG_COLOR, -1)
        cv.circle(img, c2p(end_x, end_y), 6, FG_COLOR, -1)

        end_px, end_py = c2p(end_x, end_y)
        draw_robot(img, (end_px, end_py), theta0 + w_cmd * dt)

        if t_sim < dt:
            if w_cmd != 0:
                cx = -(v_cmd / w_cmd) * math.sin(theta0) + (v_cmd / w_cmd) * math.sin(theta0 + w_cmd * t_sim)
                cy = (v_cmd / w_cmd) * math.cos(theta0) - (v_cmd / w_cmd) * math.cos(theta0 + w_cmd * t_sim)
            else:
                cx = v_cmd * math.cos(theta0) * t_sim
                cy = v_cmd * math.sin(theta0) * t_sim
            sim_px, sim_py = c2p(cx, cy)
            draw_robot(img, (sim_px, sim_py), theta0 + w_cmd * t_sim)

        draw_alpha_labels(img, alpha_vel, alpha_select)

    elif mode == 1:
        goal_x = (mouse_x_pdf / SIZE) * (xmax - xmin) + xmin
        goal_y = ((SIZE - mouse_y_pdf) / SIZE) * (ymax - ymin) + ymin
        delta_trans = math.sqrt(goal_x**2 + goal_y**2)

        max_rot = math.pi
        theta0 = ((mouse_x_ctrl / GRID_SIZE) * 2 - 1) * max_rot
        delta_rot1 = math.atan2(goal_y, goal_x) - theta0
        delta_rot1 = math.atan2(np.sin(delta_rot1), np.cos(delta_rot1))
        delta_rot2 = ((GRID_SIZE - mouse_y_ctrl) / GRID_SIZE * 2 - 1) * max_rot

        text(img, f"th0={theta0:.2f}, goal=({goal_x:.2f},{goal_y:.2f})", (20, 70))
        text(img, f"rot1={delta_rot1:.2f}, trans={delta_trans:.2f} rot2={delta_rot2:.2f}", (20, 100))

        # Linear
        #sigma_rot1 = alpha_odom[0] * abs(delta_rot1) + alpha_odom[1] * abs(delta_trans)
        #sigma_trans = alpha_odom[2] * abs(delta_trans) + alpha_odom[3] * (abs(delta_rot1) + 0 * abs(delta_rot2))
        #sigma_rot2 = alpha_odom[0] * abs(delta_rot2) + alpha_odom[1] * abs(delta_trans)
        
        # Square
        sigma_rot1 = np.sqrt(alpha_odom[0] * delta_rot1 ** 2 + alpha_odom[1] * delta_trans ** 2)
        sigma_trans = np.sqrt(alpha_odom[2] * delta_trans ** 2 + alpha_odom[3] * (delta_rot1 ** 2 + delta_rot2 ** 2))
        sigma_rot2 = np.sqrt(alpha_odom[0] * delta_rot2 ** 2 + alpha_odom[1] * delta_trans ** 2)

        cv.line(img, c2p(0, 0), c2p(goal_x, goal_y), (128, 128, 128), 2, cv.LINE_AA)

        draw_alpha_labels(img, alpha_odom, alpha_select)

        rot1_samples = sample_noise(dist_type, delta_rot1, sigma_rot1, n_samples)
        trans_samples = sample_noise(dist_type, delta_trans, sigma_trans, n_samples)
        rot2_samples = sample_noise(dist_type, delta_rot2, sigma_rot2, n_samples)

        pts, cols = [], []
        for r1, t, r2 in zip(rot1_samples, trans_samples, rot2_samples):
            theta_mid = theta0 + r1
            dx = t * math.cos(theta_mid)
            dy = t * math.sin(theta_mid)
            final_theta = theta_mid + r2
            px, py = c2p(dx, dy)
            pts.append((px, py))
            cols.append(hue_color(final_theta))
        draw_samples_points(img, pts, cols, enable_samples)

        cv.circle(img, c2p(0, 0), 6, FG_COLOR, -1)
        cv.circle(img, c2p(goal_x, goal_y), 6, FG_COLOR, -1)
        cv.circle(img, c2p(0, 0), CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)
        cv.circle(img, c2p(goal_x, goal_y), CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)

        if t_sim < dt:
            cv.circle(img, c2p(goal_x * t_sim / dt, goal_y * t_sim / dt), CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)
            cv.line(img, c2p(goal_x * t_sim / dt, goal_y * t_sim / dt), c2p(goal_x * t_sim / dt + p2c(CIRC_SIZE) * np.cos(theta0 + delta_rot1), goal_y * t_sim / dt + p2c(CIRC_SIZE) * np.sin(theta0 + delta_rot1)), FG_COLOR, 3, cv.LINE_AA)
        if t_sim >= dt and t_sim < dt + 0.2:
            cv.circle(img, c2p(goal_x, goal_y), CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)
            cv.line(img, c2p(goal_x, goal_y), c2p(goal_x + p2c(CIRC_SIZE) * np.cos(theta0 + delta_rot1 + (t_sim - dt) / 0.2 * delta_rot2), goal_y + p2c(CIRC_SIZE) * np.sin(theta0 + delta_rot1 + (t_sim - dt) / 0.2 * delta_rot2)), FG_COLOR, 3, cv.LINE_AA)
        elif t_sim >= dt + 0.2:
            cv.circle(img, c2p(0, 0), CIRC_SIZE, FG_COLOR, 1, cv.LINE_AA)
            cv.line(img, c2p(0, 0), c2p(p2c(CIRC_SIZE) * np.cos(theta0 + (t_sim - dt - 0.2) / 0.2 * delta_rot1), p2c(CIRC_SIZE) * np.sin(theta0 + (t_sim - dt - 0.2) / 0.2 * delta_rot1)), FG_COLOR, 3, cv.LINE_AA)

        cv.line(img, c2p(goal_x, goal_y), c2p(goal_x + p2c(CIRC_SIZE) * np.cos(theta0 + delta_rot1 + delta_rot2), goal_y + p2c(CIRC_SIZE) * np.sin(theta0 + delta_rot1 + delta_rot2)), FG_COLOR, 3, cv.LINE_AA)

    else:
        delta_rot1_arr = [None] * len(goals)
        delta_rot2_arr = [None] * len(goals)
        delta_trans_arr = [None] * len(goals)
        theta_arr = [None] * len(goals)
        for i in range(len(goals)):
            if i == 0:
                x1, y1 = (SIZE // 2, SIZE // 2)
                theta1 = 0
            else:
                x1, y1 = goals[i - 1]
            theta2 = theta1
            x2, y2 = goals[i]
            dx, dy = (x2 - x1, y2 - y1)
            theta1 = np.arctan2(dy, dx)
            dtheta = (theta2 - theta1 + 2.0 * np.pi) % (2.0 * np.pi)
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv.LINE_AA)
            cv.putText(img, str(np.round(np.rad2deg(dtheta), 1)), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 50), FONT_THICKNESS, cv.LINE_AA)
            cv.putText(img, str(np.round(np.hypot(dx, dy), 1)), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 0), FONT_THICKNESS, cv.LINE_AA)
            delta_rot1_arr[i] = dtheta
            delta_rot2_arr[i] = 0
            delta_trans_arr[i] = np.hypot(dx, dy)
            sigma_rot1 = alpha_odom[0] * abs(delta_rot1) + alpha_odom[1] * abs(delta_trans)
            sigma_trans = alpha_odom[2] * abs(delta_trans) + alpha_odom[3] * (abs(delta_rot1) + 0 * abs(delta_rot2))
            sigma_rot2 = alpha_odom[0] * abs(delta_rot2) + alpha_odom[1] * abs(delta_trans)
            rot1_samples = sample_noise(dist_type, 0, sigma_rot1, n_samples // 10)
            trans_samples = sample_noise(dist_type, 0, sigma_trans, n_samples // 10)
            rot2_samples = sample_noise(dist_type, 0, sigma_rot2, n_samples // 10)
        for i in range(len(goals)):
            for r1, t, r2 in zip(rot1_samples, trans_samples, rot2_samples):
                final_x = goals[i][0] + r1
                final_y = goals[i][1] + r1
                final_theta = goals
                theta_mid = theta0 + r1
                dx = t * math.cos(theta_mid)
                dy = t * math.sin(theta_mid)
                final_theta = theta_mid + r2

    ctrl_img = ctrl_base_img()
    step = GRID_SIZE // 4
    for i in range(0, GRID_SIZE + 1, step):
        cv.line(ctrl_img, (0, i), (GRID_SIZE, i), (200, 200, 200), 1, cv.LINE_AA)
        cv.line(ctrl_img, (i, 0), (i, GRID_SIZE), (200, 200, 200), 1, cv.LINE_AA)
    if mode == 1:
        text(ctrl_img, "theta0 (X) | rot2 (Y)", (50, 40))
    elif mode == 0:
        text(ctrl_img, "v (X) | w (Y)", (120, 40))
    cv.circle(ctrl_img, (mouse_x_ctrl, mouse_y_ctrl), 6, (0, 0, 255), -1)

    cv.imshow("Motion PDF", img)
    cv.imshow("Control Space", ctrl_img)

    key = cv.waitKey(10) & 0xFF
    if key == 27:
        break
    elif key == ord('5'):
        dist_type = "gaussian"
    elif key == ord('6'):
        dist_type = "triangular"
    elif key == ord('7'):
        dist_type = "uniform"
    elif key == ord('m'):
        mode = (mode + 1) % 2
    elif key == ord('d'):
        if BG_COLOR == (255, 255, 255):
            BG_COLOR = (0, 0, 0)
            FG_COLOR = (255, 255, 255)
        else:
            BG_COLOR = (255, 255, 255)
            FG_COLOR = (0, 0, 0)
    elif key == ord('e'):
        enable_samples = not enable_samples
    elif key == ord('1'):
        alpha_select = 0
    elif key == ord('2'):
        alpha_select = 1
    elif key == ord('3'):
        alpha_select = 2
    elif key == ord('4'):
        alpha_select = 3
    elif key == ord('='):
        alpha_val[alpha_select] += 1
        if alpha_val[alpha_select] == 10:
            alpha_val[alpha_select] = 1
            alpha_exp[alpha_select] += 1
    elif key == ord('-'):
        if alpha_exp[alpha_select] != -4 or alpha_val[alpha_select] != 1:
            alpha_val[alpha_select] -= 1
            if alpha_val[alpha_select] == 0:
                alpha_val[alpha_select] = 9
                alpha_exp[alpha_select] -= 1
    alpha_odom[alpha_select] = (10 ** alpha_exp[alpha_select]) * alpha_val[alpha_select]
    alpha_vel[alpha_select] = (10 ** alpha_exp[alpha_select]) * alpha_val[alpha_select]
cv.destroyAllWindows()
