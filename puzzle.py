import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

p_time = 0

capture_done = False
pinch_start_time = 0
HOLD_TIME = 1.5
PINCH_THRESHOLD = 50
wait_for_release_after_capture = True
left = right = top = bottom = 0
puzzle_img = None

smooth_x = None
smooth_y = None
SMOOTH = 0.35


def create_puzzle(img, rows=3, cols=3):
    h, w = img.shape[:2]
    piece_h = h // rows
    piece_w = w // cols

    img = img[:piece_h * rows, :piece_w * cols]

    pieces = []
    for r in range(rows):
        for c in range(cols):
            piece = img[r*piece_h:(r+1)*piece_h, c*piece_w:(c+1)*piece_w].copy()
            pieces.append(piece)

    board = list(range(rows * cols))
    random.shuffle(board)
    return pieces, board


def draw_board(frame, pieces, board, left, top, right, bottom,
               selected_tile, dragging, drag_tile, finger_x, finger_y,
               smooth_x=None, smooth_y=None):

    if solved:
        cv2.putText(frame, "You Win", (250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    if not pieces:
        return

    rows = cols = 3
    width  = right - left
    height = bottom - top
    tile_w = width  // cols
    tile_h = height // rows
    gap = 4

    if tile_w <= gap or tile_h <= gap:
        return

    for i, piece_index in enumerate(board):
        if dragging and i == drag_tile:
            continue 

        r, c = divmod(i, cols)
        x = left + c * tile_w
        y = top  + r * tile_h

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + tile_w - gap), \
                 min(frame.shape[0], y + tile_h - gap)
        if x2 <= x1 or y2 <= y1:
            continue

        tile = cv2.resize(pieces[piece_index], (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = (255, 255, 255)
        frame[y1:y2, x1:x2] = tile
        cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), (0, 220, 2550), 3)

        if selected_tile == i:
            cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), (0, 255, 0), 5)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 5)

    if dragging and drag_tile is not None and 0 <= drag_tile < len(board):
        fx = smooth_x if smooth_x is not None else finger_x
        fy = smooth_y if smooth_y is not None else finger_y
        if fx is None or fy is None:
            return

        piece_index = board[drag_tile] 
        x = int(fx) - tile_w // 2
        y = int(fy) - tile_h // 2

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + tile_w - gap), \
                 min(frame.shape[0], y + tile_h - gap)
        if x2 <= x1 or y2 <= y1:
            return

        tile = cv2.resize(pieces[piece_index], (x2 - x1, y2 - y1))

        overlay = frame.copy()
        overlay[y1:y2, x1:x2] = tile
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        frame[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]

        cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), (0, 220, 255), 3)


pieces = []
board = []
selected_tile = None
solved = False

selected_tile = None
last_pinch = False
solved = False
dragging = False
drag_tile = None


def get_tile(x, y, left, top, right, bottom):
    rows = 3
    cols = 3

    width = right - left
    height = bottom - top

    tile_w = width // cols
    tile_h = height // rows

    if not (left <= x <= right and top <= y <= bottom):
        return None

    col = (x - left) // tile_w
    row = (y - top) // tile_h

    if row >= rows:
        row = rows - 1

    if col >= cols:
        col = cols - 1

    return int(row * cols + col)


def swap_tiles(board, a, b):
    board[a], board[b] = board[b], board[a]


def check_win(board):
    return board == list(range(9))


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (700, 700))

    clean_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    points = []
    pinch_count = 0
    finger_x = None
    finger_y = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape

            thumb_x, thumb_y = 0, 0
            index_x, index_y = 0, 0

            for id, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                if id == 4:
                    thumb_x, thumb_y = cx, cy

                if id == 8:
                    index_x, index_y = cx, cy
                    finger_x = index_x
                    finger_y = index_y

            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if distance < PINCH_THRESHOLD:
                pinch_count += 1
                points.append((index_x, index_y))

                cv2.putText(
                    frame,
                    "PINCH ON",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    "PINCH OFF",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

        if not capture_done and pinch_count == 2 and len(points) == 2:
            if pinch_start_time == 0:
                pinch_start_time = time.time()

            left = min(points[0][0], points[1][0])
            right = max(points[0][0], points[1][0])
            top = min(points[0][1], points[1][1])
            bottom = max(points[0][1], points[1][1])

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

            elapsed = time.time() - pinch_start_time
            remaining = max(0, HOLD_TIME - elapsed)

            cv2.putText(
                frame,
                f"{remaining:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            if (
                elapsed >= HOLD_TIME
                and not capture_done
                and right - left > 100
                and bottom - top > 100
            ):
                captured = clean_frame[top:bottom, left:right].copy()

                pieces, board = create_puzzle(captured)
                capture_done = True
                solved = False

                dragging = False
                drag_tile = None
                last_pinch = False

                print("PUZZLE CAPTURED", len(pieces), board)

        else:
            pinch_start_time = 0

    if finger_x is not None:
        if smooth_x is None:
            smooth_x, smooth_y = finger_x, finger_y
        else:
            smooth_x = smooth_x + SMOOTH * (finger_x - smooth_x)
            smooth_y = smooth_y + SMOOTH * (finger_y - smooth_y)
    else:
        smooth_x = smooth_y = None

    if capture_done and finger_x is not None:
        draw_board(
            frame,
            pieces,
            board,
            left,
            top,
            right,
            bottom,
            selected_tile,
            dragging,
            drag_tile,
            finger_x,
            finger_y,
            smooth_x,
            smooth_y,
        )

        cv2.circle(frame, (finger_x, finger_y), 12, (0, 255, 0), cv2.FILLED)

    if wait_for_release_after_capture:
        if pinch_count == 0:
            wait_for_release_after_capture = False
            last_pinch = False

        else:
            cv2.putText(
                frame,
                "Release pinch to start game",
                (250, 950),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    else:
        tile = None

        if capture_done and finger_x is not None and finger_y is not None:
            tile = get_tile(
                finger_x,
                finger_y,
                left,
                top,
                right,
                bottom,
            )

        is_pinching = pinch_count > 0

        if is_pinching and not last_pinch and tile is not None:
            drag_tile = tile
            dragging = True

        if not is_pinching and last_pinch and dragging:
            drop_tile = tile

            if (
                drag_tile is not None
                and drop_tile is not None
                and drag_tile != drop_tile
            ):
                swap_tiles(board, drag_tile, drop_tile)
                solved = check_win(board)

            dragging = False
            drag_tile = None

        last_pinch = is_pinching

    c_time = time.time()
    fps = 1 / (c_time - p_time) if p_time != 0 else 0
    p_time = c_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 980),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Press R to Reset",
        (700, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Puzzle Game", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        capture_done = False
        pieces = []
        board = []
        selected_tile = []
        solved = False
        pinch_start_time = 0
        smooth_x = None
        smooth_y = None
        wait_for_release_after_capture = True

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()