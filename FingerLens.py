import cv2
import mediapipe as mp
import numpy as np
import time

def order_points(pts):
    pts = np.array(pts)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype="int32")

def run_inverted_colours_mode(frame, hands, mp_hands, mp_drawing):
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    output = frame.copy()
    if result.multi_hand_landmarks:
        all_points = []
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))
            all_points.extend([thumb_coords, index_coords])
        if len(all_points) == 2:
            x1, y1 = all_points[0]
            x2, y2 = all_points[1]
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            window = frame[min_y:max_y, min_x:max_x]
            if window.size > 0:
                inverted = cv2.bitwise_not(window)
                output[min_y:max_y, min_x:max_x] = inverted
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif len(all_points) == 4:
            ordered_pts = order_points(all_points)
            pts = ordered_pts.reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            inverted = cv2.bitwise_not(frame)
            inverted_region = cv2.bitwise_and(inverted, inverted, mask=mask)
            normal_region = cv2.bitwise_and(output, output, mask=cv2.bitwise_not(mask))
            output = cv2.add(normal_region, inverted_region)
            cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return output

def run_main_mode(frame, hands, mp_hands, mp_draw):
    h, w, c = frame.shape
    img = frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    black_screen = np.zeros_like(img)
    left_coords, right_coords = None, None
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            x1, y1 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
            x2, y2 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
            if label == "Left":
                left_coords = [(x1, y1), (x2, y2)]
            else:
                right_coords = [(x1, y1), (x2, y2)]
            mp_draw.draw_landmarks(black_screen, handLms, mp_hands.HAND_CONNECTIONS)
        if left_coords and not right_coords:
            (x1, y1), (x2, y2) = left_coords
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if abs(x_max - x_min) > 20 and abs(y_max - y_min) > 20:
                window = img[y_min:y_max, x_min:x_max]
                black_screen[y_min:y_max, x_min:x_max] = window
                cv2.rectangle(black_screen, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        elif right_coords and not left_coords:
            (x1, y1), (x2, y2) = right_coords
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if abs(x_max - x_min) > 20 and abs(y_max - y_min) > 20:
                window = img[y_min:y_max, x_min:x_max]
                black_screen[y_min:y_max, x_min:x_max] = window
                cv2.rectangle(black_screen, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        elif left_coords and right_coords:
            pts = np.array([left_coords[0], left_coords[1], right_coords[1], right_coords[0]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(black_screen, [pts], True, (0, 255, 0), 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            region = cv2.bitwise_and(img, img, mask=mask)
            black_screen = cv2.add(black_screen, region)
    return black_screen

def draw_mode_buttons(img, selected_mode):
    h, w, _ = img.shape
    button_radius = int(min(w, h) * 0.06)
    y = int(h * 0.92)
    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    color = (180, 0, 180)
    highlight = (255, 0, 255)
    thickness = -1
    cv2.circle(img, (x1, y), button_radius, highlight if selected_mode==1 else color, thickness)
    cv2.circle(img, (x2, y), button_radius, highlight if selected_mode==2 else color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = button_radius/40.0
    font_thickness = 3
    text_color = (255,255,255)
    text1 = "1"
    text2 = "2"
    (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale, font_thickness)
    (tw2, th2), _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
    cv2.putText(img, text1, (x1-tw1//2, y+th1//2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x2-tw2//2, y+th2//2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return (x1, y, button_radius), (x2, y, button_radius)

def detect_pinch_and_button(results, w, h, button1, button2):
    if not results.multi_hand_landmarks:
        return None
    for handLms in results.multi_hand_landmarks:
        thumb_tip = handLms.landmark[4]
        index_tip = handLms.landmark[8]
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
        dist = np.hypot(x2-x1, y2-y1)
        pinch_thresh = 40
        if dist < pinch_thresh:
            pinch_x = int((x1 + x2) / 2)
            pinch_y = int((y1 + y2) / 2)
            bx1, by1, br1 = button1
            bx2, by2, br2 = button2
            if (pinch_x - bx1)**2 + (pinch_y - by1)**2 < br1**2:
                return 1
            if (pinch_x - bx2)**2 + (pinch_y - by2)**2 < br2**2:
                return 2
    return None

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, 
                           min_detection_confidence=0.7, 
                           min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    selected_mode = 1
    last_switch_time = 0
    switch_cooldown = 0.7
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        button1, button2 = draw_mode_buttons(img.copy(), selected_mode)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        pinch_option = detect_pinch_and_button(results, w, h, button1, button2)
        now = time.time()
        if pinch_option is not None and pinch_option != selected_mode and (now - last_switch_time) > switch_cooldown:
            selected_mode = pinch_option
            last_switch_time = now
        if selected_mode == 1:
            output = run_main_mode(img, hands, mp_hands, mp_draw)
        else:
            output = run_inverted_colours_mode(img, hands, mp_hands, mp_draw)
        draw_mode_buttons(output, selected_mode)
        cv2.imshow("Finger Window", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
