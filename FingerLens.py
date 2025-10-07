import cv2
import mediapipe as mp
import numpy as np
import time

screenshot_thumbnails = []
dragging_index = None
countdown_active = False
countdown_start_time = 0
frame_points = []
bin_area = None

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

def is_pinky_raised(hand_landmarks, h, w):
    pinky_tip = hand_landmarks.landmark[20]
    pinky_base = hand_landmarks.landmark[17]
    return pinky_tip.y * h < pinky_base.y * h - 20

def point_in_rect(px, py, x1, y1, x2, y2):
    return x1 <= px <= x2 and y1 <= py <= y2

def draw_trash_icon(img, center_x, center_y, size):
    color = (255, 255, 255)
    thickness = max(2, size // 12)

    body_width = int(size * 0.6)
    body_height = int(size * 0.65)
    lid_width = int(size * 0.7)
    lid_height = int(size * 0.12)
    handle_width = int(size * 0.18)
    handle_height = int(size * 0.10)

    body_x1 = center_x - body_width // 2
    body_y1 = center_y - body_height // 2 + int(size * 0.05)
    body_x2 = center_x + body_width // 2
    body_y2 = center_y + body_height // 2
    cv2.rectangle(img, (body_x1, body_y1), (body_x2, body_y2), color, thickness)

    lid_y_bottom = body_y1 - thickness
    lid_y_top = lid_y_bottom - lid_height
    lid_x1 = center_x - lid_width // 2
    lid_x2 = center_x + lid_width // 2
    cv2.rectangle(img, (lid_x1, lid_y_top), (lid_x2, lid_y_bottom), color, thickness)

    handle_y_bottom = lid_y_top - thickness // 2
    handle_y_top = handle_y_bottom - handle_height
    handle_x1 = center_x - handle_width // 2
    handle_x2 = center_x + handle_width // 2
    cv2.rectangle(img, (handle_x1, handle_y_top), (handle_x2, handle_y_bottom), color, max(1, thickness // 2))

    slat_thickness = max(1, thickness // 2)
    for offset in (-body_width // 4, 0, body_width // 4):
        x = center_x + offset
        cv2.line(img, (x, body_y1 + thickness), (x, body_y2 - thickness), color, slat_thickness)

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

def run_screenshot_canvas_mode(frame, hands, mp_hands, mp_draw):
    global screenshot_thumbnails, dragging_index, countdown_active, countdown_start_time, frame_points, bin_area

    h, w, c = frame.shape
    output = frame.copy()

    bin_size = 80
    bin_x1, bin_y1 = w - bin_size, 10
    bin_x2, bin_y2 = w - 10, 10 + bin_size
    bin_area = (bin_x1, bin_y1, bin_x2, bin_y2)
    cv2.rectangle(output, (bin_x1, bin_y1), (bin_x2, bin_y2), (0, 0, 255), -1)
    icon_center_x = (bin_x1 + bin_x2) // 2
    icon_center_y = (bin_y1 + bin_y2) // 2 + 5
    draw_trash_icon(output, icon_center_x, icon_center_y, size=60)

    frame_points = []
    current_pinches = []
    pinky_raised = False

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            pinky_tip = hand_landmarks.landmark[20]
            pinky_base = hand_landmarks.landmark[17]

            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))

            frame_points.extend([thumb_coords, index_coords])

            pinch_dist = np.hypot(index_coords[0] - thumb_coords[0], index_coords[1] - thumb_coords[1])
            if pinch_dist < 40:
                pinch_center = ((thumb_coords[0] + index_coords[0]) // 2,
                                (thumb_coords[1] + index_coords[1]) // 2)
                current_pinches.append(pinch_center)


            if not countdown_active:
                if pinky_tip.y * h < pinky_base.y * h - 20:
                    pinky_raised = True


    if pinky_raised and not countdown_active:
        countdown_active = True
        countdown_start_time = time.time()


    if countdown_active:
        elapsed = time.time() - countdown_start_time
        if elapsed < 2.0:
            countdown_text = str(2 - int(elapsed))
            cv2.putText(output, countdown_text, (w//2 - 30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
        else:
            if len(frame_points) >= 2:
                if len(frame_points) == 2:
                    pts = np.array(frame_points)
                elif len(frame_points) == 4:
                    pts = order_points(frame_points)
                else:
                    pts = np.array(frame_points[:2])

                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

                if x_max > x_min and y_max > y_min:
                    captured = frame[y_min:y_max, x_min:x_max].copy()
                    if captured.size > 0:
                        thumb_h, thumb_w = captured.shape[:2]
                        max_size = 200
                        if max(thumb_w, thumb_h) > max_size:
                            scale = max_size / max(thumb_w, thumb_h)
                            new_w = int(thumb_w * scale)
                            new_h = int(thumb_h * scale)
                            captured = cv2.resize(captured, (new_w, new_h))
                            thumb_w, thumb_h = new_w, new_h

                        
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        start_x = center_x - thumb_w // 2
                        start_y = center_y - thumb_h // 2

                        start_x = max(0, min(start_x, w - thumb_w))
                        start_y = max(0, min(start_y, h - thumb_h))

                        screenshot_thumbnails.append([captured, start_x, start_y, thumb_w, thumb_h])
            countdown_active = False

    if len(frame_points) == 2:
        cv2.rectangle(output, frame_points[0], frame_points[1], (0, 255, 0), 2)
    elif len(frame_points) == 4:
        pts = order_points(frame_points).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], True, (0, 255, 0), 2)


    if dragging_index is None and current_pinches and not countdown_active:
        for i, (img_thumb, x, y, tw, th) in enumerate(screenshot_thumbnails):
            for (px, py) in current_pinches:
                if point_in_rect(px, py, x, y, x + tw, y + th):
                    dragging_index = i
                    break
            if dragging_index is not None:
                break


    if dragging_index is not None and current_pinches:
        img_thumb, _, _, tw, th = screenshot_thumbnails[dragging_index]
        px, py = current_pinches[0]
        new_x = px - tw // 2
        new_y = py - th // 2
        new_x = max(0, min(new_x, w - tw))
        new_y = max(0, min(new_y, h - th))
        screenshot_thumbnails[dragging_index][1] = new_x
        screenshot_thumbnails[dragging_index][2] = new_y


        t_x1, t_y1 = new_x, new_y
        t_x2, t_y2 = new_x + tw, new_y + th
        b_x1, b_y1, b_x2, b_y2 = bin_area

        if not (t_x2 < b_x1 or t_x1 > b_x2 or t_y2 < b_y1 or t_y1 > b_y2):
            screenshot_thumbnails.pop(dragging_index)
            dragging_index = None
            return output

    elif dragging_index is not None and not current_pinches:
        dragging_index = None


    for img_thumb, x, y, tw, th in screenshot_thumbnails:
        if tw > 0 and th > 0 and 0 <= x < w and 0 <= y < h:
            end_x = min(x + tw, w)
            end_y = min(y + th, h)
            roi_w = end_x - x
            roi_h = end_y - y
            if roi_w > 0 and roi_h > 0:
                output[y:end_y, x:end_x] = img_thumb[:roi_h, :roi_w]

    return output


def draw_mode_buttons(img, selected_mode):
    h, w, _ = img.shape
    button_radius = int(min(w, h) * 0.06)
    y = int(h * 0.92)
    x1 = int(w * 0.25)
    x2 = int(w * 0.5)
    x3 = int(w * 0.75)
    color = (180, 0, 180)
    highlight = (255, 0, 255)
    thickness = -1
    cv2.circle(img, (x1, y), button_radius, highlight if selected_mode == 1 else color, thickness)
    cv2.circle(img, (x2, y), button_radius, highlight if selected_mode == 2 else color, thickness)
    cv2.circle(img, (x3, y), button_radius, highlight if selected_mode == 3 else color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = button_radius / 40.0
    font_thickness = 3
    text_color = (255, 255, 255)
    for i, x in enumerate([x1, x2, x3], 1):
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.putText(img, text, (x - tw // 2, y + th // 2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return [(x1, y, button_radius), (x2, y, button_radius), (x3, y, button_radius)]

def detect_pinch_and_button(results, w, h, buttons):
    if not results.multi_hand_landmarks:
        return None
    for handLms in results.multi_hand_landmarks:
        thumb_tip = handLms.landmark[4]
        index_tip = handLms.landmark[8]
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
        dist = np.hypot(x2 - x1, y2 - y1)
        pinch_thresh = 40
        if dist < pinch_thresh:
            pinch_x = int((x1 + x2) / 2)
            pinch_y = int((y1 + y2) / 2)
            for i, (bx, by, br) in enumerate(buttons, 1):
                if (pinch_x - bx) ** 2 + (pinch_y - by) ** 2 < br ** 2:
                    return i
    return None


def main():
    global screenshot_thumbnails, dragging_index, countdown_active

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

        buttons = draw_mode_buttons(img.copy(), selected_mode)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        pinch_option = detect_pinch_and_button(results, w, h, buttons)
        now = time.time()
        if pinch_option is not None and pinch_option != selected_mode and (now - last_switch_time) > switch_cooldown:
            selected_mode = pinch_option
            last_switch_time = now

            if selected_mode != 3:
                screenshot_thumbnails = []
                dragging_index = None
                countdown_active = False

        if selected_mode == 1:
            output = run_main_mode(img, hands, mp_hands, mp_draw)
        elif selected_mode == 2:
            output = run_inverted_colours_mode(img, hands, mp_hands, mp_draw)
        elif selected_mode == 3:
            output = run_screenshot_canvas_mode(img, hands, mp_hands, mp_draw)
        else:
            output = img

        draw_mode_buttons(output, selected_mode)
        cv2.imshow("Finger Window", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
