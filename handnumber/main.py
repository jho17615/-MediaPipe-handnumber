import cv2
import mediapipe as mp
import numpy as np
import os
import json

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def extract_features(multi_hand_landmarks):
    features = []
    for hand_landmarks in multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

    # 두 번째 손이 없는 경우 0으로 채움
    if len(multi_hand_landmarks) == 1:
        features.extend([0] * 63)  # 21 landmarks * 3 coordinates

    return features


def collect_data(gesture_name, num_samples=1000):
    data = []
    cap = cv2.VideoCapture(0)
    count = 0

    # 숫자별 폴더 생성
    folder_path = f"data/{gesture_name}"
    os.makedirs(folder_path, exist_ok=True)

    two_hands_required = int(gesture_name) >= 6

    while count < num_samples:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            if (two_hands_required and len(results.multi_hand_landmarks) == 2) or (not two_hands_required):
                features = extract_features(results.multi_hand_landmarks)
                data.append(features)

                # 데이터 저장
                with open(f"{folder_path}/sample_{count}.json", 'w') as f:
                    json.dump(features, f)

                # 손 랜드마크 그리기
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 각 랜드마크의 좌표와 번호 표시
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                count += 1
                print(f"Collected {count}/{num_samples} samples for {gesture_name}")

        cv2.putText(image, f"Collecting {gesture_name}: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        if two_hands_required:
            cv2.putText(image, "Please use two hands", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Collect Data', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return data


if __name__ == "__main__":
    gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for gesture in gestures:
        print(f"'{gesture}' 제스처의 데이터 수집을 시작합니다. 1000개의 샘플을 수집합니다.")
        if int(gesture) >= 6:
            print("이 제스처는 두 손을 사용해야 합니다.")
        print("준비되면 Enter를 눌러주세요.")
        input()

        gesture_data = collect_data(gesture, num_samples=1000)

        print(f"'{gesture}' 제스처의 데이터 수집이 완료되었습니다.")

    print("모든 데이터 수집이 완료되었습니다.")


