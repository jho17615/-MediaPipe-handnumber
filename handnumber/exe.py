import cv2
import mediapipe as mp
import numpy as np
import joblib

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def extract_features(multi_hand_landmarks):
    features = []
    for hand_landmarks in multi_hand_landmarks[:2]:  # 최대 2개의 손만 처리
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

    # 부족한 특징을 0으로 채움
    while len(features) < 126:
        features.append(0)

    # 126개의 특징으로 제한
    return features[:126]


def recognize_gesture(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            features = extract_features(results.multi_hand_landmarks)

            # 모델 예측
            gesture = model.predict([features])[0]
            confidence = np.max(model.predict_proba([features]))

            # 손 랜드마크 그리기
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            cv2.putText(image, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_filename = "hand_gesture_model.joblib"

    try:
        print(f"'{model_filename}' 모델을 로딩 중...")
        loaded_data = joblib.load(model_filename)
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            model = loaded_data['model']
            accuracy = loaded_data.get('accuracy', 'Unknown')
        else:
            model = loaded_data
            accuracy = 'Unknown'
        print(f"모델 로드 완료. 모델 정확도: {accuracy}")
    except FileNotFoundError:
        print(f"'{model_filename}' 파일을 찾을 수 없습니다. 프로그램을 종료합니다.")
        exit()
    except Exception as e:
        print(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
        exit()

    print("실시간 인식을 시작합니다. ESC를 눌러 종료하세요.")
    recognize_gesture(model)