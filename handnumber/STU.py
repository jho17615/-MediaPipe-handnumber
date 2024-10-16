import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_data(data_dir):
    data = []
    labels = []
    for gesture in range(10):  # 0부터 9까지
        gesture_dir = os.path.join(data_dir, str(gesture))
        if not os.path.exists(gesture_dir):
            print(f"Warning: Directory for gesture {gesture} not found.")
            continue

        print(f"Loading data for gesture {gesture}...")
        for file_name in os.listdir(gesture_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(gesture_dir, file_name), 'r') as f:
                    sample = json.load(f)
                    data.append(sample)
                    labels.append(gesture)

    return np.array(data), np.array(labels)


def train_model(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    train_scores = []
    test_scores = []

    print("모델 훈련 시작...")
    for i in tqdm(range(1, 101), desc="Training Progress"):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        if i % 10 == 0:
            print(f"Trees: {i}, Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")

    # 최종 성능 평가
    final_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_pred)
    print("\n최종 모델 성능:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, final_pred))

    # 성능 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), train_scores, label='Training Accuracy')
    plt.plot(range(1, 101), test_scores, label='Testing Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.legend()
    plt.savefig('model_performance.png')
    plt.show()

    return model, final_accuracy


def save_model(model, accuracy, filename='hand_gesture_model.joblib'):
    joblib.dump({'model': model, 'accuracy': accuracy}, filename)
    print(f"Model saved as {filename}")


if __name__ == "__main__":
    data_dir = "data"  # 데이터가 저장된 디렉토리

    print("데이터 로딩 중...")
    data, labels = load_data(data_dir)

    if len(data) > 0:
        print(f"데이터 로딩 완료. 총 샘플 수: {len(data)}")
        print("모델 훈련을 시작합니다.")

        model, accuracy = train_model(data, labels)

        save_model(model, accuracy)

        print("훈련 완료. 모델이 저장되었으며 성능 그래프가 'model_performance.png' 파일로 저장되었습니다.")
    else:
        print("로드된 데이터가 없습니다. 프로그램을 종료합니다.")