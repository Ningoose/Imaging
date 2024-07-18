import os
import numpy as np
import torch
from PIL import Image, ImageTk
# from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

def load_dataset(dataset_dir, target_size, batch_size=64):
    images = []
    labels = []
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            img_paths = [os.path.join(class_dir, img_file) for img_file in os.listdir(class_dir)]
            for batch_start in range(0, len(img_paths), batch_size):
                batch_img_paths = img_paths[batch_start:batch_start + batch_size]
                batch_images = []
                for img_path in batch_img_paths:
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize(target_size)
                        batch_images.append(np.array(img, dtype=np.float64))
                        labels.append(class_name)
                    except Exception as e:
                        print(f"加载图像失败： {img_path}: {e}")
                images.append(np.array(batch_images))
    images = np.concatenate(images, axis=0)
    print(f"加载了 {len(images)} 个图像")
    return images, np.array(labels)


def load_test_dataset(test_dataset_dir, target_size):
    test_images = []
    test_filenames = []
    for img_file in os.listdir(test_dataset_dir):
        try:
            img_path = os.path.join(test_dataset_dir, img_file)
            img = Image.open(img_path).convert('L')
            img = img.resize(target_size)
            test_images.append(np.array(img, dtype=np.float64))
            test_filenames.append(img_file)
        except Exception as e:
            print(f"加载图像失败： {img_path}: {e}")
    test_images = np.array(test_images)
    print(f"加载了 {len(test_images)} 个测试图像")
    return test_images, test_filenames


class Lasso:
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-3):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def soft_threshold(self, value, threshold):
        if value > threshold:
            return value - threshold
        elif value < -threshold:
            return value + threshold
        else:
            return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for _ in range(self.max_iter):
            gradients = (X.T @ (X @ self.coef_ - y)) / n_samples + self.alpha * np.sign(self.coef_)

            # 应用软阈值运算符
            self.coef_ = np.vectorize(self.soft_threshold)(self.coef_, self.alpha)

            # 收敛性检查
            if np.linalg.norm(gradients) <= self.tol:
                break

    def predict(self, X):
        return X @ self.coef_

def create_gui(model, scaler, selected_features, target_size):
    def select_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path).convert('L')
            image = image.resize(target_size)
            image_array = np.array(image, dtype=np.float64)
            image_flatten = image_array.reshape(1, -1)
            image_scaled = scaler.transform(image_flatten)
            image_selected = image_scaled[:, selected_features]
            image_tensor = torch.tensor(image_selected, dtype=torch.float64)

            model.eval()
            with torch.no_grad():
                pred_prob = torch.sigmoid(model(image_tensor.double())).squeeze().numpy()
                pred = "患者 (乳腺癌)" if pred_prob > 0.5 else "健康 (没有乳腺癌)"
                confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

            result_label.config(text=f"预测结果: {pred}\n可信度: {confidence:.2%}")

            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo

    root = tk.Tk()
    root.title("Breast Cancer MRI Classifier")

    select_button = tk.Button(root, text="选择图像", command=select_image)
    select_button.pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root, text="")
    result_label.pack(pady=10)

    root.mainloop()


class DeepMIPNN(nn.Module):
    def __init__(self, input_size):
        super(DeepMIPNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc6(x)
        return x

def main():
    dataset_dir = r'F:\\Pycharm\\Tuxiang\\image'
    test_dataset_dir = r'F:\\Pycharm\\Tuxiang\\validation'
    target_size = (300, 300)
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001

    X, y = load_dataset(dataset_dir, target_size, batch_size)
    print(f"图像数量: {len(X)}, 标签数量: {len(y)}")
    print(f"自定义标签: {np.unique(y)}")

    if len(X) == 0 or len(y) == 0:
        print("没有训练数据，请检查数据集位置以及加载代码.")
        return

    y = np.where(y == 'patient', 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flatten.astype(np.float64))
    X_test_scaled = scaler.transform(X_test_flatten.astype(np.float64))

    # lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_train_scaled, y_train)
    # best_alpha = lasso_cv.alpha_
    # print(f"最佳alpha: {best_alpha}")
    # 用最佳alpha调整Lasso
    # lasso_model = Lasso(alpha=best_alpha, max_iter=10000)
    # lasso_model.fit(X_train_scaled, y_train)
    lasso_model = Lasso(alpha=0.003, max_iter=2000)
    lasso_model.fit(X_train_scaled, y_train)
    selected_features = np.where(lasso_model.coef_ != 0)[0]
    print(f"已筛选的特征: {selected_features}")


    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float64)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64)


    input_size = len(selected_features)
    model = DeepMIPNN(input_size)
    model.double()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X.double())
            loss = criterion(outputs.squeeze(), batch_y.double())
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


    model.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(X_test_tensor.double())).squeeze().numpy()
        y_pred = (y_pred_prob > 0.5).astype(int)
        test_loss = criterion(torch.tensor(y_pred_prob), y_test_tensor.double())
        print(f"Test Loss: {test_loss.item():.4f}")

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


    test_images, test_filenames = load_test_dataset(test_dataset_dir, target_size)
    test_images_flatten = test_images.reshape(test_images.shape[0], -1)
    test_images_scaled = scaler.transform(test_images_flatten.astype(np.float64))
    test_images_selected = test_images_scaled[:, selected_features]
    test_images_tensor = torch.tensor(test_images_selected, dtype=torch.float64)


    model.eval()
    with torch.no_grad():
        y_pred_prob_test = torch.sigmoid(model(test_images_tensor.double())).squeeze().numpy()
        y_pred_test = (y_pred_prob_test > 0.5).astype(int)


    for filename, pred in zip(test_filenames, y_pred_test):
        print(f"图像: {filename}, 预测结果: {'健康' if pred == 1 else '患者'}")

    create_gui(model, scaler, selected_features, target_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
